"""Smart VRAM Offloader — moves idle pipeline models between GPU and CPU.

When VRAM is tight before loading a new pipeline, the offloader evicts the
least-recently-used idle pipeline to CPU RAM.  When an offloaded pipeline is
needed again it is transparently moved back to GPU.

Design constraints:
  - Pipelines are ``torch.nn.Module`` subclasses → ``.to(device)`` works.
  - Only *idle* pipelines (not currently streaming) are eligible for offload.
  - The offloader never deletes a pipeline — it only moves it between devices.
  - Thread-safe: all public methods acquire the internal lock.
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .vram_monitor import get_vram_monitor

logger = logging.getLogger(__name__)

# Minimum free VRAM (bytes) to maintain after offloading.
# If free VRAM drops below this threshold, offloading is triggered.
DEFAULT_VRAM_HEADROOM_GB = 1.5


class DeviceLocation(Enum):
    """Where a pipeline's model weights currently reside."""

    GPU = "gpu"
    CPU = "cpu"


@dataclass
class OffloadRecord:
    """Tracks a pipeline's device location and access time."""

    pipeline_id: str
    location: DeviceLocation
    last_accessed: float  # monotonic timestamp
    is_active: bool  # True while pipeline is in a streaming session
    measured_vram_bytes: int  # approximate GPU memory footprint


class VRAMOffloader:
    """Manages pipeline model placement across GPU and CPU.

    Integrates with PipelineManager to transparently offload/reload models.
    """

    def __init__(self, headroom_gb: float = DEFAULT_VRAM_HEADROOM_GB):
        self._lock = threading.Lock()
        self._records: dict[str, OffloadRecord] = {}
        self._headroom_bytes = int(headroom_gb * (1024**3))
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._cuda_available = torch.cuda.is_available()
        logger.info(
            "VRAMOffloader initialized (headroom=%.1f GB, CUDA %s)",
            headroom_gb,
            "available" if self._cuda_available else "not available",
        )

    # ── Registration ──────────────────────────────────────────────

    def register_pipeline(
        self,
        pipeline_id: str,
        measured_vram_bytes: int = 0,
    ) -> None:
        """Register a newly loaded pipeline as on-GPU and active."""
        with self._lock:
            self._records[pipeline_id] = OffloadRecord(
                pipeline_id=pipeline_id,
                location=DeviceLocation.GPU,
                last_accessed=time.monotonic(),
                is_active=False,
                measured_vram_bytes=measured_vram_bytes,
            )
        logger.debug("Offloader: registered %s on GPU", pipeline_id)

    def unregister_pipeline(self, pipeline_id: str) -> None:
        """Remove tracking for an unloaded pipeline."""
        with self._lock:
            self._records.pop(pipeline_id, None)
        logger.debug("Offloader: unregistered %s", pipeline_id)

    # ── Active / Idle Marking ─────────────────────────────────────

    def mark_active(self, pipeline_id: str) -> None:
        """Mark a pipeline as actively streaming (not eligible for offload)."""
        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec:
                rec.is_active = True
                rec.last_accessed = time.monotonic()

    def mark_idle(self, pipeline_id: str) -> None:
        """Mark a pipeline as idle (eligible for offload)."""
        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec:
                rec.is_active = False
                rec.last_accessed = time.monotonic()

    # ── Core Offload / Reload ─────────────────────────────────────

    def ensure_on_gpu(
        self,
        pipeline: Any,
        pipeline_id: str,
        pipelines: dict[str, Any] | None = None,
    ) -> Any:
        """Ensure a pipeline's models are on GPU, reloading from CPU if needed.

        Args:
            pipeline: The pipeline instance (torch.nn.Module subclass).
            pipeline_id: Unique pipeline identifier.
            pipelines: Optional dict of all loaded pipelines (needed for OOM
                       eviction — without it, eviction cannot move weights).

        Returns:
            The pipeline (same object, now on GPU).
        """
        if not self._cuda_available:
            return pipeline

        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec is None:
                return pipeline
            if rec.location == DeviceLocation.GPU:
                rec.last_accessed = time.monotonic()
                return pipeline

        # Outside lock — the .to() call can be slow
        logger.info("Offloader: reloading %s to GPU", pipeline_id)
        start = time.monotonic()
        try:
            pipeline.to(self._device)
            elapsed = time.monotonic() - start
            logger.info("Offloader: reloaded %s to GPU in %.2fs", pipeline_id, elapsed)
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "Offloader: OOM reloading %s to GPU — attempting eviction first",
                pipeline_id,
            )
            # Evict an idle pipeline and actually move its weights to CPU
            evicted_pid = self._evict_one_idle(exclude={pipeline_id})
            if evicted_pid and pipelines and evicted_pid in pipelines:
                evicted = pipelines[evicted_pid]
                evicted.to(torch.device("cpu"))
                torch.cuda.empty_cache()
                logger.info(
                    "Offloader: evicted %s to CPU for OOM recovery", evicted_pid
                )
            pipeline.to(self._device)

        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec:
                rec.location = DeviceLocation.GPU
                rec.last_accessed = time.monotonic()

        return pipeline

    def offload_to_cpu(self, pipeline: Any, pipeline_id: str) -> Any:
        """Move a pipeline's models to CPU to free VRAM.

        Args:
            pipeline: The pipeline instance.
            pipeline_id: Unique pipeline identifier.

        Returns:
            The pipeline (same object, now on CPU).
        """
        if not self._cuda_available:
            return pipeline

        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec is None or rec.location == DeviceLocation.CPU:
                return pipeline
            if rec.is_active:
                logger.warning(
                    "Offloader: refusing to offload active pipeline %s", pipeline_id
                )
                return pipeline

        logger.info("Offloader: offloading %s to CPU", pipeline_id)
        start = time.monotonic()
        pipeline.to(torch.device("cpu"))
        elapsed = time.monotonic() - start

        # Free CUDA cache after moving to CPU
        torch.cuda.empty_cache()

        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec:
                rec.location = DeviceLocation.CPU
                rec.last_accessed = time.monotonic()

        logger.info("Offloader: offloaded %s to CPU in %.2fs", pipeline_id, elapsed)
        return pipeline

    # ── Automatic Eviction ────────────────────────────────────────

    def ensure_headroom(
        self,
        needed_bytes: int = 0,
        pipelines: dict[str, Any] | None = None,
    ) -> bool:
        """Evict idle pipelines until enough VRAM headroom exists.

        Call this *before* loading a new pipeline to make room.

        Args:
            needed_bytes: Additional VRAM needed beyond the headroom threshold.
            pipelines: Dict of pipeline_id -> pipeline instance (needed to call .to()).

        Returns:
            True if enough headroom was achieved, False if not possible.
        """
        if not self._cuda_available or pipelines is None:
            return True

        target_free = self._headroom_bytes + needed_bytes

        # Check current free VRAM
        snap = get_vram_monitor().snapshot()
        if snap.free_bytes >= target_free:
            return True

        logger.info(
            "Offloader: need %.1f MB free, have %.1f MB — evicting idle pipelines",
            target_free / (1024**2),
            snap.free_bytes / (1024**2),
        )

        # Get eviction candidates: idle pipelines on GPU, sorted by LRU
        with self._lock:
            candidates = [
                rec
                for rec in self._records.values()
                if rec.location == DeviceLocation.GPU and not rec.is_active
            ]
            candidates.sort(key=lambda r: r.last_accessed)

        for candidate in candidates:
            pid = candidate.pipeline_id
            pipeline = pipelines.get(pid)
            if pipeline is None:
                continue

            self.offload_to_cpu(pipeline, pid)

            # Re-check free VRAM
            gc.collect()
            snap = get_vram_monitor().snapshot()
            if snap.free_bytes >= target_free:
                logger.info(
                    "Offloader: headroom achieved (%.1f MB free)",
                    snap.free_bytes / (1024**2),
                )
                return True

        # Couldn't free enough
        snap = get_vram_monitor().snapshot()
        logger.warning(
            "Offloader: could not achieve headroom (%.1f MB free, needed %.1f MB)",
            snap.free_bytes / (1024**2),
            target_free / (1024**2),
        )
        return False

    def _evict_one_idle(self, exclude: set[str] | None = None) -> str | None:
        """Evict the least-recently-used idle pipeline from GPU.

        This is a fallback used during OOM recovery. It does NOT call .to()
        directly — the caller must handle that.

        Returns:
            pipeline_id of the evicted pipeline, or None if nothing to evict.
        """
        with self._lock:
            candidates = [
                rec
                for rec in self._records.values()
                if (
                    rec.location == DeviceLocation.GPU
                    and not rec.is_active
                    and (exclude is None or rec.pipeline_id not in exclude)
                )
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda r: r.last_accessed)
            target = candidates[0]
            target.location = DeviceLocation.CPU
            return target.pipeline_id

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> list[dict[str, Any]]:
        """Return offloader status for each tracked pipeline."""
        with self._lock:
            return [
                {
                    "pipeline_id": rec.pipeline_id,
                    "location": rec.location.value,
                    "is_active": rec.is_active,
                    "last_accessed": rec.last_accessed,
                    "measured_vram_mb": round(rec.measured_vram_bytes / (1024**2), 1),
                }
                for rec in self._records.values()
            ]

    def get_gpu_pipeline_ids(self) -> list[str]:
        """Return IDs of pipelines currently on GPU."""
        with self._lock:
            return [
                rec.pipeline_id
                for rec in self._records.values()
                if rec.location == DeviceLocation.GPU
            ]

    def get_cpu_pipeline_ids(self) -> list[str]:
        """Return IDs of pipelines currently offloaded to CPU."""
        with self._lock:
            return [
                rec.pipeline_id
                for rec in self._records.values()
                if rec.location == DeviceLocation.CPU
            ]


# Module-level singleton
_offloader: VRAMOffloader | None = None
_offloader_lock = threading.Lock()


def get_vram_offloader() -> VRAMOffloader:
    """Get the singleton VRAMOffloader instance."""
    global _offloader
    with _offloader_lock:
        if _offloader is None:
            _offloader = VRAMOffloader()
        return _offloader
