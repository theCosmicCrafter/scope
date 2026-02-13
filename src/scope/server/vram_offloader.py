"""Smart VRAM Offloader — moves idle pipeline models between GPU and CPU.

When VRAM is tight before loading a new pipeline, the offloader evicts the
least-recently-used idle pipeline to CPU RAM.  When an offloaded pipeline is
needed again it is transparently moved back to GPU.

Video-optimised component-level offloading (WAN architecture):
  - **VAE** is pinned to GPU — small footprint, runs every frame for
    encode + decode, latency-critical for real-time streaming.
  - **Text encoder** (T5) is offloaded to CPU between prompt changes —
    ~6 GB freed, only needed when the user changes the prompt.  The
    ``WanTextEncoderWrapper.output_device`` property is used so embeddings
    land on GPU even when the encoder itself lives on CPU.
  - **Generator** (CausalWanModel / DiT) stays on GPU while the pipeline
    is actively streaming and is offloaded to CPU when the pipeline goes
    idle, freeing the bulk of VRAM.

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


class ComponentRole(Enum):
    """Semantic role of a pipeline component for offload prioritisation."""

    VAE = "vae"  # encode/decode every frame — pin to GPU
    GENERATOR = "generator"  # DiT / CausalWanModel — offload when idle
    TEXT_ENCODER = "text_encoder"  # T5 — offload between prompt changes
    OTHER = "other"  # scheduler, blender, etc. — follow pipeline


@dataclass
class VideoComponentPolicy:
    """Per-component offload rules for video pipelines."""

    role: ComponentRole
    pin_to_gpu: bool = False  # True → never offload this component
    offload_when_idle: bool = True  # offload when pipeline goes idle
    offload_between_uses: bool = False  # offload immediately after use
    priority: int = 0  # lower = offloaded first when freeing VRAM


# WAN-architecture defaults: VAE pinned, text_encoder offloaded aggressively,
# generator offloaded only when the pipeline is idle.
DEFAULT_VIDEO_COMPONENT_POLICIES: dict[ComponentRole, VideoComponentPolicy] = {
    ComponentRole.VAE: VideoComponentPolicy(
        role=ComponentRole.VAE,
        pin_to_gpu=True,
        offload_when_idle=False,
        offload_between_uses=False,
        priority=99,  # never evicted
    ),
    ComponentRole.GENERATOR: VideoComponentPolicy(
        role=ComponentRole.GENERATOR,
        pin_to_gpu=False,
        offload_when_idle=True,
        offload_between_uses=False,
        priority=1,  # evicted second (large, but needed every frame)
    ),
    ComponentRole.TEXT_ENCODER: VideoComponentPolicy(
        role=ComponentRole.TEXT_ENCODER,
        pin_to_gpu=False,
        offload_when_idle=True,
        offload_between_uses=True,  # safe to offload between prompt changes
        priority=0,  # evicted first (only needed on prompt change)
    ),
    ComponentRole.OTHER: VideoComponentPolicy(
        role=ComponentRole.OTHER,
        pin_to_gpu=False,
        offload_when_idle=True,
        offload_between_uses=False,
        priority=2,
    ),
}


@dataclass
class ComponentOffloadRecord:
    """Tracks a single component's device location within a pipeline."""

    component_name: str
    role: ComponentRole
    location: DeviceLocation
    policy: VideoComponentPolicy
    vram_bytes: int = 0  # estimated GPU footprint of this component


@dataclass
class OffloadRecord:
    """Tracks a pipeline's device location and access time."""

    pipeline_id: str
    location: DeviceLocation
    last_accessed: float  # monotonic timestamp
    is_active: bool  # True while pipeline is in a streaming session
    measured_vram_bytes: int  # approximate GPU memory footprint
    components: dict[str, ComponentOffloadRecord] | None = (
        None  # per-component tracking
    )


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
        pipeline: Any = None,
    ) -> None:
        """Register a newly loaded pipeline as on-GPU and active.

        If *pipeline* is provided and has a ``components`` attribute
        (i.e. a ``ComponentsManager``), per-component tracking is
        automatically initialised using the default video policies.
        """
        components = self._detect_components(pipeline) if pipeline else None
        with self._lock:
            self._records[pipeline_id] = OffloadRecord(
                pipeline_id=pipeline_id,
                location=DeviceLocation.GPU,
                last_accessed=time.monotonic(),
                is_active=False,
                measured_vram_bytes=measured_vram_bytes,
                components=components,
            )
        comp_names = list(components.keys()) if components else []
        logger.debug(
            "Offloader: registered %s on GPU (components: %s)",
            pipeline_id,
            comp_names or "whole-pipeline",
        )

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

    def mark_idle(self, pipeline_id: str, pipeline: Any = None) -> None:
        """Mark a pipeline as idle (eligible for offload).

        When a pipeline goes idle, components with ``offload_when_idle``
        policy are immediately moved to CPU to reclaim VRAM.
        """
        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec:
                rec.is_active = False
                rec.last_accessed = time.monotonic()

        # Proactively offload components that should not stay on GPU when idle
        if pipeline is not None and self._cuda_available:
            self._offload_idle_components(pipeline_id, pipeline)

    # ── Component Detection ──────────────────────────────────────

    @staticmethod
    def _detect_components(pipeline: Any) -> dict[str, ComponentOffloadRecord] | None:
        """Inspect a pipeline for WAN-style components and build tracking records.

        Returns ``None`` if the pipeline does not expose a ``ComponentsManager``.
        """
        components_mgr = getattr(pipeline, "components", None)
        if components_mgr is None:
            return None

        # Map known component names to roles
        role_map: dict[str, ComponentRole] = {
            "vae": ComponentRole.VAE,
            "generator": ComponentRole.GENERATOR,
            "text_encoder": ComponentRole.TEXT_ENCODER,
        }

        inner = getattr(components_mgr, "_components", None)
        if not isinstance(inner, dict):
            return None

        records: dict[str, ComponentOffloadRecord] = {}
        for name in inner:
            role = role_map.get(name, ComponentRole.OTHER)
            policy = DEFAULT_VIDEO_COMPONENT_POLICIES.get(
                role, DEFAULT_VIDEO_COMPONENT_POLICIES[ComponentRole.OTHER]
            )
            records[name] = ComponentOffloadRecord(
                component_name=name,
                role=role,
                location=DeviceLocation.GPU,
                policy=policy,
            )
        return records if records else None

    # ── Component-Level Offload / Reload ─────────────────────────

    def _offload_idle_components(self, pipeline_id: str, pipeline: Any) -> None:
        """Offload components that should not stay on GPU when the pipeline is idle."""
        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec is None or rec.components is None:
                return
            targets = [
                (name, crec)
                for name, crec in rec.components.items()
                if crec.policy.offload_when_idle
                and crec.location == DeviceLocation.GPU
                and not crec.policy.pin_to_gpu
            ]
            # Sort by priority (lowest first = offload first)
            targets.sort(key=lambda t: t[1].policy.priority)

        if not targets:
            return

        components_mgr = getattr(pipeline, "components", None)
        if components_mgr is None:
            return
        inner = getattr(components_mgr, "_components", {})

        for name, crec in targets:
            component = inner.get(name)
            if component is None or not hasattr(component, "to"):
                continue
            try:
                logger.info(
                    "Offloader: offloading component %s.%s to CPU (role=%s)",
                    pipeline_id,
                    name,
                    crec.role.value,
                )
                component.to(torch.device("cpu"))
                with self._lock:
                    crec.location = DeviceLocation.CPU
            except Exception as exc:
                logger.warning(
                    "Offloader: failed to offload %s.%s: %s",
                    pipeline_id,
                    name,
                    exc,
                )

        torch.cuda.empty_cache()
        logger.info(
            "Offloader: idle offload complete for %s — freed components: %s",
            pipeline_id,
            [n for n, _ in targets],
        )

    def offload_text_encoder(self, pipeline_id: str, pipeline: Any) -> None:
        """Offload the text encoder to CPU after prompt encoding.

        Call this after the text encoder has produced embeddings.  The
        ``WanTextEncoderWrapper.output_device`` property ensures embeddings
        are already on GPU even when the encoder lives on CPU.
        """
        if not self._cuda_available:
            return
        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec is None or rec.components is None:
                return
            te_rec = rec.components.get("text_encoder")
            if te_rec is None or te_rec.location == DeviceLocation.CPU:
                return
            if te_rec.policy.pin_to_gpu:
                return

        components_mgr = getattr(pipeline, "components", None)
        if components_mgr is None:
            return
        inner = getattr(components_mgr, "_components", {})
        text_encoder = inner.get("text_encoder")
        if text_encoder is None or not hasattr(text_encoder, "to"):
            return

        logger.info("Offloader: offloading text_encoder for %s to CPU", pipeline_id)
        start = time.monotonic()
        text_encoder.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - start

        with self._lock:
            if te_rec:
                te_rec.location = DeviceLocation.CPU
        logger.info(
            "Offloader: text_encoder for %s offloaded to CPU in %.2fs",
            pipeline_id,
            elapsed,
        )

    def ensure_text_encoder_on_gpu(self, pipeline_id: str, pipeline: Any) -> None:
        """Reload the text encoder to GPU before prompt encoding."""
        if not self._cuda_available:
            return
        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec is None or rec.components is None:
                return
            te_rec = rec.components.get("text_encoder")
            if te_rec is None or te_rec.location == DeviceLocation.GPU:
                return

        components_mgr = getattr(pipeline, "components", None)
        if components_mgr is None:
            return
        inner = getattr(components_mgr, "_components", {})
        text_encoder = inner.get("text_encoder")
        if text_encoder is None or not hasattr(text_encoder, "to"):
            return

        logger.info("Offloader: reloading text_encoder for %s to GPU", pipeline_id)
        start = time.monotonic()
        text_encoder.to(self._device)
        elapsed = time.monotonic() - start

        with self._lock:
            if te_rec:
                te_rec.location = DeviceLocation.GPU
        logger.info(
            "Offloader: text_encoder for %s reloaded to GPU in %.2fs",
            pipeline_id,
            elapsed,
        )

    def ensure_components_on_gpu(self, pipeline_id: str, pipeline: Any) -> None:
        """Reload all offloaded components to GPU before streaming.

        Called when a pipeline transitions from idle to active.  Components
        that are pinned to GPU are skipped (they never left).
        """
        if not self._cuda_available:
            return
        with self._lock:
            rec = self._records.get(pipeline_id)
            if rec is None or rec.components is None:
                return
            offloaded = [
                (name, crec)
                for name, crec in rec.components.items()
                if crec.location == DeviceLocation.CPU
            ]

        if not offloaded:
            return

        components_mgr = getattr(pipeline, "components", None)
        if components_mgr is None:
            return
        inner = getattr(components_mgr, "_components", {})

        for name, crec in offloaded:
            component = inner.get(name)
            if component is None or not hasattr(component, "to"):
                continue
            logger.info(
                "Offloader: reloading component %s.%s to GPU",
                pipeline_id,
                name,
            )
            start = time.monotonic()
            try:
                component.to(self._device)
                elapsed = time.monotonic() - start
                with self._lock:
                    crec.location = DeviceLocation.GPU
                logger.info(
                    "Offloader: reloaded %s.%s to GPU in %.2fs",
                    pipeline_id,
                    name,
                    elapsed,
                )
            except torch.cuda.OutOfMemoryError:
                logger.error(
                    "Offloader: OOM reloading %s.%s — skipping",
                    pipeline_id,
                    name,
                )
                break

    # ── Core Offload / Reload ─────────────────────────────────────

    def ensure_on_gpu(
        self,
        pipeline: Any,
        pipeline_id: str,
        pipelines: dict[str, Any] | None = None,
    ) -> Any:
        """Ensure a pipeline's models are on GPU, reloading from CPU if needed.

        For video pipelines with component tracking, this uses
        ``ensure_components_on_gpu`` to reload only the offloaded
        components (respecting pin policies).  For pipelines without
        component tracking it falls back to whole-pipeline ``.to()``.

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
                # Even if the pipeline record says GPU, individual components
                # may have been offloaded — reload them.
                if rec.components:
                    any_offloaded = any(
                        c.location == DeviceLocation.CPU
                        for c in rec.components.values()
                    )
                    if any_offloaded:
                        # Release lock before slow GPU moves
                        pass
                    else:
                        rec.last_accessed = time.monotonic()
                        return pipeline
                else:
                    rec.last_accessed = time.monotonic()
                    return pipeline

        # Component-level reload for video pipelines
        with self._lock:
            rec = self._records.get(pipeline_id)
            has_components = rec is not None and rec.components is not None

        if has_components:
            self.ensure_components_on_gpu(pipeline_id, pipeline)
            with self._lock:
                rec = self._records.get(pipeline_id)
                if rec:
                    rec.location = DeviceLocation.GPU
                    rec.last_accessed = time.monotonic()
            return pipeline

        # Whole-pipeline fallback for non-component pipelines
        logger.info("Offloader: reloading %s to GPU (whole-pipeline)", pipeline_id)
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

        For video pipelines with component tracking, only non-pinned
        components are moved.  The VAE stays on GPU.

        Args:
            pipeline: The pipeline instance.
            pipeline_id: Unique pipeline identifier.

        Returns:
            The pipeline (same object, components on CPU where allowed).
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
            has_components = rec.components is not None

        if has_components:
            # Component-level offload: skip pinned components (VAE)
            self._offload_idle_components(pipeline_id, pipeline)
            with self._lock:
                rec = self._records.get(pipeline_id)
                if rec:
                    rec.location = DeviceLocation.CPU
                    rec.last_accessed = time.monotonic()
            return pipeline

        # Whole-pipeline fallback
        logger.info("Offloader: offloading %s to CPU (whole-pipeline)", pipeline_id)
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
            result = []
            for rec in self._records.values():
                entry: dict[str, Any] = {
                    "pipeline_id": rec.pipeline_id,
                    "location": rec.location.value,
                    "is_active": rec.is_active,
                    "last_accessed": rec.last_accessed,
                    "measured_vram_mb": round(rec.measured_vram_bytes / (1024**2), 1),
                }
                if rec.components:
                    entry["components"] = {
                        name: {
                            "role": crec.role.value,
                            "location": crec.location.value,
                            "pin_to_gpu": crec.policy.pin_to_gpu,
                        }
                        for name, crec in rec.components.items()
                    }
                result.append(entry)
            return result

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
