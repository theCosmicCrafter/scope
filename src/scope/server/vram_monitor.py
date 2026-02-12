"""VRAM Monitor — real-time GPU memory tracking with per-pipeline attribution.

Provides a singleton VRAMMonitor that:
  - Snapshots global GPU memory (total, used, free, reserved)
  - Tracks per-pipeline VRAM deltas at load/unload boundaries
  - Exposes a lightweight status dict for the API layer
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class PipelineMemoryRecord:
    """Memory snapshot for a single loaded pipeline."""

    pipeline_id: str
    estimated_vram_gb: float | None = None  # from schema config
    measured_vram_bytes: int = 0  # delta measured at load time
    loaded_at: float = 0.0  # monotonic timestamp


@dataclass
class GPUSnapshot:
    """Point-in-time GPU memory snapshot."""

    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    reserved_bytes: int = 0
    active_bytes: int = 0
    timestamp: float = 0.0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024**3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024**3)

    @property
    def reserved_gb(self) -> float:
        return self.reserved_bytes / (1024**3)

    @property
    def utilization_pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100.0


class VRAMMonitor:
    """Singleton GPU memory monitor with per-pipeline tracking.

    Thread-safe. All public methods acquire the internal lock.
    """

    _instance: "VRAMMonitor | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "VRAMMonitor":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._lock = threading.Lock()
        self._pipeline_records: dict[str, PipelineMemoryRecord] = {}
        self._cuda_available = torch.cuda.is_available()
        self._device = torch.device("cuda:0") if self._cuda_available else None
        self._last_snapshot = GPUSnapshot()
        self._initialized = True
        logger.info(
            "VRAMMonitor initialized (CUDA %s)",
            "available" if self._cuda_available else "not available",
        )

    # ── GPU Snapshot ──────────────────────────────────────────────

    def snapshot(self) -> GPUSnapshot:
        """Take a point-in-time GPU memory snapshot.

        Returns a GPUSnapshot with zeros if CUDA is not available.
        """
        if not self._cuda_available:
            return GPUSnapshot()

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
            stats = torch.cuda.memory_stats(self._device)
            active = stats.get("active_bytes.all.current", 0)
            reserved = stats.get("reserved_bytes.all.current", 0)

            snap = GPUSnapshot(
                total_bytes=total_bytes,
                used_bytes=total_bytes - free_bytes,
                free_bytes=free_bytes,
                reserved_bytes=reserved,
                active_bytes=active,
                timestamp=time.monotonic(),
            )
            with self._lock:
                self._last_snapshot = snap
            return snap
        except Exception as e:
            logger.warning("VRAMMonitor snapshot failed: %s", e)
            return GPUSnapshot()

    @property
    def last_snapshot(self) -> GPUSnapshot:
        with self._lock:
            return self._last_snapshot

    # ── Per-Pipeline Tracking ─────────────────────────────────────

    def record_pipeline_load(
        self,
        pipeline_id: str,
        vram_delta_bytes: int,
        estimated_vram_gb: float | None = None,
    ) -> None:
        """Record that a pipeline was loaded and how much VRAM it consumed.

        Args:
            pipeline_id: Unique pipeline identifier.
            vram_delta_bytes: Measured VRAM increase during load (bytes).
            estimated_vram_gb: Schema-declared estimate (informational).
        """
        record = PipelineMemoryRecord(
            pipeline_id=pipeline_id,
            estimated_vram_gb=estimated_vram_gb,
            measured_vram_bytes=max(0, vram_delta_bytes),
            loaded_at=time.monotonic(),
        )
        with self._lock:
            self._pipeline_records[pipeline_id] = record
        logger.info(
            "VRAMMonitor: pipeline %s loaded, delta %.2f MB",
            pipeline_id,
            vram_delta_bytes / (1024**2),
        )

    def record_pipeline_unload(self, pipeline_id: str) -> None:
        """Remove tracking for an unloaded pipeline."""
        with self._lock:
            removed = self._pipeline_records.pop(pipeline_id, None)
        if removed:
            logger.info("VRAMMonitor: pipeline %s unloaded", pipeline_id)

    def get_pipeline_records(self) -> dict[str, PipelineMemoryRecord]:
        """Return a copy of all pipeline memory records."""
        with self._lock:
            return dict(self._pipeline_records)

    # ── Aggregate Helpers ─────────────────────────────────────────

    def total_pipeline_vram_bytes(self) -> int:
        """Sum of measured VRAM across all loaded pipelines."""
        with self._lock:
            return sum(r.measured_vram_bytes for r in self._pipeline_records.values())

    def total_pipeline_vram_gb(self) -> float:
        return self.total_pipeline_vram_bytes() / (1024**3)

    # ── Budget / Chain Estimation ───────────────────────────────────

    def estimate_chain_vram_gb(self, pipeline_ids: list[str]) -> float:
        """Estimate total VRAM needed for a pipeline chain.

        Uses measured VRAM for already-loaded pipelines and schema estimates
        for unloaded ones.  Returns 0.0 if no estimates are available.

        Args:
            pipeline_ids: Ordered list of pipeline IDs in the chain.
        """
        total = 0.0
        records = self.get_pipeline_records()

        for pid in pipeline_ids:
            rec = records.get(pid)
            if rec and rec.measured_vram_bytes > 0:
                total += rec.measured_vram_bytes / (1024**3)
            else:
                # Try schema estimate for unloaded pipelines
                try:
                    from scope.core.pipelines.registry import PipelineRegistry

                    pipeline_class = PipelineRegistry.get(pid)
                    if pipeline_class is not None:
                        config_class = pipeline_class.get_config_class()
                        est = getattr(config_class, "estimated_vram_gb", None)
                        if est is not None:
                            total += est
                except Exception:
                    pass
        return total

    def can_fit_chain(
        self,
        pipeline_ids: list[str],
        safety_margin_gb: float = 1.0,
    ) -> tuple[bool, str]:
        """Check whether a pipeline chain can fit in available VRAM.

        Args:
            pipeline_ids: Pipeline IDs to check.
            safety_margin_gb: Extra headroom to reserve beyond the estimate.

        Returns:
            (fits, message) — fits is True if the chain should fit,
            message explains the reasoning.
        """
        if not self._cuda_available:
            return True, "No GPU — running on CPU"

        snap = self.snapshot()
        total_gb = snap.total_gb
        estimated_gb = self.estimate_chain_vram_gb(pipeline_ids)

        if estimated_gb == 0.0:
            return True, "No VRAM estimates available — allowing load"

        # Already-loaded pipelines don't need additional VRAM
        records = self.get_pipeline_records()
        already_loaded_gb = sum(
            rec.measured_vram_bytes / (1024**3)
            for pid in pipeline_ids
            if (rec := records.get(pid)) is not None and rec.measured_vram_bytes > 0
        )
        new_vram_needed = estimated_gb - already_loaded_gb

        if new_vram_needed <= 0:
            return True, "All pipelines already loaded"

        available = snap.free_gb
        fits = available >= (new_vram_needed + safety_margin_gb)

        if fits:
            msg = (
                f"Chain needs ~{new_vram_needed:.1f} GB new VRAM, "
                f"{available:.1f} GB free (total estimate {estimated_gb:.1f} GB)"
            )
        else:
            msg = (
                f"Chain needs ~{new_vram_needed:.1f} GB new VRAM but only "
                f"{available:.1f} GB free ({total_gb:.1f} GB total). "
                f"Offloader will attempt to free space."
            )
        return fits, msg

    # ── Status Dict (for API) ─────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return a JSON-serialisable status dict for the API layer."""
        snap = self.snapshot()
        records = self.get_pipeline_records()

        pipelines_info = []
        for pid, rec in records.items():
            pipelines_info.append(
                {
                    "pipeline_id": pid,
                    "measured_vram_mb": round(rec.measured_vram_bytes / (1024**2), 1),
                    "estimated_vram_gb": rec.estimated_vram_gb,
                    "loaded_at": rec.loaded_at,
                }
            )

        return {
            "cuda_available": self._cuda_available,
            "gpu": {
                "total_gb": round(snap.total_gb, 2),
                "used_gb": round(snap.used_gb, 2),
                "free_gb": round(snap.free_gb, 2),
                "reserved_gb": round(snap.reserved_gb, 2),
                "utilization_pct": round(snap.utilization_pct, 1),
            },
            "pipelines": pipelines_info,
            "total_pipeline_vram_gb": round(self.total_pipeline_vram_gb(), 2),
            "timestamp": snap.timestamp,
        }


# Module-level convenience accessor
def get_vram_monitor() -> VRAMMonitor:
    """Get the singleton VRAMMonitor instance."""
    return VRAMMonitor()
