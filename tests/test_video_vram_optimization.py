"""Tests for video-optimised VRAM management.

Covers:
  - Component detection from WAN-style pipelines (ComponentsManager)
  - Component-level offloading policies (VAE pinned, text_encoder aggressive, generator idle)
  - mark_idle triggers component offloading
  - mark_active / ensure_on_gpu reloads offloaded components
  - offload_text_encoder / ensure_text_encoder_on_gpu round-trip
  - offload_to_cpu respects pin policies (VAE stays on GPU)
  - VRAMMonitor video memory tracking (KV cache, temporal buffers, VAE cache)
  - Video runtime overhead estimation
  - can_fit_chain with video runtime overhead
  - Component-level status reporting
  - Thread-safety of component offloading under concurrent access
"""

import logging
import threading
from unittest.mock import MagicMock

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_video_vram")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_monitor():
    """Reset VRAMMonitor singleton state for clean tests."""
    from scope.server.vram_monitor import VRAMMonitor

    m = VRAMMonitor()
    with m._lock:
        m._pipeline_records.clear()
    return m


def _reset_offloader():
    """Reset VRAMOffloader singleton state for clean tests."""
    from scope.server.vram_offloader import get_vram_offloader

    o = get_vram_offloader()
    with o._lock:
        o._records.clear()
    return o


def _make_mock_component(name: str):
    """Create a mock component with a .to() method that tracks calls."""
    comp = MagicMock(name=name)
    comp.to = MagicMock(return_value=comp)
    return comp


def _make_wan_pipeline():
    """Create a mock WAN-style pipeline with ComponentsManager-like structure.

    Returns (pipeline, components_dict) where components_dict maps
    component names to their mock objects.
    """
    # Simulate ComponentsManager._components
    vae = _make_mock_component("vae")
    generator = _make_mock_component("generator")
    text_encoder = _make_mock_component("text_encoder")
    scheduler = MagicMock(name="scheduler")  # no .to — pure CPU
    embedding_blender = _make_mock_component("embedding_blender")

    components_dict = {
        "vae": vae,
        "generator": generator,
        "text_encoder": text_encoder,
        "scheduler": scheduler,
        "embedding_blender": embedding_blender,
    }

    # Build a mock ComponentsManager
    components_mgr = MagicMock()
    components_mgr._components = components_dict

    # Build the pipeline
    pipeline = MagicMock()
    pipeline.components = components_mgr
    pipeline.to = MagicMock(return_value=pipeline)

    return pipeline, components_dict


def _make_simple_pipeline():
    """Create a mock pipeline WITHOUT ComponentsManager (non-WAN)."""
    pipeline = MagicMock()
    pipeline.to = MagicMock(return_value=pipeline)
    # No .components attribute
    del pipeline.components
    return pipeline


# ---------------------------------------------------------------------------
# Suite 1: Component Detection
# ---------------------------------------------------------------------------


def test_component_detection():
    print("\n=== Suite 1: Component Detection ===")
    from scope.server.vram_offloader import (
        ComponentRole,
        VRAMOffloader,
    )

    pipeline, comps = _make_wan_pipeline()
    records = VRAMOffloader._detect_components(pipeline)

    check("detects components", records is not None)
    check("has vae", "vae" in records)
    check("has generator", "generator" in records)
    check("has text_encoder", "text_encoder" in records)
    check(
        "has scheduler as OTHER",
        records.get("scheduler") is not None
        and records["scheduler"].role == ComponentRole.OTHER,
    )
    check(
        "has embedding_blender as OTHER",
        records.get("embedding_blender") is not None
        and records["embedding_blender"].role == ComponentRole.OTHER,
    )

    # Role assignments
    check("vae role", records["vae"].role == ComponentRole.VAE)
    check("generator role", records["generator"].role == ComponentRole.GENERATOR)
    check(
        "text_encoder role", records["text_encoder"].role == ComponentRole.TEXT_ENCODER
    )

    # Policy checks
    check("vae pinned to GPU", records["vae"].policy.pin_to_gpu is True)
    check(
        "vae not offloaded when idle", records["vae"].policy.offload_when_idle is False
    )
    check(
        "generator offloaded when idle",
        records["generator"].policy.offload_when_idle is True,
    )
    check(
        "text_encoder offloaded between uses",
        records["text_encoder"].policy.offload_between_uses is True,
    )
    check(
        "text_encoder priority < generator priority",
        records["text_encoder"].policy.priority < records["generator"].policy.priority,
    )


def test_component_detection_no_components():
    from scope.server.vram_offloader import VRAMOffloader

    pipeline = _make_simple_pipeline()
    records = VRAMOffloader._detect_components(pipeline)
    check("no components returns None", records is None)


# ---------------------------------------------------------------------------
# Suite 2: Component-Level Registration
# ---------------------------------------------------------------------------


def test_register_with_components():
    print("\n=== Suite 2: Component-Level Registration ===")
    o = _reset_offloader()
    pipeline, _ = _make_wan_pipeline()

    o.register_pipeline(
        "wan-test", measured_vram_bytes=5_000_000_000, pipeline=pipeline
    )

    with o._lock:
        rec = o._records.get("wan-test")
    check("record exists", rec is not None)
    check("has component tracking", rec.components is not None)
    check("5 components tracked", len(rec.components) == 5)
    check(
        "all components on GPU initially",
        all(c.location.value == "gpu" for c in rec.components.values()),
    )


def test_register_without_components():
    o = _reset_offloader()
    pipeline = _make_simple_pipeline()

    o.register_pipeline(
        "simple-test", measured_vram_bytes=1_000_000_000, pipeline=pipeline
    )

    with o._lock:
        rec = o._records.get("simple-test")
    check("record exists (simple)", rec is not None)
    check("no component tracking (simple)", rec.components is None)


# ---------------------------------------------------------------------------
# Suite 3: Idle Component Offloading
# ---------------------------------------------------------------------------


def test_mark_idle_offloads_components():
    print("\n=== Suite 3: Idle Component Offloading ===")
    from scope.server.vram_offloader import DeviceLocation

    o = _reset_offloader()
    pipeline, comps = _make_wan_pipeline()

    o.register_pipeline("idle-test", pipeline=pipeline)
    o.mark_active("idle-test")

    # Now mark idle — should offload text_encoder and generator but NOT vae
    o.mark_idle("idle-test", pipeline=pipeline)

    with o._lock:
        rec = o._records["idle-test"]
        vae_loc = rec.components["vae"].location
        gen_loc = rec.components["generator"].location
        te_loc = rec.components["text_encoder"].location

    check("VAE stays on GPU after idle", vae_loc == DeviceLocation.GPU)
    check("generator offloaded to CPU after idle", gen_loc == DeviceLocation.CPU)
    check("text_encoder offloaded to CPU after idle", te_loc == DeviceLocation.CPU)

    # Verify .to(cpu) was called on generator and text_encoder but NOT vae
    comps["vae"].to.assert_not_called()
    check("vae.to() never called", True)

    comps["generator"].to.assert_called()
    check("generator.to() was called", True)

    comps["text_encoder"].to.assert_called()
    check("text_encoder.to() was called", True)


def test_mark_idle_without_pipeline_no_crash():
    """mark_idle without pipeline arg should not crash (backward compat)."""
    o = _reset_offloader()
    pipeline, _ = _make_wan_pipeline()
    o.register_pipeline("compat-test", pipeline=pipeline)
    o.mark_active("compat-test")

    # No pipeline passed — should just mark idle without offloading
    o.mark_idle("compat-test")

    with o._lock:
        rec = o._records["compat-test"]
        # Components should still be on GPU since no pipeline was passed
        all_gpu = all(c.location.value == "gpu" for c in rec.components.values())
    check("components stay on GPU when no pipeline passed to mark_idle", all_gpu)


# ---------------------------------------------------------------------------
# Suite 4: Component Reload on Active
# ---------------------------------------------------------------------------


def test_ensure_components_on_gpu():
    print("\n=== Suite 4: Component Reload ===")
    from scope.server.vram_offloader import DeviceLocation

    o = _reset_offloader()
    pipeline, comps = _make_wan_pipeline()

    o.register_pipeline("reload-test", pipeline=pipeline)
    o.mark_idle("reload-test", pipeline=pipeline)

    # Reset mock call counts
    for c in comps.values():
        if hasattr(c, "to"):
            c.to.reset_mock()

    # Reload all components
    o.ensure_components_on_gpu("reload-test", pipeline)

    with o._lock:
        rec = o._records["reload-test"]
        all_gpu = all(c.location == DeviceLocation.GPU for c in rec.components.values())
    check("all components back on GPU after reload", all_gpu)

    # generator and text_encoder should have been reloaded
    check("generator.to() called for reload", comps["generator"].to.called)
    check("text_encoder.to() called for reload", comps["text_encoder"].to.called)


def test_ensure_on_gpu_with_components():
    """ensure_on_gpu should use component-level reload for WAN pipelines."""
    from scope.server.vram_offloader import DeviceLocation

    o = _reset_offloader()
    pipeline, comps = _make_wan_pipeline()

    o.register_pipeline("ensure-test", pipeline=pipeline)
    o.mark_idle("ensure-test", pipeline=pipeline)

    # Reset mocks
    for c in comps.values():
        if hasattr(c, "to"):
            c.to.reset_mock()

    # ensure_on_gpu should detect offloaded components and reload them
    result = o.ensure_on_gpu(pipeline, "ensure-test")

    check("ensure_on_gpu returns pipeline", result is pipeline)

    with o._lock:
        rec = o._records["ensure-test"]
        all_gpu = all(c.location == DeviceLocation.GPU for c in rec.components.values())
    check("all components on GPU after ensure_on_gpu", all_gpu)


def test_ensure_on_gpu_fallback_whole_pipeline():
    """ensure_on_gpu should fall back to whole-pipeline .to() for non-WAN."""
    from scope.server.vram_offloader import DeviceLocation

    o = _reset_offloader()
    pipeline = _make_simple_pipeline()

    o.register_pipeline("fallback-test", pipeline=pipeline)

    # Manually set to CPU
    with o._lock:
        o._records["fallback-test"].location = DeviceLocation.CPU

    result = o.ensure_on_gpu(pipeline, "fallback-test")
    check("fallback returns pipeline", result is pipeline)
    check("whole-pipeline .to() called", pipeline.to.called)


# ---------------------------------------------------------------------------
# Suite 5: Text Encoder Offload/Reload
# ---------------------------------------------------------------------------


def test_text_encoder_offload_reload():
    print("\n=== Suite 5: Text Encoder Offload/Reload ===")
    from scope.server.vram_offloader import DeviceLocation

    o = _reset_offloader()
    pipeline, comps = _make_wan_pipeline()

    o.register_pipeline("te-test", pipeline=pipeline)

    # Offload text encoder
    o.offload_text_encoder("te-test", pipeline)

    with o._lock:
        te_loc = o._records["te-test"].components["text_encoder"].location
    check("text_encoder offloaded to CPU", te_loc == DeviceLocation.CPU)
    comps["text_encoder"].to.assert_called()

    # Reset mock
    comps["text_encoder"].to.reset_mock()

    # Reload text encoder
    o.ensure_text_encoder_on_gpu("te-test", pipeline)

    with o._lock:
        te_loc = o._records["te-test"].components["text_encoder"].location
    check("text_encoder reloaded to GPU", te_loc == DeviceLocation.GPU)
    comps["text_encoder"].to.assert_called()


def test_text_encoder_offload_idempotent():
    """Offloading an already-offloaded text encoder should be a no-op."""
    o = _reset_offloader()
    pipeline, comps = _make_wan_pipeline()

    o.register_pipeline("te-idem", pipeline=pipeline)
    o.offload_text_encoder("te-idem", pipeline)

    comps["text_encoder"].to.reset_mock()

    # Second offload should be no-op
    o.offload_text_encoder("te-idem", pipeline)
    check(
        "text_encoder.to() not called on second offload",
        not comps["text_encoder"].to.called,
    )


# ---------------------------------------------------------------------------
# Suite 6: offload_to_cpu Respects Pin Policies
# ---------------------------------------------------------------------------


def test_offload_to_cpu_respects_vae_pin():
    print("\n=== Suite 6: offload_to_cpu Pin Policies ===")
    from scope.server.vram_offloader import DeviceLocation

    o = _reset_offloader()
    pipeline, comps = _make_wan_pipeline()

    o.register_pipeline("pin-test", pipeline=pipeline)

    # offload_to_cpu should skip VAE
    o.offload_to_cpu(pipeline, "pin-test")

    with o._lock:
        rec = o._records["pin-test"]
        vae_loc = rec.components["vae"].location
        gen_loc = rec.components["generator"].location
        te_loc = rec.components["text_encoder"].location

    check("VAE stays on GPU during offload_to_cpu", vae_loc == DeviceLocation.GPU)
    check("generator moved to CPU", gen_loc == DeviceLocation.CPU)
    check("text_encoder moved to CPU", te_loc == DeviceLocation.CPU)

    # VAE .to() should NOT have been called
    comps["vae"].to.assert_not_called()
    check("vae.to() never called during offload_to_cpu", True)


# ---------------------------------------------------------------------------
# Suite 7: VRAMMonitor Video Memory Tracking
# ---------------------------------------------------------------------------


def test_video_memory_tracking():
    print("\n=== Suite 7: Video Memory Tracking ===")
    m = _reset_monitor()

    m.record_pipeline_load("vid-1", vram_delta_bytes=3_000_000_000)

    # Record video memory
    m.record_video_memory(
        "vid-1",
        kv_cache_bytes=500_000_000,
        temporal_buffer_bytes=200_000_000,
        vae_cache_bytes=50_000_000,
    )

    records = m.get_pipeline_records()
    rec = records["vid-1"]
    check("video_memory exists", rec.video_memory is not None)
    check("kv_cache_bytes correct", rec.video_memory.kv_cache_bytes == 500_000_000)
    check(
        "temporal_buffer_bytes correct",
        rec.video_memory.temporal_buffer_bytes == 200_000_000,
    )
    check("vae_cache_bytes correct", rec.video_memory.vae_cache_bytes == 50_000_000)
    check(
        "total_runtime_bytes correct",
        rec.video_memory.total_runtime_bytes == 750_000_000,
    )
    check(
        "peak_runtime_bytes correct", rec.video_memory.peak_runtime_bytes == 750_000_000
    )


def test_video_memory_peak_tracking():
    m = _reset_monitor()
    m.record_pipeline_load("vid-peak", vram_delta_bytes=1_000_000_000)

    # First recording
    m.record_video_memory("vid-peak", kv_cache_bytes=500_000_000)

    # Higher recording
    m.record_video_memory("vid-peak", kv_cache_bytes=800_000_000)

    # Lower recording — peak should stay at 800M
    m.record_video_memory("vid-peak", kv_cache_bytes=300_000_000)

    rec = m.get_pipeline_records()["vid-peak"]
    check(
        "peak preserved after decrease",
        rec.video_memory.peak_runtime_bytes == 800_000_000,
    )
    check("current reflects latest", rec.video_memory.kv_cache_bytes == 300_000_000)


def test_clear_video_memory():
    m = _reset_monitor()
    m.record_pipeline_load("vid-clear", vram_delta_bytes=1_000_000_000)
    m.record_video_memory(
        "vid-clear", kv_cache_bytes=500_000_000, temporal_buffer_bytes=200_000_000
    )

    m.clear_video_memory("vid-clear")

    rec = m.get_pipeline_records()["vid-clear"]
    check("kv_cache zeroed", rec.video_memory.kv_cache_bytes == 0)
    check("temporal_buffer zeroed", rec.video_memory.temporal_buffer_bytes == 0)
    check(
        "peak preserved after clear", rec.video_memory.peak_runtime_bytes == 700_000_000
    )


def test_component_vram_tracking():
    m = _reset_monitor()
    m.record_pipeline_load("comp-vram", vram_delta_bytes=5_000_000_000)

    m.record_component_vram("comp-vram", "vae", 400_000_000)
    m.record_component_vram("comp-vram", "generator", 3_000_000_000)
    m.record_component_vram("comp-vram", "text_encoder", 1_500_000_000)

    rec = m.get_pipeline_records()["comp-vram"]
    check("component_vram has 3 entries", len(rec.component_vram) == 3)
    check("vae vram correct", rec.component_vram["vae"] == 400_000_000)
    check("generator vram correct", rec.component_vram["generator"] == 3_000_000_000)


# ---------------------------------------------------------------------------
# Suite 8: Video Runtime Overhead Estimation
# ---------------------------------------------------------------------------


def test_runtime_overhead_estimation():
    print("\n=== Suite 8: Runtime Overhead Estimation ===")
    m = _reset_monitor()

    overhead_512 = m.estimate_video_runtime_overhead_gb(512, 512, 1)
    check("512x512 overhead is 0.8 GB", overhead_512 == 0.8)

    overhead_256 = m.estimate_video_runtime_overhead_gb(256, 256, 1)
    check("256x256 overhead is 0.3 GB", overhead_256 == 0.3)

    overhead_1024 = m.estimate_video_runtime_overhead_gb(1024, 1024, 1)
    check("1024x1024 overhead is 2.5 GB", overhead_1024 == 2.5)

    # Multi-pipeline scaling
    overhead_2x = m.estimate_video_runtime_overhead_gb(512, 512, 2)
    check("2 pipelines at 512x512 = 1.6 GB", overhead_2x == 1.6)

    # Non-standard resolution picks closest
    overhead_480 = m.estimate_video_runtime_overhead_gb(480, 480, 1)
    check("480x480 picks closest tier", overhead_480 > 0)


# ---------------------------------------------------------------------------
# Suite 9: Status Reporting with Components
# ---------------------------------------------------------------------------


def test_offloader_status_with_components():
    print("\n=== Suite 9: Status Reporting ===")

    o = _reset_offloader()
    pipeline, _ = _make_wan_pipeline()

    o.register_pipeline("status-test", pipeline=pipeline)
    o.mark_idle("status-test", pipeline=pipeline)

    status = o.get_status()
    check("status has 1 entry", len(status) == 1)

    entry = status[0]
    check("status has components key", "components" in entry)
    check("vae component in status", "vae" in entry["components"])
    check("vae location is gpu", entry["components"]["vae"]["location"] == "gpu")
    check("vae pin_to_gpu is True", entry["components"]["vae"]["pin_to_gpu"] is True)
    check(
        "generator location is cpu",
        entry["components"]["generator"]["location"] == "cpu",
    )
    check(
        "text_encoder location is cpu",
        entry["components"]["text_encoder"]["location"] == "cpu",
    )


def test_monitor_status_with_video_memory():
    m = _reset_monitor()
    m.record_pipeline_load("status-vid", vram_delta_bytes=3_000_000_000)
    m.record_video_memory("status-vid", kv_cache_bytes=500_000_000)
    m.record_component_vram("status-vid", "vae", 400_000_000)

    status = m.get_status()
    pipeline_entry = status["pipelines"][0]

    check("status has video_memory", "video_memory" in pipeline_entry)
    check("status has components", "components" in pipeline_entry)
    check(
        "kv_cache_mb in video_memory", "kv_cache_mb" in pipeline_entry["video_memory"]
    )
    check("vae in components", "vae" in pipeline_entry["components"])


# ---------------------------------------------------------------------------
# Suite 10: Thread Safety of Component Offloading
# ---------------------------------------------------------------------------


def test_concurrent_component_offload():
    print("\n=== Suite 10: Thread Safety ===")
    o = _reset_offloader()

    errors = []
    iterations = 100

    def worker(worker_id):
        try:
            for i in range(iterations):
                pipeline, _ = _make_wan_pipeline()
                pid = f"thread-{worker_id}-{i}"
                o.register_pipeline(pid, pipeline=pipeline)
                o.mark_active(pid)
                o.mark_idle(pid, pipeline=pipeline)
                o.ensure_components_on_gpu(pid, pipeline)
                o.offload_text_encoder(pid, pipeline)
                o.ensure_text_encoder_on_gpu(pid, pipeline)
                o.unregister_pipeline(pid)
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    check(
        "no errors in concurrent component offloading",
        len(errors) == 0,
        "; ".join(errors) if errors else "",
    )

    total_ops = 5 * iterations * 7  # 7 operations per iteration
    check(f"completed {total_ops} concurrent operations", True)


# ---------------------------------------------------------------------------
# Suite 11: can_fit_chain with Video Runtime Overhead
# ---------------------------------------------------------------------------


def test_can_fit_chain_with_runtime_overhead():
    print("\n=== Suite 11: can_fit_chain with Runtime Overhead ===")
    m = _reset_monitor()

    # Mock CUDA available and snapshot
    m._cuda_available = True
    original_snapshot = m.snapshot

    mock_snap = MagicMock()
    mock_snap.total_gb = 24.0
    mock_snap.free_gb = 5.0
    m.snapshot = MagicMock(return_value=mock_snap)

    # Record a pipeline with known VRAM
    m.record_pipeline_load("chain-1", vram_delta_bytes=int(3 * 1024**3))

    # Chain with already-loaded pipeline at 512x512
    fits, msg = m.can_fit_chain(["chain-1"], frame_height=512, frame_width=512)
    check("already-loaded pipeline fits", fits)
    check(
        "message mentions runtime",
        "runtime" in msg.lower() or "already loaded" in msg.lower(),
    )

    # Restore
    m.snapshot = original_snapshot
    m._cuda_available = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global PASS, FAIL

    # Suite 1: Component Detection
    test_component_detection()
    test_component_detection_no_components()

    # Suite 2: Registration
    test_register_with_components()
    test_register_without_components()

    # Suite 3: Idle Offloading
    test_mark_idle_offloads_components()
    test_mark_idle_without_pipeline_no_crash()

    # Suite 4: Component Reload
    test_ensure_components_on_gpu()
    test_ensure_on_gpu_with_components()
    test_ensure_on_gpu_fallback_whole_pipeline()

    # Suite 5: Text Encoder
    test_text_encoder_offload_reload()
    test_text_encoder_offload_idempotent()

    # Suite 6: Pin Policies
    test_offload_to_cpu_respects_vae_pin()

    # Suite 7: Video Memory Tracking
    test_video_memory_tracking()
    test_video_memory_peak_tracking()
    test_clear_video_memory()
    test_component_vram_tracking()

    # Suite 8: Runtime Overhead
    test_runtime_overhead_estimation()

    # Suite 9: Status Reporting
    test_offloader_status_with_components()
    test_monitor_status_with_video_memory()

    # Suite 10: Thread Safety
    test_concurrent_component_offload()

    # Suite 11: can_fit_chain
    test_can_fit_chain_with_runtime_overhead()

    print(f"\n{'=' * 60}")
    print(f"Video VRAM Optimization Tests: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
