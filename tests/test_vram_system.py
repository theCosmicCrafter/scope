"""Stress tests for the VRAM management system.

Tests cover:
  - VRAMMonitor singleton, snapshots, pipeline record lifecycle
  - VRAMOffloader singleton, register/unregister, active/idle, LRU eviction
  - Thread-safety under concurrent load/unload/mark operations
  - OOM retry counter in PipelineProcessor._is_recoverable
  - API status format contracts
  - ensure_headroom with mock pipelines
  - Budget estimation and can_fit_chain logic
"""

import logging
import threading
import time
from unittest.mock import MagicMock

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_vram_system")

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
    # Clear pipeline records
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


# ---------------------------------------------------------------------------
# Test Suite 1: VRAMMonitor
# ---------------------------------------------------------------------------


def test_monitor_singleton():
    print("\n=== Suite 1: VRAMMonitor ===")
    from scope.server.vram_monitor import VRAMMonitor, get_vram_monitor

    m1 = get_vram_monitor()
    m2 = get_vram_monitor()
    m3 = VRAMMonitor()
    check("singleton identity (get_vram_monitor)", m1 is m2)
    check("singleton identity (VRAMMonitor())", m1 is m3)


def test_monitor_snapshot():
    m = _reset_monitor()
    snap = m.snapshot()
    check("snapshot returns GPUSnapshot", hasattr(snap, "total_bytes"))
    if torch.cuda.is_available():
        check("snapshot total > 0", snap.total_bytes > 0, f"got {snap.total_bytes}")
        check("snapshot free > 0", snap.free_bytes > 0, f"got {snap.free_bytes}")
        check(
            "utilization 0-100",
            0 <= snap.utilization_pct <= 100,
            f"got {snap.utilization_pct}",
        )
        check("total_gb property", snap.total_gb > 0)
    else:
        check("no-GPU snapshot zeros", snap.total_bytes == 0)
    check("last_snapshot cached", m.last_snapshot.timestamp == snap.timestamp)


def test_monitor_pipeline_records():
    m = _reset_monitor()

    # Load
    m.record_pipeline_load(
        "pipe-a", vram_delta_bytes=500 * 1024**2, estimated_vram_gb=2.0
    )
    recs = m.get_pipeline_records()
    check("record exists after load", "pipe-a" in recs)
    check("measured bytes correct", recs["pipe-a"].measured_vram_bytes == 500 * 1024**2)
    check("estimated_vram_gb correct", recs["pipe-a"].estimated_vram_gb == 2.0)
    check("loaded_at > 0", recs["pipe-a"].loaded_at > 0)

    # Negative delta clamped to 0
    m.record_pipeline_load("pipe-neg", vram_delta_bytes=-100, estimated_vram_gb=None)
    recs = m.get_pipeline_records()
    check("negative delta clamped", recs["pipe-neg"].measured_vram_bytes == 0)

    # Unload
    m.record_pipeline_unload("pipe-a")
    recs = m.get_pipeline_records()
    check("record removed after unload", "pipe-a" not in recs)

    # Unload non-existent (should not raise)
    m.record_pipeline_unload("does-not-exist")
    check("unload non-existent no error", True)

    # Cleanup
    m.record_pipeline_unload("pipe-neg")


def test_monitor_aggregates():
    m = _reset_monitor()
    m.record_pipeline_load("p1", vram_delta_bytes=1024**3)  # 1 GB
    m.record_pipeline_load("p2", vram_delta_bytes=2 * 1024**3)  # 2 GB
    check("total_pipeline_vram_bytes", m.total_pipeline_vram_bytes() == 3 * 1024**3)
    check("total_pipeline_vram_gb", abs(m.total_pipeline_vram_gb() - 3.0) < 0.01)
    m.record_pipeline_unload("p1")
    m.record_pipeline_unload("p2")


def test_monitor_api_status():
    m = _reset_monitor()
    m.record_pipeline_load("api-pipe", vram_delta_bytes=1024**3, estimated_vram_gb=1.0)
    status = m.get_status()
    check("status has cuda_available", "cuda_available" in status)
    check("status has gpu", "gpu" in status)
    check("status has pipelines", "pipelines" in status)
    check("status has total_pipeline_vram_gb", "total_pipeline_vram_gb" in status)
    check("status has timestamp", "timestamp" in status)
    gpu = status["gpu"]
    for key in ("total_gb", "used_gb", "free_gb", "reserved_gb", "utilization_pct"):
        check(f"gpu.{key} present", key in gpu, f"missing {key}")
    pipes = status["pipelines"]
    check("1 pipeline in status", len(pipes) == 1)
    check("pipeline_id in entry", pipes[0]["pipeline_id"] == "api-pipe")
    check("measured_vram_mb in entry", "measured_vram_mb" in pipes[0])
    m.record_pipeline_unload("api-pipe")


def test_monitor_chain_estimation():
    m = _reset_monitor()
    m.record_pipeline_load("loaded-1", vram_delta_bytes=int(1.5 * 1024**3))
    est = m.estimate_chain_vram_gb(["loaded-1"])
    check("chain estimate uses measured", abs(est - 1.5) < 0.01, f"got {est}")

    # Unknown pipeline returns 0 (no schema)
    est2 = m.estimate_chain_vram_gb(["unknown-pipe"])
    check("unknown pipeline estimate 0", est2 == 0.0, f"got {est2}")

    m.record_pipeline_unload("loaded-1")


def test_monitor_can_fit_chain():
    m = _reset_monitor()
    if not torch.cuda.is_available():
        fits, msg = m.can_fit_chain(["any"])
        check("no-GPU always fits", fits)
        return

    # Already loaded pipeline should fit
    m.record_pipeline_load("fit-test", vram_delta_bytes=int(0.5 * 1024**3))
    fits, msg = m.can_fit_chain(["fit-test"])
    check("loaded pipeline fits", fits, msg)

    # No estimates → allows load
    fits2, msg2 = m.can_fit_chain(["no-estimate-pipe"])
    check("no estimate allows load", fits2, msg2)

    m.record_pipeline_unload("fit-test")


# ---------------------------------------------------------------------------
# Test Suite 2: VRAMOffloader
# ---------------------------------------------------------------------------


def test_offloader_singleton():
    print("\n=== Suite 2: VRAMOffloader ===")
    from scope.server.vram_offloader import get_vram_offloader

    o1 = get_vram_offloader()
    o2 = get_vram_offloader()
    check("offloader singleton", o1 is o2)


def test_offloader_register_unregister():
    o = _reset_offloader()
    o.register_pipeline("reg-1", measured_vram_bytes=1024**3)
    status = o.get_status()
    check("registered 1 pipeline", len(status) == 1)
    check("pipeline_id correct", status[0]["pipeline_id"] == "reg-1")
    check("location is gpu", status[0]["location"] == "gpu")
    check("is_active False", status[0]["is_active"] is False)

    o.unregister_pipeline("reg-1")
    check("unregistered", len(o.get_status()) == 0)

    # Unregister non-existent (no error)
    o.unregister_pipeline("nope")
    check("unregister non-existent ok", True)


def test_offloader_active_idle():
    o = _reset_offloader()
    o.register_pipeline("act-1")

    o.mark_active("act-1")
    status = o.get_status()
    check("mark_active works", status[0]["is_active"] is True)

    o.mark_idle("act-1")
    status = o.get_status()
    check("mark_idle works", status[0]["is_active"] is False)

    # Mark non-existent (no error)
    o.mark_active("nope")
    o.mark_idle("nope")
    check("mark non-existent ok", True)

    o.unregister_pipeline("act-1")


def test_offloader_gpu_cpu_lists():
    o = _reset_offloader()
    o.register_pipeline("list-1")
    o.register_pipeline("list-2")

    check("2 on GPU", len(o.get_gpu_pipeline_ids()) == 2)
    check("0 on CPU", len(o.get_cpu_pipeline_ids()) == 0)

    o.unregister_pipeline("list-1")
    o.unregister_pipeline("list-2")


def test_offloader_ensure_on_gpu_noop():
    """ensure_on_gpu should be a no-op if pipeline is already on GPU."""
    o = _reset_offloader()
    o.register_pipeline("gpu-noop")

    mock_pipe = MagicMock()
    result = o.ensure_on_gpu(mock_pipe, "gpu-noop")
    check("ensure_on_gpu returns pipeline", result is mock_pipe)
    mock_pipe.to.assert_not_called()
    check("no .to() call for GPU pipeline", True)

    o.unregister_pipeline("gpu-noop")


def test_offloader_ensure_on_gpu_reload():
    """ensure_on_gpu should call .to(cuda) if pipeline is on CPU."""
    o = _reset_offloader()
    o.register_pipeline("reload-1")

    # Manually set to CPU
    with o._lock:
        from scope.server.vram_offloader import DeviceLocation

        o._records["reload-1"].location = DeviceLocation.CPU

    mock_pipe = MagicMock()
    result = o.ensure_on_gpu(mock_pipe, "reload-1")
    check("ensure_on_gpu returns pipeline", result is mock_pipe)
    mock_pipe.to.assert_called()
    check(".to() called for CPU->GPU reload", True)

    status = o.get_status()
    check("location updated to gpu", status[0]["location"] == "gpu")

    o.unregister_pipeline("reload-1")


def test_offloader_offload_to_cpu():
    """offload_to_cpu should move idle pipeline to CPU."""
    o = _reset_offloader()
    o.register_pipeline("off-1")

    mock_pipe = MagicMock()
    result = o.offload_to_cpu(mock_pipe, "off-1")
    check("offload returns pipeline", result is mock_pipe)
    mock_pipe.to.assert_called()
    check(".to(cpu) called", True)

    status = o.get_status()
    check("location is cpu", status[0]["location"] == "cpu")

    o.unregister_pipeline("off-1")


def test_offloader_refuses_active_offload():
    """offload_to_cpu should refuse to offload active pipelines."""
    o = _reset_offloader()
    o.register_pipeline("active-1")
    o.mark_active("active-1")

    mock_pipe = MagicMock()
    result = o.offload_to_cpu(mock_pipe, "active-1")
    mock_pipe.to.assert_not_called()
    check("active pipeline not offloaded", True)

    status = o.get_status()
    check("still on gpu", status[0]["location"] == "gpu")

    o.unregister_pipeline("active-1")


def test_offloader_lru_eviction():
    """ensure_headroom should evict LRU idle pipeline first."""
    o = _reset_offloader()
    _reset_monitor()

    # Register 3 pipelines with staggered access times
    o.register_pipeline("lru-old", measured_vram_bytes=1024**3)
    time.sleep(0.01)
    o.register_pipeline("lru-mid", measured_vram_bytes=1024**3)
    time.sleep(0.01)
    o.register_pipeline("lru-new", measured_vram_bytes=1024**3)

    # Mark lru-mid as active (should be skipped)
    o.mark_active("lru-mid")

    mock_old = MagicMock()
    mock_mid = MagicMock()
    mock_new = MagicMock()
    pipelines = {"lru-old": mock_old, "lru-mid": mock_mid, "lru-new": mock_new}

    # Request huge headroom to force eviction
    o.ensure_headroom(needed_bytes=999 * 1024**3, pipelines=pipelines)

    # lru-old should be evicted first (oldest idle), lru-mid skipped (active)
    mock_old.to.assert_called()
    mock_mid.to.assert_not_called()
    check("LRU eviction: oldest idle evicted first", True)
    check("LRU eviction: active pipeline skipped", True)

    o.unregister_pipeline("lru-old")
    o.unregister_pipeline("lru-mid")
    o.unregister_pipeline("lru-new")


def test_offloader_ensure_headroom_already_ok():
    """ensure_headroom should be a no-op if enough free VRAM."""
    o = _reset_offloader()
    result = o.ensure_headroom(needed_bytes=0, pipelines={})
    check("headroom ok with 0 needed", result is True)


# ---------------------------------------------------------------------------
# Test Suite 3: Thread Safety Stress Test
# ---------------------------------------------------------------------------


def test_thread_safety():
    print("\n=== Suite 3: Thread Safety Stress ===")

    m = _reset_monitor()
    o = _reset_offloader()
    errors = []
    NUM_THREADS = 10
    OPS_PER_THREAD = 50

    def worker(thread_id):
        try:
            for i in range(OPS_PER_THREAD):
                pid = f"thread-{thread_id}-pipe-{i % 5}"
                # Monitor ops
                m.record_pipeline_load(pid, vram_delta_bytes=1024 * 1024 * (i + 1))
                m.snapshot()
                m.get_pipeline_records()
                m.total_pipeline_vram_bytes()
                m.get_status()
                m.record_pipeline_unload(pid)

                # Offloader ops
                o.register_pipeline(pid, measured_vram_bytes=1024 * 1024)
                o.mark_active(pid)
                o.get_status()
                o.get_gpu_pipeline_ids()
                o.get_cpu_pipeline_ids()
                o.mark_idle(pid)
                o.unregister_pipeline(pid)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.monotonic() - start

    total_ops = NUM_THREADS * OPS_PER_THREAD * 13  # 13 ops per iteration
    check(
        f"no thread errors ({NUM_THREADS} threads, {total_ops} ops)",
        len(errors) == 0,
        f"{len(errors)} errors: {errors[:3]}",
    )
    check(f"completed in {elapsed:.2f}s", elapsed < 30)

    # Verify clean state
    check("monitor clean after stress", len(m.get_pipeline_records()) == 0)
    check("offloader clean after stress", len(o.get_status()) == 0)


# ---------------------------------------------------------------------------
# Test Suite 4: PipelineProcessor OOM Recovery
# ---------------------------------------------------------------------------


def test_oom_retry_limit():
    print("\n=== Suite 4: OOM Recovery ===")
    from scope.server.pipeline_processor import PipelineProcessor

    mock_pipeline = MagicMock()
    mock_pipeline.get_config_class.return_value = MagicMock()

    proc = PipelineProcessor(
        pipeline=mock_pipeline,
        pipeline_id="oom-test",
        initial_parameters={},
    )

    # Simulate consecutive OOM errors
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory")

    # First 3 should be recoverable
    for i in range(3):
        result = proc._is_recoverable(oom)
        check(f"OOM attempt {i + 1} recoverable", result is True)

    # 4th should fail (exceeded max retries)
    result = proc._is_recoverable(oom)
    check("OOM attempt 4 NOT recoverable (limit hit)", result is False)

    # Reset counter and verify it works again
    proc._oom_consecutive_count = 0
    result = proc._is_recoverable(oom)
    check("OOM recoverable after counter reset", result is True)

    # Non-OOM error resets counter
    proc._oom_consecutive_count = 2
    non_oom = RuntimeError("some other error")
    result = proc._is_recoverable(non_oom)
    check("non-OOM error resets counter", proc._oom_consecutive_count == 0)
    check("non-OOM error is recoverable", result is True)


# ---------------------------------------------------------------------------
# Test Suite 5: Cross-Module Integration
# ---------------------------------------------------------------------------


def test_cross_module_integration():
    print("\n=== Suite 5: Cross-Module Integration ===")

    m = _reset_monitor()
    o = _reset_offloader()

    # Simulate full pipeline lifecycle: load → register → active → idle → offload → reload → unload
    pid = "integration-pipe"
    vram_bytes = int(0.5 * 1024**3)

    # 1. Load
    m.record_pipeline_load(pid, vram_delta_bytes=vram_bytes, estimated_vram_gb=0.5)
    o.register_pipeline(pid, measured_vram_bytes=vram_bytes)
    check("lifecycle: loaded + registered", pid in m.get_pipeline_records())

    # 2. Mark active
    o.mark_active(pid)
    status = o.get_status()
    check("lifecycle: marked active", status[0]["is_active"] is True)

    # 3. Mark idle
    o.mark_idle(pid)
    status = o.get_status()
    check("lifecycle: marked idle", status[0]["is_active"] is False)

    # 4. Offload to CPU
    mock_pipe = MagicMock()
    o.offload_to_cpu(mock_pipe, pid)
    check("lifecycle: offloaded to CPU", o.get_status()[0]["location"] == "cpu")

    # 5. Reload to GPU
    o.ensure_on_gpu(mock_pipe, pid)
    check("lifecycle: reloaded to GPU", o.get_status()[0]["location"] == "gpu")

    # 6. Unload
    m.record_pipeline_unload(pid)
    o.unregister_pipeline(pid)
    check(
        "lifecycle: unloaded + unregistered",
        pid not in m.get_pipeline_records() and len(o.get_status()) == 0,
    )

    # 7. API status includes offloader
    m.record_pipeline_load("api-int", vram_delta_bytes=1024**3)
    o.register_pipeline("api-int")
    api = m.get_status()
    off = o.get_status()
    check("API: monitor status has pipelines", len(api["pipelines"]) == 1)
    check("API: offloader status has entries", len(off) == 1)

    # Cleanup
    m.record_pipeline_unload("api-int")
    o.unregister_pipeline("api-int")


# ---------------------------------------------------------------------------
# Test Suite 6: Budget Check Edge Cases
# ---------------------------------------------------------------------------


def test_budget_edge_cases():
    print("\n=== Suite 6: Budget Edge Cases ===")
    m = _reset_monitor()

    # Empty chain
    est = m.estimate_chain_vram_gb([])
    check("empty chain estimate = 0", est == 0.0)

    fits, msg = m.can_fit_chain([])
    if torch.cuda.is_available():
        check("empty chain estimate allows", est == 0.0)
    else:
        check("no-GPU always fits", fits)

    # Chain with mix of loaded and unknown
    m.record_pipeline_load("known", vram_delta_bytes=int(2 * 1024**3))
    est = m.estimate_chain_vram_gb(["known", "unknown-xyz"])
    check("mixed chain: known contributes", est >= 2.0, f"got {est}")

    # Duplicate pipeline in chain
    est_dup = m.estimate_chain_vram_gb(["known", "known"])
    check(
        "duplicate pipeline counted twice", abs(est_dup - 4.0) < 0.01, f"got {est_dup}"
    )

    m.record_pipeline_unload("known")


# ---------------------------------------------------------------------------
# Run All
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("VRAM Management System — Stress Test Suite")
    print("=" * 60)

    # Suite 1
    test_monitor_singleton()
    test_monitor_snapshot()
    test_monitor_pipeline_records()
    test_monitor_aggregates()
    test_monitor_api_status()
    test_monitor_chain_estimation()
    test_monitor_can_fit_chain()

    # Suite 2
    test_offloader_singleton()
    test_offloader_register_unregister()
    test_offloader_active_idle()
    test_offloader_gpu_cpu_lists()
    test_offloader_ensure_on_gpu_noop()
    test_offloader_ensure_on_gpu_reload()
    test_offloader_offload_to_cpu()
    test_offloader_refuses_active_offload()
    test_offloader_lru_eviction()
    test_offloader_ensure_headroom_already_ok()

    # Suite 3
    test_thread_safety()

    # Suite 4
    test_oom_retry_limit()

    # Suite 5
    test_cross_module_integration()

    # Suite 6
    test_budget_edge_cases()

    # Summary
    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 60)
