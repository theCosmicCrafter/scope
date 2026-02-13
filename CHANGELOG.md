# Changelog

## [Unreleased] — feat/vram-management-system

### Added

#### VRAM Monitoring (`vram_monitor.py`)
- Singleton `VRAMMonitor` with thread-safe GPU memory snapshots via `torch.cuda.mem_get_info`
- `GPUSnapshot` dataclass: total, used, free, reserved, active bytes + utilization %
- `PipelineMemoryRecord` per-pipeline tracking with measured VRAM deltas at load/unload
- `estimate_chain_vram_gb()` for pre-flight budget estimation of pipeline chains
- `can_fit_chain()` returns `(fits, message)` accounting for already-loaded pipelines + safety margin
- `GET /api/v1/hardware/vram` endpoint returning full VRAM status + per-pipeline breakdown

#### Smart Model Offloading (`vram_offloader.py`)
- `VRAMOffloader` singleton managing pipeline device placement (GPU/CPU)
- LRU-based eviction of idle pipelines to CPU when VRAM headroom drops below threshold (default 1.5 GB)
- `ensure_on_gpu()` transparently reloads offloaded pipelines with OOM retry + eviction fallback
- `mark_active()` / `mark_idle()` protects streaming pipelines from eviction
- Offloader status included in `/api/v1/hardware/vram` response

#### Multi-Pipeline Memory Optimization
- Pre-flight VRAM budget check in `_setup_pipeline_chain_sync()` before chain creation
- `POST /api/v1/hardware/vram/check` endpoint for frontend pre-flight budget validation
- OOM recovery in `PipelineProcessor._is_recoverable()` — clears CUDA cache + GC, allows retry
- `get_loaded_pipelines()` public method on `PipelineManager` (replaces protected `_pipelines` access)

#### Frontend VRAM Dashboard (`VRAMDashboard.tsx`)
- Compact VRAM usage bar in status bar with color-coded utilization (green <60%, amber <80%, red >=80%)
- Real-time polling every 3s from `/api/v1/hardware/vram`
- Tooltip with detailed breakdown: total/used/free/reserved memory
- Per-pipeline VRAM usage with GPU/CPU placement indicators (green=active, amber=idle, gray=offloaded)
- Critical VRAM warning at 90%+ utilization
- Graceful fallback for no-GPU, loading, and error states

#### Video-Optimised Component-Level Offloading (`vram_offloader.py`)
- `ComponentRole` enum: VAE, GENERATOR, TEXT_ENCODER, OTHER — semantic classification for offload priority
- `VideoComponentPolicy` dataclass: per-component offload rules (pin_to_gpu, offload_when_idle, offload_between_uses, priority)
- `DEFAULT_VIDEO_COMPONENT_POLICIES` — WAN-architecture defaults:
  - **VAE** pinned to GPU (small, runs every frame, latency-critical)
  - **Text encoder** (T5) offloaded to CPU between prompt changes (~6 GB freed)
  - **Generator** (CausalWanModel/DiT) offloaded to CPU when pipeline goes idle
- `ComponentOffloadRecord` per-component device location tracking
- Auto-detection of WAN components via `ComponentsManager` introspection at registration
- `_offload_idle_components()` — proactive component offload on `mark_idle()`
- `offload_text_encoder()` / `ensure_text_encoder_on_gpu()` — targeted text encoder management
- `ensure_components_on_gpu()` — selective reload of offloaded components before streaming
- `offload_to_cpu()` now respects pin policies (VAE stays on GPU)
- `ensure_on_gpu()` uses component-level reload for WAN pipelines, whole-pipeline fallback for others
- Component-level status in offloader API response

#### Video Memory Tracking (`vram_monitor.py`)
- `VideoMemoryBreakdown` dataclass: KV cache, temporal buffer, VAE cache, peak runtime tracking
- `record_video_memory()` for runtime video memory tracking during streaming
- `clear_video_memory()` zeroes current allocations while preserving peak for diagnostics
- `record_component_vram()` for per-component VRAM attribution
- `estimate_video_runtime_overhead_gb()` — resolution-aware overhead estimation for KV caches and temporal buffers
- `can_fit_chain()` now includes video runtime overhead in pre-flight budget checks
- Component and video memory details exposed in API status response

#### Tests (`tests/test_vram_system.py`)
- 82-test stress suite across 6 suites
- 10-thread concurrent stress test (6,500 operations)
- Full pipeline lifecycle integration tests
- OOM retry limit verification
- Budget estimation edge cases

#### Tests (`tests/test_video_vram_optimization.py`)
- 76-test suite across 11 suites for video-optimised offloading
- Component detection from WAN-style pipelines
- VAE pin policy enforcement (never offloaded)
- Text encoder offload/reload round-trip
- Idle component offloading and active reload
- Video memory tracking (KV cache, temporal buffers, peak)
- Runtime overhead estimation per resolution tier
- 5-thread concurrent component offloading stress test (3,500 operations)
- Status reporting with component and video memory details

### Fixed
- **Missing `get_vram_offloader` import** in `pipeline_manager.py` — would cause `NameError` on every pipeline load, get, and unload
- **Missing `get_vram_monitor` import** in `frame_processor.py` — would cause `NameError` during chain budget check
- **`_evict_one_idle` not freeing GPU memory** — only updated the record to CPU without calling `.to(cpu)` on the pipeline, making OOM retry fail again
- **Infinite OOM retry loop** in `_is_recoverable` — now caps at 3 consecutive retries, resets on successful chunk
- **Over-eviction in `ensure_headroom`** — was passing total chain VRAM estimate instead of delta (new VRAM needed), causing unnecessary pipeline evictions
