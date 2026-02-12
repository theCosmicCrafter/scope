import { useCallback, useEffect, useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";

interface GPUInfo {
  total_gb: number;
  used_gb: number;
  free_gb: number;
  reserved_gb: number;
  utilization_pct: number;
}

interface PipelineVRAM {
  pipeline_id: string;
  measured_vram_mb: number;
  estimated_vram_gb: number | null;
  loaded_at: number;
}

interface OffloaderEntry {
  pipeline_id: string;
  location: "gpu" | "cpu";
  is_active: boolean;
  last_accessed: number;
  measured_vram_mb: number;
}

interface VRAMStatus {
  cuda_available: boolean;
  gpu: GPUInfo;
  pipelines: PipelineVRAM[];
  total_pipeline_vram_gb: number;
  offloader: OffloaderEntry[];
  timestamp: number;
}

const POLL_INTERVAL_MS = 3000;

function getUtilizationColor(pct: number): string {
  if (pct < 60) return "bg-emerald-500";
  if (pct < 80) return "bg-amber-500";
  return "bg-red-500";
}

function getUtilizationTextColor(pct: number): string {
  if (pct < 60) return "text-emerald-400";
  if (pct < 80) return "text-amber-400";
  return "text-red-400";
}

export function VRAMDashboard() {
  const [status, setStatus] = useState<VRAMStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/hardware/vram");
      if (!res.ok) {
        setError(`HTTP ${res.status}`);
        return;
      }
      const data: VRAMStatus = await res.json();
      setStatus(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Fetch failed");
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  if (error) {
    return (
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <span className="font-medium">VRAM:</span>
        <span className="text-red-400">err</span>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <span className="font-medium">VRAM:</span>
        <span>...</span>
      </div>
    );
  }

  if (!status.cuda_available) {
    return (
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <span className="font-medium">VRAM:</span>
        <span>CPU only</span>
      </div>
    );
  }

  const { gpu, pipelines, offloader } = status;
  const pct = gpu.utilization_pct;
  const barColor = getUtilizationColor(pct);
  const textColor = getUtilizationTextColor(pct);

  const cpuPipelines = offloader.filter(e => e.location === "cpu");

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-2 cursor-default select-none">
            {/* Compact bar */}
            <div className="flex items-center gap-1.5 text-xs">
              <span className="font-medium text-muted-foreground">VRAM</span>
              <div className="relative w-16 h-2 rounded-full bg-secondary overflow-hidden">
                <div
                  className={`absolute inset-y-0 left-0 rounded-full transition-all duration-500 ${barColor}`}
                  style={{ width: `${Math.min(100, pct)}%` }}
                />
              </div>
              <span className={`font-mono tabular-nums ${textColor}`}>
                {gpu.used_gb.toFixed(1)}/{gpu.total_gb.toFixed(0)}G
              </span>
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs p-3">
          <div className="space-y-2 text-xs">
            {/* Header */}
            <div className="font-semibold text-sm">
              GPU Memory — {pct.toFixed(1)}% used
            </div>

            {/* Memory breakdown */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
              <span className="text-muted-foreground">Total</span>
              <span className="font-mono text-right">
                {gpu.total_gb.toFixed(2)} GB
              </span>
              <span className="text-muted-foreground">Used</span>
              <span className="font-mono text-right">
                {gpu.used_gb.toFixed(2)} GB
              </span>
              <span className="text-muted-foreground">Free</span>
              <span className="font-mono text-right">
                {gpu.free_gb.toFixed(2)} GB
              </span>
              {gpu.reserved_gb > 0.01 && (
                <>
                  <span className="text-muted-foreground">Reserved</span>
                  <span className="font-mono text-right">
                    {gpu.reserved_gb.toFixed(2)} GB
                  </span>
                </>
              )}
            </div>

            {/* Pipeline breakdown */}
            {pipelines.length > 0 && (
              <div className="border-t border-border pt-1.5">
                <div className="font-semibold mb-1">Loaded Pipelines</div>
                {pipelines.map(p => {
                  const offEntry = offloader.find(
                    o => o.pipeline_id === p.pipeline_id
                  );
                  const loc = offEntry?.location ?? "gpu";
                  const active = offEntry?.is_active ?? false;
                  return (
                    <div
                      key={p.pipeline_id}
                      className="flex items-center justify-between gap-2"
                    >
                      <span className="truncate max-w-[140px]">
                        {p.pipeline_id}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono">
                          {p.measured_vram_mb > 0
                            ? `${(p.measured_vram_mb / 1024).toFixed(1)}G`
                            : "—"}
                        </span>
                        <span
                          className={`inline-block w-1.5 h-1.5 rounded-full ${
                            loc === "gpu"
                              ? active
                                ? "bg-emerald-400"
                                : "bg-amber-400"
                              : "bg-zinc-500"
                          }`}
                          title={
                            loc === "gpu"
                              ? active
                                ? "Active on GPU"
                                : "Idle on GPU"
                              : "Offloaded to CPU"
                          }
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Offload summary */}
            {cpuPipelines.length > 0 && (
              <div className="text-muted-foreground border-t border-border pt-1">
                {cpuPipelines.length} pipeline(s) offloaded to CPU
              </div>
            )}

            {/* Warning */}
            {pct >= 90 && (
              <div className="text-red-400 font-medium border-t border-border pt-1">
                ⚠ VRAM critically low — performance may degrade
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
