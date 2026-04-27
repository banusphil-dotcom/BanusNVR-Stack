import { useState, useCallback, useRef, useEffect } from "react";
import { api } from "../api";
import { X, ImageUp, CheckCircle2, AlertTriangle, Loader2 } from "lucide-react";

interface ScanInfo {
  total: number;
  low_res: number;
}

interface Progress {
  current: number;
  total: number;
  updated: number;
  faces_found: number;
  failed: number;
}

interface Result {
  processed: number;
  faces_found: number;
  failed: number;
  total_events: number;
  low_res_found: number;
}

type Phase = "idle" | "scanning" | "extracting" | "done" | "error";

export function useReExtractModal() {
  const [open, setOpen] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [scan, setScan] = useState<ScanInfo | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Clean up on unmount
  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback(async () => {
    try {
      const s = await api.get<Record<string, unknown>>("/api/training/re-extract-thumbnails/status");

      const p = s.phase as string;
      setPhase(p as Phase);

      if (p === "scanning") {
        setScan({ total: s.total as number, low_res: s.low_res as number });
      } else if (p === "extracting" || p === "done") {
        setScan({ total: s.total as number, low_res: s.low_res as number });
        setProgress({
          current: s.current as number,
          total: s.low_res as number || s.total as number,
          updated: s.updated as number,
          faces_found: s.faces_found as number,
          failed: s.failed as number,
        });
      }

      if (p === "done") {
        setResult({
          processed: s.updated as number,
          faces_found: s.faces_found as number,
          failed: s.failed as number,
          total_events: s.total as number,
          low_res_found: s.low_res as number,
        });
        stopPolling();
      } else if (p === "error") {
        setError((s.error as string) || "Extraction failed");
        stopPolling();
      } else if (p === "idle") {
        // Task finished before we started polling, or was never started
        stopPolling();
      }
    } catch {
      // Network blip — keep polling, backend task continues
    }
  }, [stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollRef.current = setInterval(poll, 2000);
  }, [poll, stopPolling]);

  const start = useCallback(async (hours = 24, objectType = "person") => {
    setOpen(true);
    setPhase("scanning");
    setScan(null);
    setProgress(null);
    setResult(null);
    setError(null);

    try {
      await api.post(
        `/api/training/re-extract-thumbnails?hours=${hours}&object_type=${encodeURIComponent(objectType)}`,
      );

      startPolling();
    } catch (err: unknown) {
      // "already in progress" means a task is running — just attach to it
      if (err instanceof Error && err.message.toLowerCase().includes("already in progress")) {
        startPolling();
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to start extraction");
      setPhase("error");
    }
  }, [startPolling]);

  const close = useCallback(() => {
    if (phase === "done" || phase === "error" || phase === "idle") {
      stopPolling();
      setOpen(false);
      setPhase("idle");
    }
  }, [phase, stopPolling]);

  return { open, phase, scan, progress, result, error, start, close };
}

interface ModalProps {
  open: boolean;
  phase: Phase;
  scan: ScanInfo | null;
  progress: Progress | null;
  result: Result | null;
  error: string | null;
  onClose: () => void;
}

export function ReExtractModal({ open, phase, scan, progress, result, error, onClose }: ModalProps) {
  if (!open) return null;

  const pct = progress && progress.total > 0
    ? Math.round((progress.current / progress.total) * 100)
    : 0;

  const canClose = phase === "done" || phase === "error";

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-xl p-5 max-w-md w-full space-y-4 border border-slate-700 relative">
        {/* Close button */}
        {canClose && (
          <button
            onClick={onClose}
            className="absolute top-3 right-3 text-slate-400 hover:text-white"
          >
            <X size={18} />
          </button>
        )}

        {/* Title */}
        <div className="flex items-center gap-2.5">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
            phase === "done" ? "bg-green-600/20" :
            phase === "error" ? "bg-red-600/20" :
            "bg-blue-600/20"
          }`}>
            {phase === "done" ? <CheckCircle2 size={22} className="text-green-400" /> :
             phase === "error" ? <AlertTriangle size={22} className="text-red-400" /> :
             <ImageUp size={22} className="text-blue-400" />}
          </div>
          <div>
            <h3 className="font-semibold text-sm">
              {phase === "scanning" ? "Scanning Thumbnails…" :
               phase === "extracting" ? "Re-extracting HD Thumbnails" :
               phase === "done" ? "Extraction Complete" :
               phase === "error" ? "Extraction Failed" :
               "HD Thumbnail Extraction"}
            </h3>
            <p className="text-xs text-slate-400">
              {phase === "scanning" ? "Checking for low resolution thumbnails…" :
               phase === "extracting" ? "Pulling full-res frames from recordings" :
               phase === "done" ? "Thumbnails have been upgraded" :
               phase === "error" ? error :
               ""}
            </p>
          </div>
        </div>

        {/* Scan result */}
        {scan && phase !== "scanning" && (
          <div className="bg-slate-800/60 rounded-lg p-3 space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Events found</span>
              <span className="text-white font-medium">{scan.total}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Low resolution thumbnails</span>
              <span className="text-yellow-400 font-medium">{scan.low_res}</span>
            </div>
          </div>
        )}

        {/* Progress bar */}
        {(phase === "extracting" || phase === "done") && progress && (
          <div className="space-y-2">
            <div className="w-full h-2.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  phase === "done" ? "bg-green-500" : "bg-blue-500"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-slate-400">
              <span>{progress.current} / {progress.total}</span>
              <span>{pct}%</span>
            </div>
          </div>
        )}

        {/* Live stats */}
        {progress && (
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-slate-800/60 rounded-lg p-2.5 text-center">
              <p className="text-base font-bold text-green-400">{progress.updated}</p>
              <p className="text-[10px] text-slate-400">Extracted</p>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-2.5 text-center">
              <p className="text-base font-bold text-blue-400">{progress.faces_found}</p>
              <p className="text-[10px] text-slate-400">Faces Found</p>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-2.5 text-center">
              <p className="text-base font-bold text-slate-400">{progress.failed}</p>
              <p className="text-[10px] text-slate-400">No Recording</p>
            </div>
          </div>
        )}

        {/* Spinner for scanning phase */}
        {phase === "scanning" && (
          <div className="flex justify-center py-4">
            <Loader2 size={28} className="animate-spin text-blue-400" />
          </div>
        )}

        {/* Done actions */}
        {canClose && (
          <div className="flex justify-end pt-1">
            <button
              onClick={onClose}
              className="btn-primary text-sm py-2 px-5"
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
