import { useState, useCallback, useRef, useEffect } from "react";
import { api } from "../api";
import { X, RotateCcw, CheckCircle2, AlertTriangle, Loader2 } from "lucide-react";

interface Progress {
  current: number;
  total: number;
  face_found: number;
  recognition_changed: number;
  attribute_vetoed: number;
  agent_rejected: number;
  attributes_learned: number;
}

type Phase = "idle" | "running" | "done" | "error";

export function useReclassifyModal() {
  const [open, setOpen] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback(async () => {
    try {
      const s = await api.get<Record<string, unknown>>("/api/events/reclassify/status");
      const p = s.phase as string;
      setPhase(p as Phase);

      if (p === "running" || p === "done") {
        setProgress({
          current: s.current as number,
          total: s.total as number,
          face_found: s.face_found as number,
          recognition_changed: s.recognition_changed as number,
          attribute_vetoed: s.attribute_vetoed as number,
          agent_rejected: s.agent_rejected as number,
          attributes_learned: s.attributes_learned as number,
        });
      }

      if (p === "done" || p === "idle") {
        stopPolling();
      } else if (p === "error") {
        setError((s.error as string) || "Reclassification failed");
        stopPolling();
      }
    } catch {
      // Network blip — keep polling
    }
  }, [stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollRef.current = setInterval(poll, 1500);
  }, [poll, stopPolling]);

  const start = useCallback(async (hours = 24) => {
    setOpen(true);
    setPhase("running");
    setProgress(null);
    setError(null);

    try {
      const resp = await api.post<{ status: string; total: number }>(
        `/api/events/reclassify?hours=${hours}`
      );
      if (resp.status === "done" && resp.total === 0) {
        setPhase("done");
        setProgress({ current: 0, total: 0, face_found: 0, recognition_changed: 0, attribute_vetoed: 0, agent_rejected: 0, attributes_learned: 0 });
        return;
      }
      startPolling();
    } catch (err: unknown) {
      if (err instanceof Error && err.message.toLowerCase().includes("already in progress")) {
        startPolling();
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to start reclassification");
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

  return { open, phase, progress, error, start, close };
}

interface ModalProps {
  open: boolean;
  phase: Phase;
  progress: Progress | null;
  error: string | null;
  onClose: () => void;
}

export function ReclassifyModal({ open, phase, progress, error, onClose }: ModalProps) {
  if (!open) return null;

  const pct = progress && progress.total > 0
    ? Math.round((progress.current / progress.total) * 100)
    : 0;

  const canClose = phase === "done" || phase === "error";

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-xl p-5 max-w-md w-full space-y-4 border border-slate-700 relative">
        {canClose && (
          <button onClick={onClose} className="absolute top-3 right-3 text-slate-400 hover:text-white">
            <X size={18} />
          </button>
        )}

        {/* Title */}
        <div className="flex items-center gap-2.5">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
            phase === "done" ? "bg-green-600/20" :
            phase === "error" ? "bg-red-600/20" :
            "bg-indigo-600/20"
          }`}>
            {phase === "done" ? <CheckCircle2 size={22} className="text-green-400" /> :
             phase === "error" ? <AlertTriangle size={22} className="text-red-400" /> :
             <RotateCcw size={22} className="text-indigo-400 animate-spin" />}
          </div>
          <div>
            <h3 className="font-semibold text-sm">
              {phase === "running" ? "Reclassifying Events…" :
               phase === "done" ? "Reclassification Complete" :
               phase === "error" ? "Reclassification Failed" :
               "Reclassify Events"}
            </h3>
            <p className="text-xs text-slate-400">
              {phase === "running" ? "Re-running face detection & recognition" :
               phase === "done" ? "All person events have been re-evaluated" :
               phase === "error" ? error :
               ""}
            </p>
          </div>
        </div>

        {/* Progress bar */}
        {(phase === "running" || phase === "done") && progress && progress.total > 0 && (
          <div className="space-y-2">
            <div className="w-full h-2.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  phase === "done" ? "bg-green-500" : "bg-indigo-500"
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
              <p className="text-base font-bold text-blue-400">{progress.face_found}</p>
              <p className="text-[10px] text-slate-400">Faces Found</p>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-2.5 text-center">
              <p className="text-base font-bold text-green-400">{progress.recognition_changed}</p>
              <p className="text-[10px] text-slate-400">Updated</p>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-2.5 text-center">
              <p className="text-base font-bold text-purple-400">{progress.attributes_learned}</p>
              <p className="text-[10px] text-slate-400">Profiles Enriched</p>
            </div>
            {(progress.attribute_vetoed > 0 || progress.agent_rejected > 0) && (
              <>
                <div className="bg-slate-800/60 rounded-lg p-2.5 text-center">
                  <p className="text-base font-bold text-amber-400">{progress.attribute_vetoed}</p>
                  <p className="text-[10px] text-slate-400">Attr Vetoed</p>
                </div>
                <div className="bg-slate-800/60 rounded-lg p-2.5 text-center">
                  <p className="text-base font-bold text-red-400">{progress.agent_rejected}</p>
                  <p className="text-[10px] text-slate-400">Agent Rejected</p>
                </div>
              </>
            )}
          </div>
        )}

        {/* Spinner for initial startup */}
        {phase === "running" && !progress && (
          <div className="flex justify-center py-4">
            <Loader2 size={28} className="animate-spin text-indigo-400" />
          </div>
        )}

        {/* Done actions */}
        {canClose && (
          <div className="flex justify-end pt-1">
            <button onClick={onClose} className="btn-primary text-sm py-2 px-5">
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
