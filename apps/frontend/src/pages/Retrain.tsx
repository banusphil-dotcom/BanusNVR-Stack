import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api, getToken } from "../api";
import {
  ArrowLeft, ArrowRight, Check, X, AlertTriangle,
  RefreshCw, Shield, ChevronLeft, ChevronRight, RotateCcw,
  CheckCircle2, XCircle, Loader2, Eye, Camera, User, Info, ImageUp,
} from "lucide-react";
import { useReExtractModal, ReExtractModal } from "../components/ReExtractModal";

/* ── Types ── */

interface Detection {
  event_id: number;
  camera_name: string;
  confidence: number | null;
  face_similarity: number | null;
  body_similarity: number | null;
  best_similarity: number;
  thumbnail_url: string;
  snapshot_url: string | null;
  timestamp: string;
  flagged?: boolean;
  has_face?: boolean;
  system_matched?: boolean;
}

interface CoverageData {
  total: number;
  max_images: number;
  target_images: number;
  face_count: number;
  body_only_count: number;
  camera_count: number;
  cameras: string[];
  overall_score: number;
  face_score: number;
  body_score: number | null;
  camera_score: number;
  status: "poor" | "needs_work" | "good" | "excellent";
  tips: string[];
}

interface ExistingResponse {
  object_id: number;
  object_name: string;
  category: string;
  current_refs: number;
  detections: Detection[];
  total: number;
  coverage?: CoverageData;
}

interface ScanResponse {
  object_id: number;
  object_name: string;
  candidates: Detection[];
  scanned: number;
  hours: number;
}

interface CommitResponse {
  object_id: number;
  object_name: string;
  total_confirmed: number;
  trained_face: number;
  trained_body: number;
  reference_count: number;
  unlinked: number;
  max_images?: number;
  capped?: boolean;
  coverage?: CoverageData;
}

/* ── Wizard Steps ── */

type Step = "review" | "scan" | "final" | "done";

const PAGE_SIZE = 6;
const BATCH_SIZE = 20;

export default function Retrain() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const objectId = Number(id);
  const token = getToken();

  const [step, setStep] = useState<Step>("review");
  const [confirmedIds, setConfirmedIds] = useState<Set<number>>(new Set());
  const [rejectedIds, setRejectedIds] = useState<Set<number>>(new Set());
  const [newConfirmedIds, setNewConfirmedIds] = useState<Set<number>>(new Set());
  const [newRejectedIds, setNewRejectedIds] = useState<Set<number>>(new Set());
  const [reviewPage, setReviewPage] = useState(0);
  const [scanPage, setScanPage] = useState(0);
  const [finalPage, setFinalPage] = useState(0);
  const [rescoring, setRescoring] = useState(false);
  const lastRescoreCount = useRef(0);
  const [rescoreMsg, setRescoreMsg] = useState<string | null>(null);
  const [rescoreCounter, setRescoreCounter] = useState(0);

  /* ── Step 1: Load existing detections ── */
  const { data: existing, isLoading: loadingExisting, refetch: refetchExisting } = useQuery({
    queryKey: ["deep-retrain-existing", objectId],
    queryFn: () => api.post<ExistingResponse>(`/api/training/objects/${objectId}/deep-retrain/existing`),
    enabled: !!objectId,
  });

  /* ── Step 2: Scan for new candidates (streaming NDJSON) ── */
  const [scanCandidates, setScanCandidates] = useState<Detection[]>([]);
  const [scanLoading, setScanLoading] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState<{ scanned: number; total: number; found: number } | null>(null);

  const startScan = useCallback(async () => {
    setScanLoading(true);
    setScanError(null);
    setScanCandidates([]);
    setScanProgress(null);

    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;
      const res = await fetch(`/api/training/objects/${objectId}/deep-retrain/scan?hours=24`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          exclude_event_ids: (existing?.detections ?? []).map((d: Detection) => d.event_id),
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || res.statusText);
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;
        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line);
          if (msg.type === "progress") {
            setScanProgress({ scanned: msg.scanned, total: msg.total, found: msg.found });
            if (msg.new_candidates?.length) {
              setScanCandidates(prev => [...prev, ...msg.new_candidates]);
              // Auto-confirm system-matched candidates (user can still reject)
              const autoConfirm = msg.new_candidates.filter((c: Detection) => c.system_matched).map((c: Detection) => c.event_id);
              if (autoConfirm.length) {
                setNewConfirmedIds(prev => {
                  const next = new Set(prev);
                  for (const id of autoConfirm) next.add(id);
                  return next;
                });
              }
            }
          } else if (msg.type === "done" || msg.type === "result") {
            if (msg.candidates) {
              setScanCandidates(msg.candidates);
            }
          }
        }
      }
    } catch (err: unknown) {
      setScanError(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setScanLoading(false);
    }
  }, [objectId, token, existing]);

  /* ── Step 3: Commit ── */
  const commitMut = useMutation({
    mutationFn: (payload: { confirmed_event_ids: number[]; new_event_ids: number[] }) =>
      api.post<CommitResponse>(`/api/training/objects/${objectId}/deep-retrain/commit`, payload),
    onSuccess: () => setStep("done"),
  });

  /* ── Rescore: re-rank remaining candidates using confirmed images ── */
  const doRescore = useCallback(async (
    confirmed: Set<number>,
    _rejected: Set<number>,
    detections: Detection[],
  ) => {
    if (confirmed.size < 2) return;

    setRescoring(true);
    setRescoreMsg("Re-analysing with updated model...");

    try {
      const unreviewed = detections
        .filter(d => !confirmed.has(d.event_id) && !_rejected.has(d.event_id))
        .map(d => d.event_id);
      if (unreviewed.length === 0) { setRescoring(false); setRescoreMsg(null); return; }

      const res = await api.post<{ rescored: { event_id: number; best_similarity: number; face_similarity: number | null; body_similarity: number | null }[] }>(
        `/api/training/objects/${objectId}/deep-retrain/rescore`,
        { confirmed_ids: Array.from(confirmed), rejected_ids: Array.from(_rejected), unreviewed_ids: unreviewed },
      );

      // Update scores in detection list (no auto-approve/reject — user decides)
      const scoreMap = new Map(res.rescored.map(r => [r.event_id, r]));
      for (const d of detections) {
        const updated = scoreMap.get(d.event_id);
        if (updated) {
          d.best_similarity = updated.best_similarity;
          d.face_similarity = updated.face_similarity;
          d.body_similarity = updated.body_similarity;
        }
      }

      setRescoreMsg(`Re-analysed ${res.rescored.length} candidates — scores updated`);      setRescoreCounter(c => c + 1);      setTimeout(() => setRescoreMsg(null), 3000);
    } catch {
      setRescoreMsg(null);
    } finally {
      setRescoring(false);
    }
  }, [objectId]);

  /* ── Derived data ── */
  const reviewDetections = existing?.detections ?? [];
  const unreviewedExisting = reviewDetections.filter(
    (d) => !confirmedIds.has(d.event_id) && !rejectedIds.has(d.event_id)
  );
  const reviewPages = Math.ceil(reviewDetections.length / PAGE_SIZE);
  const currentReviewPage = reviewDetections.slice(reviewPage * PAGE_SIZE, (reviewPage + 1) * PAGE_SIZE);

  const unreviewedScan = scanCandidates.filter(
    (d) => !newConfirmedIds.has(d.event_id) && !newRejectedIds.has(d.event_id)
  );

  // Final review: all confirmed images from both steps (must be before currentBatch)
  const allConfirmed = useMemo(() => {
    const items: Detection[] = [];
    for (const d of reviewDetections) {
      if (confirmedIds.has(d.event_id)) items.push(d);
    }
    for (const d of scanCandidates) {
      if (newConfirmedIds.has(d.event_id)) items.push(d);
    }
    return items;
  }, [reviewDetections, scanCandidates, confirmedIds, newConfirmedIds]);

  // Batch mode: prioritise under-represented categories for balanced coverage
  const currentBatch = useMemo(() => {
    const sorted = [...unreviewedScan].sort((a, b) => b.best_similarity - a.best_similarity);
    if (existing?.category !== "person") return sorted.slice(0, BATCH_SIZE);

    // Current confirmed coverage stats
    const confirmedFaces = allConfirmed.filter(d => d.has_face).length;
    const confirmedBodies = allConfirmed.filter(d => !d.has_face).length;
    const confirmedCameras = new Set(allConfirmed.map(d => d.camera_name));
    const needFaces = confirmedFaces < 15;
    const needBodies = confirmedBodies < 8;
    const needCameras = confirmedCameras.size < 3;

    if (!needFaces && !needBodies && !needCameras) return sorted.slice(0, BATCH_SIZE);

    // Score each candidate by how much they help coverage gaps
    const scored = sorted.map(d => {
      let priority = 0;
      if (needFaces && d.has_face) priority += 3;           // Highest: face needed & present
      if (needBodies && !d.has_face) priority += 2;          // Second: body needed & no face
      if (needCameras && !confirmedCameras.has(d.camera_name)) priority += 1; // Boost new cameras
      return { d, priority };
    });

    // Sort by priority desc, then by similarity desc within same priority
    scored.sort((a, b) => b.priority - a.priority || b.d.best_similarity - a.d.best_similarity);

    return scored.slice(0, BATCH_SIZE).map(s => s.d);
  }, [unreviewedScan, rescoreCounter, allConfirmed, existing?.category]);

  const finalPages = Math.ceil(allConfirmed.length / PAGE_SIZE);
  const currentFinalPage = allConfirmed.slice(finalPage * PAGE_SIZE, (finalPage + 1) * PAGE_SIZE);

  /* \u2500\u2500 Auto-rescore trigger \u2500\u2500 */
  const reviewedCount = confirmedIds.size + rejectedIds.size;

  // Step 2 batch rescore: when all BATCH_SIZE in current batch are reviewed, rescore remaining
  const scanReviewedCount = newConfirmedIds.size + newRejectedIds.size;
  const lastScanRescoreCount = useRef(0);
  useEffect(() => {
    if (step !== "scan" || rescoring || scanLoading) return;
    // Only trigger when user has finished a full batch (every BATCH_SIZE reviews)
    const reviewedSinceLastRescore = scanReviewedCount - lastScanRescoreCount.current;
    if (reviewedSinceLastRescore >= BATCH_SIZE && unreviewedScan.length > 0) {
      lastScanRescoreCount.current = scanReviewedCount;
      const allConf = new Set([...confirmedIds, ...newConfirmedIds]);
      if (allConf.size >= 2) {
        doRescore(allConf, newRejectedIds, scanCandidates);
      }
    }
  }, [scanReviewedCount, step, rescoring, scanLoading, confirmedIds, newConfirmedIds, newRejectedIds, scanCandidates, unreviewedScan.length, doRescore]);

  /* ── Helpers ── */
  const imgUrl = (path: string) =>
    `${path}?token=${encodeURIComponent(token || "")}`;

  const simBadge = (d: Detection) => {
    const s = d.best_similarity;
    const color = s >= 0.6 ? "bg-green-900/60 text-green-300" :
                  s >= 0.4 ? "bg-yellow-900/60 text-yellow-300" :
                  "bg-red-900/60 text-red-300";
    return (
      <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${color}`}>
        {(s * 100).toFixed(0)}%
        {d.face_similarity != null && d.body_similarity != null && (
          <span className="opacity-60 ml-1">
            F{(d.face_similarity * 100).toFixed(0)} B{(d.body_similarity * 100).toFixed(0)}
          </span>
        )}
      </span>
    );
  };

  /* ── Live coverage from confirmed images ── */
  const liveCoverage = useMemo((): CoverageData | null => {
    if (allConfirmed.length === 0) return null;
    const cat = existing?.category ?? "other";
    const maxImg = ({ person: 100, pet: 40, vehicle: 30, other: 30 } as Record<string, number>)[cat] ?? 30;
    const targetImg = ({ person: 60, pet: 25, vehicle: 20, other: 20 } as Record<string, number>)[cat] ?? 20;
    const faceCount = allConfirmed.filter(d => d.has_face).length;
    const bodyOnly = allConfirmed.filter(d => !d.has_face).length;
    const cameras = [...new Set(allConfirmed.map(d => d.camera_name))];

    let faceScr = 0, bodyScr = 0, camScr = 0, overall = 0;
    if (cat === "person") {
      faceScr = Math.min(faceCount / 15, 1) * 100;
      bodyScr = Math.min(bodyOnly / 8, 1) * 100;
      camScr = Math.min(cameras.length / 3, 1) * 100;
      overall = faceScr * 0.45 + bodyScr * 0.30 + camScr * 0.25;
    } else {
      const variety = Math.min(allConfirmed.length / targetImg, 1) * 100;
      camScr = Math.min(cameras.length / 3, 1) * 100;
      overall = variety * 0.70 + camScr * 0.30;
    }

    const tips: string[] = [];
    if (cat === "person") {
      if (faceCount < 5) tips.push("Needs more front-facing photos where the face is clearly visible");
      else if (faceCount < 15) tips.push(`More face photos would help (${faceCount}/15 ideal)`);
      if (bodyOnly < 3) tips.push("Needs photos showing the back or side (no face visible) for body matching");
      else if (bodyOnly < 8) tips.push(`More back/side views would improve body recognition (${bodyOnly}/8 ideal)`);
    } else if (cat === "pet") {
      if (allConfirmed.length < 10) tips.push("Needs more photos from different angles — front, side, and back");
      else if (allConfirmed.length < targetImg) tips.push(`More varied photos will improve accuracy (${allConfirmed.length}/${targetImg} ideal)`);
    } else {
      if (allConfirmed.length < 10) tips.push("Needs more photos from different angles and distances");
    }
    if (cameras.length < 2) tips.push("Try to include photos from different cameras for varied lighting");
    else if (cameras.length < 3) tips.push(`Photos from more cameras help (${cameras.length}/3+ ideal)`);
    if (allConfirmed.length >= targetImg && tips.length === 0) tips.push("Model has good coverage — ready for accurate detection");

    const status = overall >= 80 ? "excellent" : overall >= 55 ? "good" : overall >= 30 ? "needs_work" : "poor";
    return {
      total: allConfirmed.length, max_images: maxImg, target_images: targetImg,
      face_count: faceCount, body_only_count: bodyOnly, camera_count: cameras.length,
      cameras, overall_score: Math.round(overall),
      face_score: Math.round(faceScr), body_score: cat === "person" ? Math.round(bodyScr) : null,
      camera_score: Math.round(camScr), status: status as CoverageData["status"], tips,
    };
  }, [allConfirmed, existing?.category]);

  /* ── Coverage Card renderer ── */
  const renderCoverage = (cov: CoverageData, isPerson: boolean, compact = false) => {
    const statusColor = { poor: "text-red-400", needs_work: "text-yellow-400", good: "text-blue-400", excellent: "text-green-400" }[cov.status];
    const statusLabel = { poor: "Poor", needs_work: "Needs Work", good: "Good", excellent: "Excellent" }[cov.status];
    const barColor = { poor: "bg-red-500", needs_work: "bg-yellow-500", good: "bg-blue-500", excellent: "bg-green-500" }[cov.status];

    return (
      <div className="card p-3 space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-semibold flex items-center gap-1.5">
            <Eye size={14} className="text-blue-400" /> Model Coverage
          </h4>
          <span className={`text-xs font-bold ${statusColor}`}>{statusLabel} — {cov.overall_score}%</span>
        </div>

        {/* Overall bar */}
        <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
          <div className={`h-full ${barColor} transition-all duration-500`} style={{ width: `${cov.overall_score}%` }} />
        </div>

        {/* Stats row */}
        <div className="flex gap-3 text-[10px] flex-wrap">
          <span className="text-slate-400">
            {cov.total}/{cov.target_images} target · {cov.max_images} max
          </span>
          {isPerson && (
            <>
              <span className="text-blue-300 flex items-center gap-0.5">
                <User size={10} /> Face: {cov.face_count}/15
              </span>
              <span className="text-amber-300 flex items-center gap-0.5">
                <RotateCcw size={10} /> Back/side: {cov.body_only_count}/8
              </span>
            </>
          )}
          <span className="text-purple-300 flex items-center gap-0.5">
            <Camera size={10} /> Cameras: {cov.camera_count}/3+
          </span>
        </div>

        {/* Sub-bars for person category */}
        {isPerson && !compact && (
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-[10px]">
              <span className="w-14 text-slate-400">Face</span>
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 transition-all" style={{ width: `${cov.face_score}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-2 text-[10px]">
              <span className="w-14 text-slate-400">Body</span>
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-amber-500 transition-all" style={{ width: `${cov.body_score ?? 0}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-2 text-[10px]">
              <span className="w-14 text-slate-400">Cameras</span>
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-purple-500 transition-all" style={{ width: `${cov.camera_score}%` }} />
              </div>
            </div>
          </div>
        )}

        {/* Tips */}
        {cov.tips.length > 0 && (
          <div className="space-y-1 pt-1">
            {cov.tips.map((tip, i) => (
              <p key={i} className="text-[11px] flex items-start gap-1.5">
                <Info size={12} className={tip.includes("good coverage") ? "text-green-400 shrink-0 mt-0.5" : "text-yellow-400 shrink-0 mt-0.5"} />
                <span className={tip.includes("good coverage") ? "text-green-300" : "text-slate-300"}>{tip}</span>
              </p>
            ))}
          </div>
        )}
      </div>
    );
  };

  const toggleConfirm = (eventId: number, confirmed: Set<number>, rejected: Set<number>,
    setConfirmed: (s: Set<number>) => void, setRejected: (s: Set<number>) => void) => {
    const nc = new Set(confirmed);
    const nr = new Set(rejected);
    if (nc.has(eventId)) {
      nc.delete(eventId);
    } else {
      nc.add(eventId);
      nr.delete(eventId);
    }
    setConfirmed(nc);
    setRejected(nr);
  };

  const toggleReject = (eventId: number, confirmed: Set<number>, rejected: Set<number>,
    setConfirmed: (s: Set<number>) => void, setRejected: (s: Set<number>) => void) => {
    const nc = new Set(confirmed);
    const nr = new Set(rejected);
    if (nr.has(eventId)) {
      nr.delete(eventId);
    } else {
      nr.add(eventId);
      nc.delete(eventId);
    }
    setConfirmed(nc);
    setRejected(nr);
  };

  const confirmAllOnPage = (page: Detection[], confirmed: Set<number>, rejected: Set<number>,
    setConfirmed: (s: Set<number>) => void, setRejected: (s: Set<number>) => void) => {
    const nc = new Set(confirmed);
    const nr = new Set(rejected);
    for (const d of page) {
      nc.add(d.event_id);
      nr.delete(d.event_id);
    }
    setConfirmed(nc);
    setRejected(nr);
  };

  const rejectAllOnPage = (page: Detection[], confirmed: Set<number>, rejected: Set<number>,
    setConfirmed: (s: Set<number>) => void, setRejected: (s: Set<number>) => void) => {
    const nc = new Set(confirmed);
    const nr = new Set(rejected);
    for (const d of page) {
      nr.add(d.event_id);
      nc.delete(d.event_id);
    }
    setConfirmed(nc);
    setRejected(nr);
  };

  /* ── Proceed logic ── */
  const canProceedFromReview = unreviewedExisting.length === 0 && confirmedIds.size >= 3;

  const handleStartScan = () => {
    setStep("scan");
    setScanPage(0);
    startScan();
  };

  const handleFinalReview = () => {
    setStep("final");
    setFinalPage(0);
  };

  const removeFinalImage = (eventId: number) => {
    if (confirmedIds.has(eventId)) {
      const nc = new Set(confirmedIds);
      nc.delete(eventId);
      const nr = new Set(rejectedIds);
      nr.add(eventId);
      setConfirmedIds(nc);
      setRejectedIds(nr);
    } else if (newConfirmedIds.has(eventId)) {
      const nc = new Set(newConfirmedIds);
      nc.delete(eventId);
      const nr = new Set(newRejectedIds);
      nr.add(eventId);
      setNewConfirmedIds(nc);
      setNewRejectedIds(nr);
    }
  };

  const handleCommit = () => {
    const existingConfirmed = Array.from(confirmedIds);
    const newIds = Array.from(newConfirmedIds);
    commitMut.mutate({ confirmed_event_ids: existingConfirmed, new_event_ids: newIds });
  };

  /* ── Render helpers ── */
  const renderImageGrid = (
    items: Detection[],
    confirmed: Set<number>,
    rejected: Set<number>,
    setConfirmed: (s: Set<number>) => void,
    setRejected: (s: Set<number>) => void,
    readOnly = false,
  ) => (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
      {items.map((d) => {
        const isConfirmed = confirmed.has(d.event_id);
        const isRejected = rejected.has(d.event_id);
        const border = isConfirmed ? "ring-2 ring-green-500" : isRejected ? "ring-2 ring-red-500" : "ring-1 ring-slate-700";

        return (
          <div key={d.event_id} className={`relative rounded-lg overflow-hidden ${border} bg-slate-900`}>
            <img
              src={imgUrl(d.thumbnail_url)}
              alt=""
              loading="lazy"
              className="w-full aspect-[3/4] object-contain bg-slate-950"
              onError={(e) => { (e.target as HTMLImageElement).src = ""; }}
            />

            {/* Overlay badges */}
            <div className="absolute top-1 left-1 flex flex-col gap-0.5">
              {simBadge(d)}
              {d.system_matched && (
                <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-purple-900/70 text-purple-300">
                  Auto
                </span>
              )}
              {existing?.category === "person" && (
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                  d.has_face ? "bg-blue-900/70 text-blue-300" : "bg-amber-900/70 text-amber-300"
                }`}>
                  {d.has_face ? "Face" : "Body"}
                </span>
              )}
            </div>
            <div className="absolute top-1 right-1">
              {isConfirmed && <CheckCircle2 size={20} className="text-green-400 drop-shadow" />}
              {isRejected && <XCircle size={20} className="text-red-400 drop-shadow" />}
            </div>

            {/* Info bar */}
            <div className="p-1.5 bg-slate-900/90">
              <p className="text-[10px] text-slate-400 truncate">{d.camera_name}</p>
              <p className="text-[10px] text-slate-500">{new Date(d.timestamp).toLocaleString()}</p>
            </div>

            {/* Action buttons — always visible on mobile */}
            {!readOnly && (
              <div className="absolute bottom-10 left-0 right-0 flex justify-center gap-2 sm:opacity-0 sm:hover:opacity-100 transition-opacity">
                <button
                  onClick={() => toggleConfirm(d.event_id, confirmed, rejected, setConfirmed, setRejected)}
                  className={`p-2 rounded-full shadow-lg ${
                    isConfirmed ? "bg-green-600" : "bg-slate-800/90 hover:bg-green-700"
                  }`}
                >
                  <Check size={18} className="text-white" />
                </button>
                <button
                  onClick={() => toggleReject(d.event_id, confirmed, rejected, setConfirmed, setRejected)}
                  className={`p-2 rounded-full shadow-lg ${
                    isRejected ? "bg-red-600" : "bg-slate-800/90 hover:bg-red-700"
                  }`}
                >
                  <X size={18} className="text-white" />
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  const renderPagination = (current: number, total: number, setCurrent: (n: number) => void) => (
    <div className="flex items-center justify-between mt-4">
      <button
        disabled={current === 0}
        onClick={() => setCurrent(current - 1)}
        className="btn-secondary text-sm py-1.5 px-3 flex items-center gap-1 disabled:opacity-30"
      >
        <ChevronLeft size={16} /> Prev
      </button>
      <span className="text-sm text-slate-400">{current + 1} / {total || 1}</span>
      <button
        disabled={current >= total - 1}
        onClick={() => setCurrent(current + 1)}
        className="btn-secondary text-sm py-1.5 px-3 flex items-center gap-1 disabled:opacity-30"
      >
        Next <ChevronRight size={16} />
      </button>
    </div>
  );

  /* ── Step indicators ── */
  const steps: { key: Step; label: string; num: number }[] = [
    { key: "review", label: "Review Existing", num: 1 },
    { key: "scan", label: "Find New", num: 2 },
    { key: "final", label: "Confirm & Commit", num: 3 },
  ];

  const reExtract = useReExtractModal();

  return (
    <div className="p-4 space-y-4 max-w-2xl mx-auto">
      {/* Re-extract modal */}
      <ReExtractModal
        open={reExtract.open} phase={reExtract.phase} scan={reExtract.scan}
        progress={reExtract.progress} result={reExtract.result} error={reExtract.error}
        onClose={reExtract.close}
      />
      {/* Header */}
      <div className="flex items-center gap-3">
        <button onClick={() => navigate(-1)} className="btn-secondary p-2">
          <ArrowLeft size={18} />
        </button>
        <div className="flex-1">
          <h2 className="text-lg font-bold flex items-center gap-2">
            <Shield size={20} className="text-blue-400" />
            Deep Retrain{existing ? `: ${existing.object_name}` : ""}
          </h2>
          <p className="text-xs text-slate-500">
            High-accuracy model rebuild — review every image before committing
          </p>
        </div>
        <button
          onClick={() => reExtract.start()}
          disabled={reExtract.phase !== "idle" && reExtract.phase !== "done" && reExtract.phase !== "error"}
          className="btn btn-sm bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3 py-1.5 rounded flex items-center gap-1.5 shrink-0"
          title="Re-extract HD thumbnails from recordings for better face detection"
        >
          <ImageUp size={14} />
          Re-extract HD
        </button>
      </div>

      {/* Step progress */}
      <div className="flex items-center gap-2">
        {steps.map((s, i) => {
          const active = step === s.key || (step === "done" && s.key === "final");
          const completed = steps.findIndex((x) => x.key === step) > i || step === "done";
          return (
            <div key={s.key} className="flex items-center gap-2 flex-1">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                completed ? "bg-green-600 text-white" : active ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-400"
              }`}>
                {completed ? <Check size={14} /> : s.num}
              </div>
              <span className={`text-xs truncate ${active ? "text-white font-medium" : "text-slate-500"}`}>
                {s.label}
              </span>
              {i < steps.length - 1 && <div className={`h-px flex-1 ${completed ? "bg-green-600" : "bg-slate-700"}`} />}
            </div>
          );
        })}
      </div>

      {/* ═══════════ Step 1: Review Existing ═══════════ */}
      {step === "review" && (
        <div className="space-y-4">
          <div className="card p-3 space-y-1">
            <h3 className="font-semibold text-sm">Review Existing Photos</h3>
            <p className="text-xs text-slate-400">
              Confirm each image is correct. Reject any that show the wrong person/object.
              All images must be reviewed before proceeding.
            </p>
            {existing && (
              <div className="flex gap-4 text-xs pt-1 flex-wrap">
                <span className="text-slate-400">Total: <span className="text-white font-medium">{existing.total}</span></span>
                <span className="text-green-400">Confirmed: {confirmedIds.size}</span>
                <span className="text-red-400">Rejected: {rejectedIds.size}</span>
                <span className="text-yellow-400">Remaining: {unreviewedExisting.length}</span>
              </div>
            )}
            {(rescoring || rescoreMsg) && (
              <div className="flex items-center gap-2 text-xs pt-1">
                {rescoring && <Loader2 size={14} className="animate-spin text-blue-400" />}
                <span className={rescoring ? "text-blue-400" : "text-green-400"}>{rescoreMsg}</span>
              </div>
            )}
          </div>

          {/* Coverage analysis from backend */}
          {existing?.coverage && renderCoverage(existing.coverage, existing.category === "person")}

          {loadingExisting ? (
            <div className="text-center py-12 text-slate-400 flex items-center justify-center gap-2">
              <Loader2 size={20} className="animate-spin" /> Analyzing detections...
            </div>
          ) : reviewDetections.length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              <p>No existing detections found.</p>
              <button onClick={handleStartScan} className="btn-primary mt-4">
                Skip to Scan
              </button>
            </div>
          ) : (
            <>
              {renderImageGrid(currentReviewPage, confirmedIds, rejectedIds, setConfirmedIds, setRejectedIds)}
              {renderPagination(reviewPage, reviewPages, setReviewPage)}

              {/* Bulk actions */}
              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={() => confirmAllOnPage(currentReviewPage, confirmedIds, rejectedIds, setConfirmedIds, setRejectedIds)}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <Check size={14} /> Confirm All on Page
                </button>
                <button
                  onClick={() => rejectAllOnPage(currentReviewPage, confirmedIds, rejectedIds, setConfirmedIds, setRejectedIds)}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <X size={14} /> Reject All on Page
                </button>
                <button
                  onClick={() => {
                    const nc = new Set(confirmedIds);
                    for (const d of reviewDetections) {
                      if (d.best_similarity >= 0.5) nc.add(d.event_id);
                    }
                    setConfirmedIds(nc);
                    setRejectedIds(new Set(
                      reviewDetections.filter(d => d.best_similarity < 0.5 && !nc.has(d.event_id)).map(d => d.event_id)
                    ));
                  }}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <AlertTriangle size={14} /> Auto-approve &gt;50%
                </button>
              </div>

              {/* Proceed */}
              {canProceedFromReview && existing?.coverage && confirmedIds.size < existing.coverage.target_images && (
                <p className="text-xs text-yellow-400 text-center">
                  {confirmedIds.size}/{existing.coverage.target_images} target photos confirmed —
                  scan will find more to reach the target
                </p>
              )}
              <div className="flex justify-end">
                <button
                  onClick={handleStartScan}
                  disabled={!canProceedFromReview}
                  className="btn-primary flex items-center gap-2 disabled:opacity-40"
                >
                  {canProceedFromReview ? (
                    <>Scan 24h for New Matches <ArrowRight size={16} /></>
                  ) : unreviewedExisting.length > 0 ? (
                    <>{unreviewedExisting.length} images still need review</>
                  ) : (
                    <>Need at least 3 confirmed images</>
                  )}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ═══════════ Step 2: Scan for New ═══════════ */}
      {step === "scan" && (
        <div className="space-y-4">
          <div className="card p-3 space-y-1">
            <h3 className="font-semibold text-sm">Scanning 24 Hours of Recordings</h3>
            <p className="text-xs text-slate-400">
              Review {BATCH_SIZE} candidates at a time. After each batch the model
              re-analyses remaining candidates with your feedback.
            </p>
            {(scanCandidates.length > 0 || scanProgress) && (
              <div className="flex gap-4 text-xs pt-1 flex-wrap">
                <span className="text-slate-400">Total found: <span className="text-white font-medium">{scanCandidates.length || scanProgress?.found || 0}</span></span>
                <span className="text-green-400">Confirmed: {newConfirmedIds.size}</span>
                <span className="text-red-400">Rejected: {newRejectedIds.size}</span>
                <span className="text-yellow-400">Remaining: {unreviewedScan.length}</span>
              </div>
            )}
            {(rescoring || rescoreMsg) && (
              <div className="flex items-center gap-2 text-xs pt-1">
                {rescoring && <Loader2 size={14} className="animate-spin text-blue-400" />}
                <span className={rescoring ? "text-blue-400" : "text-green-400"}>{rescoreMsg}</span>
              </div>
            )}
          </div>

          {/* Live coverage — updates as user confirms/rejects */}
          {liveCoverage && renderCoverage(liveCoverage, existing?.category === "person", true)}

          {/* Compact progress bar during scan (shown with candidates) */}
          {scanLoading && scanCandidates.length > 0 && scanProgress && (
            <div className="card p-2.5 flex items-center gap-3">
              <Loader2 size={16} className="animate-spin text-blue-400 shrink-0" />
              <div className="flex-1">
                <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${scanProgress.total ? (scanProgress.scanned / scanProgress.total) * 100 : 0}%` }} />
                </div>
                <p className="text-[11px] text-slate-500 mt-0.5">{scanProgress.scanned}/{scanProgress.total} scanned — {scanCandidates.length} matches found so far</p>
              </div>
            </div>
          )}

          {scanLoading && scanCandidates.length === 0 ? (
            <div className="text-center py-12 text-slate-400 flex flex-col items-center gap-3">
              <Loader2 size={28} className="animate-spin" />
              <p>Scanning recordings with face + body recognition...</p>
              {scanProgress && (
                <>
                  <div className="w-48 h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${scanProgress.total ? (scanProgress.scanned / scanProgress.total) * 100 : 0}%` }} />
                  </div>
                  <p className="text-xs text-slate-500">{scanProgress.scanned} / {scanProgress.total} events — {scanProgress.found} matches found</p>
                </>
              )}
            </div>
          ) : scanError ? (
            <div className="text-center py-12 text-red-400 space-y-3">
              <p>Scan failed: {scanError}</p>
              <button onClick={() => startScan()} className="btn-secondary">
                <RefreshCw size={14} /> Retry
              </button>
            </div>
          ) : rescoring ? (
            <div className="text-center py-12 text-slate-400 flex flex-col items-center gap-3">
              <Loader2 size={28} className="animate-spin" />
              <p>Re-analysing remaining candidates...</p>
            </div>
          ) : scanCandidates.length === 0 ? (
            <div className="text-center py-12 text-slate-500 space-y-3">
              <p>No new candidates found in the last 24 hours.</p>
              <p className="text-xs">That's okay — the model will be rebuilt from your confirmed images.</p>
              <button onClick={handleFinalReview} className="btn-primary mt-2">
                Proceed to Final Review <ArrowRight size={16} />
              </button>
            </div>
          ) : unreviewedScan.length === 0 ? (
            <div className="text-center py-12 text-slate-500 space-y-3">
              <p>All candidates reviewed!</p>
              <div className="flex gap-4 text-sm justify-center">
                <span className="text-green-400">✓ {newConfirmedIds.size} confirmed</span>
                <span className="text-red-400">✗ {newRejectedIds.size} rejected</span>
              </div>
              <button onClick={handleFinalReview} className="btn-primary mt-2">
                Proceed to Final Review <ArrowRight size={16} />
              </button>
            </div>
          ) : (
            <>
              {/* Target reached notification */}
              {liveCoverage && allConfirmed.length >= liveCoverage.target_images && (
                <div className="card p-3 bg-green-900/20 border border-green-800/40 flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-green-400 shrink-0" />
                  <p className="text-xs text-green-300">
                    Target reached ({allConfirmed.length}/{liveCoverage.target_images} photos).
                    {allConfirmed.length >= liveCoverage.max_images
                      ? " Max limit reached — proceed to final review."
                      : " You can continue reviewing or skip to final."}
                  </p>
                </div>
              )}

              {/* Priority indicator for coverage-driven batch ordering */}
              {existing?.category === "person" && liveCoverage && (() => {
                const needFaces = liveCoverage.face_count < 15;
                const needBodies = (liveCoverage.body_only_count ?? 0) < 8;
                const needCameras = liveCoverage.camera_count < 3;
                if (!needFaces && !needBodies && !needCameras) return null;
                const facesInBatch = currentBatch.filter(d => d.has_face).length;
                const bodiesInBatch = currentBatch.filter(d => !d.has_face).length;
                const newCamsInBatch = new Set(currentBatch.map(d => d.camera_name).filter(c => !liveCoverage.cameras.includes(c))).size;
                const parts: string[] = [];
                if (needFaces) parts.push(`faces ${liveCoverage.face_count}/15`);
                if (needBodies) parts.push(`body shots ${liveCoverage.body_only_count}/8`);
                if (needCameras) parts.push(`cameras ${liveCoverage.camera_count}/3`);
                return (
                  <div className="card p-2.5 bg-blue-900/20 border border-blue-800/40 flex items-center gap-2">
                    <User size={14} className="text-blue-400 shrink-0" />
                    <p className="text-[11px] text-blue-300">
                      Prioritising: {parts.join(", ")}.
                      {" "}Batch: {facesInBatch} face, {bodiesInBatch} body{newCamsInBatch > 0 ? `, ${newCamsInBatch} new camera${newCamsInBatch > 1 ? "s" : ""}` : ""}.
                    </p>
                  </div>
                );
              })()}

              <div className="text-xs text-slate-400 text-center">
                Batch {Math.floor((newConfirmedIds.size + newRejectedIds.size) / BATCH_SIZE) + 1}
                {" · "}{currentBatch.length} to review · {unreviewedScan.length} remaining
              </div>

              {renderImageGrid(currentBatch, newConfirmedIds, newRejectedIds, setNewConfirmedIds, setNewRejectedIds)}

              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={() => confirmAllOnPage(currentBatch, newConfirmedIds, newRejectedIds, setNewConfirmedIds, setNewRejectedIds)}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <Check size={14} /> Confirm All {currentBatch.length}
                </button>
                <button
                  onClick={() => rejectAllOnPage(currentBatch, newConfirmedIds, newRejectedIds, setNewConfirmedIds, setNewRejectedIds)}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <X size={14} /> Reject All {currentBatch.length}
                </button>
              </div>

              <div className="flex justify-between">
                <button onClick={() => setStep("review")} className="btn-secondary flex items-center gap-1">
                  <ArrowLeft size={16} /> Back
                </button>
                <button
                  onClick={handleFinalReview}
                  className={`btn-primary flex items-center gap-2 ${
                    liveCoverage && allConfirmed.length >= liveCoverage.target_images
                      ? "bg-green-600 hover:bg-green-700 animate-pulse" : ""
                  }`}
                >
                  Skip Remaining · Final Review <ArrowRight size={16} />
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ═══════════ Step 3: Final Confirmation ═══════════ */}
      {step === "final" && (
        <div className="space-y-4">
          <div className="card p-3 space-y-1">
            <h3 className="font-semibold text-sm">Final Review Before Commit</h3>
            <p className="text-xs text-slate-400">
              These are all the images that will be used to rebuild the model.
              The old model will be completely replaced. This is your last chance to remove any bad images.
            </p>
            <div className="flex gap-4 text-xs pt-1">
              <span className="text-green-400">
                Total images: <span className="font-bold">{allConfirmed.length}</span>
              </span>
              <span className="text-slate-400">
                From existing: {confirmedIds.size}
              </span>
              <span className="text-blue-400">
                New matches: {newConfirmedIds.size}
              </span>
            </div>
          </div>

          {/* Live coverage for final review */}
          {liveCoverage && renderCoverage(liveCoverage, existing?.category === "person")}

          {allConfirmed.length < 3 ? (
            <div className="text-center py-12 text-red-400 space-y-3">
              <AlertTriangle size={32} className="mx-auto" />
              <p>Need at least 3 confirmed images to rebuild the model.</p>
              <p className="text-xs text-slate-500">Go back and confirm more images.</p>
              <button onClick={() => setStep("review")} className="btn-secondary mt-2">
                <ArrowLeft size={14} /> Back to Review
              </button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {currentFinalPage.map((d) => (
                  <div key={d.event_id} className="relative rounded-lg overflow-hidden ring-2 ring-green-500 bg-slate-900">
                    <img src={imgUrl(d.thumbnail_url)} alt="" loading="lazy" className="w-full aspect-[3/4] object-contain bg-slate-950" onError={(e) => { (e.target as HTMLImageElement).src = ""; }} />
                    <div className="absolute top-1 left-1 flex flex-col gap-0.5">
                      {simBadge(d)}
                      {existing?.category === "person" && (
                        <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${d.has_face ? "bg-blue-900/70 text-blue-300" : "bg-amber-900/70 text-amber-300"}`}>{d.has_face ? "Face" : "Body"}</span>
                      )}
                    </div>
                    <div className="p-1.5 bg-slate-900/90">
                      <p className="text-[10px] text-slate-400 truncate">{d.camera_name}</p>
                      <p className="text-[10px] text-slate-500">{new Date(d.timestamp).toLocaleString()}</p>
                    </div>
                    <div className="absolute bottom-10 left-0 right-0 flex justify-center">
                      <button onClick={() => removeFinalImage(d.event_id)} className="p-2 rounded-full shadow-lg bg-slate-800/90 hover:bg-red-700" title="Remove from training">
                        <X size={18} className="text-white" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              {renderPagination(finalPage, finalPages, setFinalPage)}

              <p className="text-xs text-slate-500">
                Tap <X size={12} className="inline" /> to remove any bad images before committing.
              </p>

              <div className="flex justify-between items-center pt-2">
                <button onClick={() => setStep("scan")} className="btn-secondary flex items-center gap-1">
                  <ArrowLeft size={16} /> Back
                </button>
                <button
                  onClick={handleCommit}
                  disabled={commitMut.isPending || allConfirmed.length < 3}
                  className="btn-primary flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:opacity-40"
                >
                  {commitMut.isPending ? (
                    <><Loader2 size={16} className="animate-spin" /> Rebuilding Model...</>
                  ) : (
                    <><Shield size={16} /> Commit {allConfirmed.length} Images &amp; Rebuild</>
                  )}
                </button>
              </div>

              {commitMut.isError && (
                <p className="text-xs text-red-400 text-center">
                  Error: {commitMut.error?.message}
                </p>
              )}
            </>
          )}
        </div>
      )}

      {/* ═══════════ Done ═══════════ */}
      {step === "done" && commitMut.data && (
        <div className="text-center py-12 space-y-4">
          <CheckCircle2 size={48} className="mx-auto text-green-400" />
          <h3 className="text-lg font-bold text-green-300">Model Rebuilt Successfully</h3>
          <div className="card p-4 text-left mx-auto max-w-sm space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Confirmed images:</span>
              <span className="font-medium">{commitMut.data.total_confirmed}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Face embeddings:</span>
              <span className="font-medium">{commitMut.data.trained_face}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Body embeddings:</span>
              <span className="font-medium">{commitMut.data.trained_body}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Reference count:</span>
              <span className="font-medium text-green-400">{commitMut.data.reference_count}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Removed mismatches:</span>
              <span className="font-medium text-red-400">{commitMut.data.unlinked}</span>
            </div>
            {commitMut.data.capped && (
              <div className="flex justify-between text-sm">
                <span className="text-yellow-400">Capped at max:</span>
                <span className="font-medium text-yellow-400">{commitMut.data.max_images}</span>
              </div>
            )}
          </div>
          {commitMut.data.coverage && renderCoverage(commitMut.data.coverage, existing?.category === "person")}
          <div className="flex gap-3 justify-center">
            <button
              onClick={() => navigate(`/profiles/${objectId}`)}
              className="btn-secondary flex items-center gap-1"
            >
              View Profile
            </button>
            <button
              onClick={() => navigate("/training")}
              className="btn-primary flex items-center gap-1"
            >
              Back to Training
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
