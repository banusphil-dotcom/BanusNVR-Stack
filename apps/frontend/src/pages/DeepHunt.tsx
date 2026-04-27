import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, getToken } from "../api";
import {
  Crosshair,
  Play,
  Square,
  Camera,
  Clock,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronDown,
  Image,
  User,
  Cat,
  Car,
  Box,
} from "lucide-react";

interface NamedObj {
  id: number;
  name: string;
  category: string;
  reference_image_count: number;
}

interface CameraInfo {
  id: number;
  name: string;
  enabled: boolean;
}

interface Sighting {
  index: number;
  timestamp: number;
  camera: string;
  confidence: number;
  det_confidence?: number;
  class_name: string;
  bbox?: { x1: number; y1: number; x2: number; y2: number };
  thumbnail_url: string;
}

interface HuntJob {
  job_id: string;
  target: string;
  status: string;
  progress: number;
  segments_total: number;
  segments_done: number;
  frames_scanned: number;
  detections_total: number;
  detections_relevant: number;
  sightings_count: number;
  error?: string;
  sightings?: Sighting[];
}

const CATEGORY_ICONS: Record<string, typeof User> = {
  person: User,
  pet: Cat,
  vehicle: Car,
  other: Box,
};

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatCamera(name: string): string {
  return name.replace("camera_", "Camera ");
}

function thumbUrl(url: string): string {
  const token = getToken();
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token || "")}`;
}

export default function DeepHunt() {
  const [selectedObject, setSelectedObject] = useState<number | null>(null);
  const [selectedCameras, setSelectedCameras] = useState<number[]>([]);
  const [hours, setHours] = useState(24);
  const [frameInterval, setFrameInterval] = useState(2.0);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobData, setJobData] = useState<HuntJob | null>(null);
  const [sightings, setSightings] = useState<Sighting[]>([]);
  const [previewIdx, setPreviewIdx] = useState<number | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Load named objects
  const { data: namedObjects } = useQuery({
    queryKey: ["named-objects-hunt"],
    queryFn: () => api.get<NamedObj[]>("/api/training/objects"),
  });

  // Load cameras
  const { data: cameras } = useQuery({
    queryKey: ["cameras-hunt"],
    queryFn: () => api.get<CameraInfo[]>("/api/cameras"),
  });

  // Start hunt
  const startHunt = useCallback(async () => {
    if (!selectedObject) return;
    const params = new URLSearchParams({
      named_object_id: String(selectedObject),
      hours: String(hours),
      frame_interval: String(frameInterval),
    });
    if (selectedCameras.length > 0) {
      params.set("camera_ids", selectedCameras.join(","));
    }
    try {
      const result = await api.post<{ job_id: string }>(`/api/search/deep-hunt?${params}`);
      setActiveJobId(result.job_id);
      setSightings([]);
      setJobData(null);
    } catch (e: any) {
      alert(e.message || "Failed to start hunt");
    }
  }, [selectedObject, selectedCameras, hours, frameInterval]);

  // SSE streaming for active job
  useEffect(() => {
    if (!activeJobId) return;

    const token = getToken();
    // Use polling instead of SSE since auth headers can't be sent with EventSource
    let cancelled = false;

    const poll = async () => {
      while (!cancelled) {
        try {
          const data = await api.get<HuntJob>(`/api/search/deep-hunt/${activeJobId}`);
          if (cancelled) break;
          setJobData(data);
          if (data.sightings) {
            setSightings(data.sightings.map((s, i) => ({ ...s, index: i })));
          }
          if (["completed", "cancelled", "error"].includes(data.status)) break;
        } catch {
          break;
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
    };
    poll();

    return () => {
      cancelled = true;
    };
  }, [activeJobId]);

  // Cancel hunt
  const cancelHunt = useCallback(async () => {
    if (!activeJobId) return;
    try {
      await api.post(`/api/search/deep-hunt/${activeJobId}/cancel`);
    } catch {}
  }, [activeJobId]);

  const targetObj = namedObjects?.find((o) => o.id === selectedObject);
  const isRunning = jobData?.status === "running";
  const isDone = jobData && ["completed", "cancelled", "error"].includes(jobData.status);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Crosshair className="w-6 h-6 text-amber-400" />
        <h1 className="text-2xl font-bold text-white">Deep Hunt</h1>
      </div>
      <p className="text-sm text-slate-400">
        Scan all continuous recordings — not just event alerts — to find a specific person, pet, or vehicle.
        Every frame is analysed with YOLO detection + CNN embedding matching.
      </p>

      {/* Configuration */}
      <div className="bg-slate-800 rounded-xl p-5 space-y-4 border border-slate-700">
        <h2 className="text-lg font-semibold text-white">Hunt Configuration</h2>

        {/* Target object selector */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1">Target Object</label>
          <select
            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-amber-500 focus:border-transparent"
            value={selectedObject || ""}
            onChange={(e) => setSelectedObject(e.target.value ? Number(e.target.value) : null)}
            disabled={isRunning}
          >
            <option value="">Select who to hunt for...</option>
            {namedObjects
              ?.filter((o) => o.reference_image_count > 0)
              .map((o) => {
                const Icon = CATEGORY_ICONS[o.category] || Box;
                return (
                  <option key={o.id} value={o.id}>
                    {o.name} ({o.category}, {o.reference_image_count} refs)
                  </option>
                );
              })}
          </select>
        </div>

        {/* Cameras (multi-select) */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1">
            Cameras <span className="text-slate-500">(leave empty for all)</span>
          </label>
          <div className="flex flex-wrap gap-2">
            {cameras
              ?.filter((c) => c.enabled)
              .map((c) => (
                <button
                  key={c.id}
                  onClick={() =>
                    setSelectedCameras((prev) =>
                      prev.includes(c.id) ? prev.filter((id) => id !== c.id) : [...prev, c.id]
                    )
                  }
                  disabled={isRunning}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    selectedCameras.includes(c.id)
                      ? "bg-amber-600 text-white"
                      : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                  } ${isRunning ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <Camera className="w-3.5 h-3.5 inline mr-1" />
                  {c.name}
                </button>
              ))}
          </div>
        </div>

        {/* Time range and interval */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              <Clock className="w-3.5 h-3.5 inline mr-1" />
              Hours back
            </label>
            <select
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white"
              value={hours}
              onChange={(e) => setHours(Number(e.target.value))}
              disabled={isRunning}
            >
              <option value={1}>Last 1 hour</option>
              <option value={3}>Last 3 hours</option>
              <option value={6}>Last 6 hours</option>
              <option value={12}>Last 12 hours</option>
              <option value={24}>Last 24 hours</option>
              <option value={48}>Last 48 hours</option>
              <option value={72}>Last 72 hours</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Frame interval (seconds)
            </label>
            <select
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white"
              value={frameInterval}
              onChange={(e) => setFrameInterval(Number(e.target.value))}
              disabled={isRunning}
            >
              <option value={1}>1s (thorough, slow)</option>
              <option value={2}>2s (balanced)</option>
              <option value={3}>3s (faster)</option>
              <option value={5}>5s (quick scan)</option>
            </select>
          </div>
        </div>

        {/* Start / Cancel buttons */}
        <div className="flex gap-3 pt-2">
          {!isRunning ? (
            <button
              onClick={startHunt}
              disabled={!selectedObject}
              className="flex items-center gap-2 px-5 py-2.5 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
            >
              <Play className="w-4 h-4" />
              Start Hunt
            </button>
          ) : (
            <button
              onClick={cancelHunt}
              className="flex items-center gap-2 px-5 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors"
            >
              <Square className="w-4 h-4" />
              Cancel Hunt
            </button>
          )}
          {targetObj && (
            <span className="text-sm text-slate-400 self-center">
              Hunting for <strong className="text-amber-400">{targetObj.name}</strong>
            </span>
          )}
        </div>
      </div>

      {/* Progress */}
      {jobData && (
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {jobData.status === "running" && <Loader2 className="w-5 h-5 text-amber-400 animate-spin" />}
              {jobData.status === "completed" && <CheckCircle2 className="w-5 h-5 text-green-400" />}
              {jobData.status === "cancelled" && <XCircle className="w-5 h-5 text-slate-400" />}
              {jobData.status === "error" && <AlertTriangle className="w-5 h-5 text-red-400" />}
              <span className="text-white font-medium capitalize">{jobData.status}</span>
            </div>
            <span className="text-sm text-slate-400">
              {jobData.segments_done}/{jobData.segments_total} segments
              {" · "}
              {jobData.frames_scanned || 0} frames
              {" · "}
              {jobData.detections_total || 0} detections ({jobData.detections_relevant || 0} relevant)
              {" · "}
              {sightings.length} sighting{sightings.length !== 1 ? "s" : ""}
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-amber-600 to-amber-400"
              style={{ width: `${Math.round(jobData.progress * 100)}%` }}
            />
          </div>
          <div className="text-right text-xs text-slate-500 mt-1">
            {Math.round(jobData.progress * 100)}%
          </div>
          {jobData.error && (
            <p className="text-sm text-red-400 mt-2">{jobData.error}</p>
          )}
        </div>
      )}

      {/* Sightings results */}
      {sightings.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h2 className="text-lg font-semibold text-white mb-4">
            Sightings ({sightings.length})
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {sightings.map((s, i) => (
              <div
                key={i}
                onClick={() => setPreviewIdx(i)}
                className="bg-slate-700/50 rounded-lg overflow-hidden border border-slate-600 hover:border-amber-500 cursor-pointer transition-colors group"
              >
                <div className="aspect-square bg-slate-900 relative">
                  <img
                    src={thumbUrl(s.thumbnail_url)}
                    alt={`Sighting ${i + 1}`}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1.5">
                    <div className="text-xs text-white font-medium">
                      {(s.confidence * 100).toFixed(0)}% match
                    </div>
                  </div>
                </div>
                <div className="p-2">
                  <div className="text-xs text-slate-300 truncate">{formatCamera(s.camera)}</div>
                  <div className="text-xs text-slate-500">{formatTime(s.timestamp)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No sightings message */}
      {isDone && sightings.length === 0 && (
        <div className="bg-slate-800 rounded-xl p-8 text-center border border-slate-700">
          <Crosshair className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">
            No sightings found for <strong>{jobData?.target}</strong> in the scanned recordings.
          </p>
          <p className="text-sm text-slate-500 mt-1">
            Try a longer time range, more cameras, or a shorter frame interval.
          </p>
        </div>
      )}

      {/* Sighting preview modal */}
      {previewIdx !== null && sightings[previewIdx] && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setPreviewIdx(null)}
        >
          <div
            className="bg-slate-800 rounded-xl max-w-2xl w-full overflow-hidden border border-slate-600"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={thumbUrl(sightings[previewIdx].thumbnail_url)}
              alt="Sighting detail"
              className="w-full max-h-[60vh] object-contain bg-black"
            />
            <div className="p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-white font-medium">
                  Sighting #{previewIdx + 1}
                </span>
                <span className="text-amber-400 font-mono">
                  {(sightings[previewIdx].confidence * 100).toFixed(1)}% match
                </span>
              </div>
              <div className="text-sm text-slate-300">
                <span className="text-slate-500">Camera:</span>{" "}
                {formatCamera(sightings[previewIdx].camera)}
              </div>
              <div className="text-sm text-slate-300">
                <span className="text-slate-500">Time:</span>{" "}
                {formatTime(sightings[previewIdx].timestamp)}
              </div>
              <div className="text-sm text-slate-300">
                <span className="text-slate-500">Detection:</span>{" "}
                {sightings[previewIdx].class_name} ({((sightings[previewIdx].det_confidence || 0) * 100).toFixed(0)}% YOLO)
              </div>
              {/* Navigation */}
              <div className="flex justify-between pt-2">
                <button
                  onClick={() => setPreviewIdx(Math.max(0, previewIdx - 1))}
                  disabled={previewIdx === 0}
                  className="px-3 py-1 text-sm bg-slate-700 text-slate-300 rounded hover:bg-slate-600 disabled:opacity-30"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPreviewIdx(null)}
                  className="px-3 py-1 text-sm bg-slate-700 text-slate-300 rounded hover:bg-slate-600"
                >
                  Close
                </button>
                <button
                  onClick={() => setPreviewIdx(Math.min(sightings.length - 1, previewIdx + 1))}
                  disabled={previewIdx === sightings.length - 1}
                  className="px-3 py-1 text-sm bg-slate-700 text-slate-300 rounded hover:bg-slate-600 disabled:opacity-30"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
