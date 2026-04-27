import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, getToken } from "../api";
import {
  ArrowLeft,
  Camera,
  Clock,
  RefreshCw,
  User,
  Cat,
  Car,
  Box,
  Eye,
  X,
  ShieldAlert,
  Shield,
  Trash2,
  Check,
  CheckCheck,
  AlertTriangle,
  Radio,
  Pencil,
  TriangleAlert,
  Crosshair,
  Play,
  Square,
  Loader2,
  CheckCircle2,
  XCircle,
  History,
  Plus,
} from "lucide-react";

interface ObjectInfo {
  id: number;
  name: string;
  category: string;
  reference_image_count: number;
  attributes: Record<string, any> | null;
  created_at: string;
  last_seen: string | null;
  last_camera: string | null;
}

interface Detection {
  event_id: number;
  camera_id: number;
  camera_name: string;
  object_type: string;
  confidence: number | null;
  timestamp: string;
  thumbnail_url: string | null;
  snapshot_url: string | null;
  narrative: string | null;
}

interface ProfileData {
  object: ObjectInfo;
  total_detections: number;
  cameras: { camera_id: number; camera_name: string; count: number }[];
  recent_detections: Detection[];
  ai_summary: string | null;
  profile_image_url: string | null;
  is_live: boolean;
  live_camera_id: number | null;
  live_camera_name: string | null;
  attributes: Record<string, { value: string; confidence: number; samples: number }> | null;
  last_detection_attrs: Record<string, string | number | null> | null;
  needs_retrain: boolean;
  retrain_reasons: string[];
}

interface AuditDetection {
  event_id: number;
  camera_name: string;
  object_type: string;
  similarity: number;
  thumbnail_url: string;
  timestamp: string;
  flagged: boolean;
}

interface AuditResult {
  detections: AuditDetection[];
  summary: string;
  mean_similarity: number;
  flagged_count: number;
  threshold: number;
}

interface RescanCandidate {
  event_id: number;
  camera_name: string;
  object_type: string;
  similarity: number;
  thumbnail_url: string;
  snapshot_url: string | null;
  timestamp: string;
}

interface RescanResult {
  candidates: RescanCandidate[];
  scanned: number;
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

interface HuntJobSummary {
  job_id: string;
  target: string;
  target_id: number;
  status: string;
  progress: number;
  segments_total: number;
  segments_done: number;
  frames_scanned: number;
  sightings_count: number;
  created_at: number;
}

function formatHuntTime(ts: number): string {
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

const CATEGORY_ICONS: Record<string, typeof User> = {
  person: User,
  pet: Cat,
  vehicle: Car,
  other: Box,
};

export default function ObjectProfile() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewLabel, setPreviewLabel] = useState("");
  const [liveSnapshot, setLiveSnapshot] = useState<string | null>(null);
  const liveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Audit state
  const [auditResult, setAuditResult] = useState<AuditResult | null>(null);
  const [auditSelected, setAuditSelected] = useState<Set<number>>(new Set());

  // Rescan state
  const [rescanResult, setRescanResult] = useState<RescanResult | null>(null);
  const [rescanSelected, setRescanSelected] = useState<Set<number>>(new Set());

  // Manual remove state
  const [removeMode, setRemoveMode] = useState(false);
  const [removeSelected, setRemoveSelected] = useState<Set<number>>(new Set());

  const { data: profile, isLoading } = useQuery({
    queryKey: ["object-profile", id],
    queryFn: () => api.get<ProfileData>(`/api/training/objects/${id}/profile`),
    enabled: !!id,
  });

  const auditMut = useMutation({
    mutationFn: () => api.post<AuditResult>(`/api/training/objects/${id}/audit`, {}),
    onSuccess: (data) => {
      setAuditResult(data);
      setAuditSelected(new Set(data.detections.filter((d) => d.flagged).map((d) => d.event_id)));
    },
  });

  const removeDetMut = useMutation({
    mutationFn: (event_ids: number[]) =>
      api.post(`/api/training/objects/${id}/remove-detections`, { event_ids }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["object-profile", id] });
      setAuditResult(null);
      setAuditSelected(new Set());
      setRemoveMode(false);
      setRemoveSelected(new Set());
    },
  });

  const rescanMut = useMutation({
    mutationFn: () => api.post<RescanResult>(`/api/training/objects/${id}/rescan`, {}),
    onSuccess: (data) => {
      setRescanResult(data);
      setRescanSelected(new Set(data.candidates.filter((c) => c.similarity >= 0.80).map((c) => c.event_id)));
    },
  });

  const confirmMut = useMutation({
    mutationFn: (event_ids: number[]) =>
      api.post(`/api/training/objects/${id}/confirm-matches`, { event_ids }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["object-profile", id] });
      setRescanResult(null);
      setRescanSelected(new Set());
    },
  });

  // Delete state
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const deleteMut = useMutation({
    mutationFn: () => api.delete(`/api/training/objects/${id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["named-objects"] });
      qc.invalidateQueries({ queryKey: ["named-objects-status"] });
      navigate("/training");
    },
  });

  // ── Deep Hunt state ──
  const [huntOpen, setHuntOpen] = useState(false);
  const [huntCameras, setHuntCameras] = useState<number[]>([]);
  const [huntHours, setHuntHours] = useState(24);
  const [huntInterval, setHuntInterval] = useState(2.0);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobData, setJobData] = useState<HuntJob | null>(null);
  const [huntSightings, setHuntSightings] = useState<Sighting[]>([]);
  const [huntPreviewIdx, setHuntPreviewIdx] = useState<number | null>(null);
  const [huntSelected, setHuntSelected] = useState<Set<number>>(new Set());

  // Hunt history
  const { data: huntHistory, refetch: refetchHistory } = useQuery({
    queryKey: ["hunt-history", id],
    queryFn: () => api.get<HuntJobSummary[]>(`/api/search/deep-hunt-jobs?target_id=${id}`),
    enabled: huntOpen,
    refetchInterval: activeJobId ? 5000 : false,
  });

  // Load a past job's results
  const [viewingHistoryJob, setViewingHistoryJob] = useState<string | null>(null);
  const loadHistoryJob = useCallback(async (jobId: string) => {
    try {
      const data = await api.get<HuntJob>(`/api/search/deep-hunt/${jobId}`);
      setViewingHistoryJob(jobId);
      setActiveJobId(null);
      setJobData(data);
      if (data.sightings) {
        setHuntSightings(data.sightings.map((s, i) => ({ ...s, index: i })));
      }
      setHuntSelected(new Set());
    } catch { /* ignore */ }
  }, []);

  // Add selected sightings to training model
  const addToTrainingMut = useMutation({
    mutationFn: async (indices: number[]) => {
      const targetJob = viewingHistoryJob || activeJobId;
      if (!targetJob) throw new Error("No job selected");
      return api.post<{ added: number; total_training_images: number }>(
        `/api/search/deep-hunt/${targetJob}/add-to-training?${indices.map(i => `sighting_indices=${i}`).join("&")}`
      );
    },
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["object-profile", id] });
      setHuntSelected(new Set());
      alert(`Added ${data.added} sighting(s) to model. Total training images: ${data.total_training_images}`);
    },
  });

  const { data: allCameras } = useQuery({
    queryKey: ["cameras-hunt"],
    queryFn: () => api.get<CameraInfo[]>("/api/cameras"),
    enabled: huntOpen,
  });

  const startHunt = useCallback(async () => {
    if (!id) return;
    const params = new URLSearchParams({
      named_object_id: id,
      hours: String(huntHours),
      frame_interval: String(huntInterval),
    });
    if (huntCameras.length > 0) {
      params.set("camera_ids", huntCameras.join(","));
    }
    try {
      const result = await api.post<{ job_id: string }>(`/api/search/deep-hunt?${params}`);
      setActiveJobId(result.job_id);
      setViewingHistoryJob(null);
      setHuntSightings([]);
      setHuntSelected(new Set());
      setJobData(null);
    } catch (e: any) {
      alert(e.message || "Failed to start hunt");
    }
  }, [id, huntCameras, huntHours, huntInterval]);

  // Polling for active hunt job
  useEffect(() => {
    if (!activeJobId) return;
    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        try {
          const data = await api.get<HuntJob>(`/api/search/deep-hunt/${activeJobId}`);
          if (cancelled) break;
          setJobData(data);
          if (data.sightings) {
            setHuntSightings(data.sightings.map((s, i) => ({ ...s, index: i })));
          }
          if (["completed", "cancelled", "error"].includes(data.status)) break;
        } catch { break; }
        await new Promise((r) => setTimeout(r, 2000));
      }
    };
    poll();
    return () => { cancelled = true; };
  }, [activeJobId]);

  const cancelHunt = useCallback(async () => {
    if (!activeJobId) return;
    try { await api.post(`/api/search/deep-hunt/${activeJobId}/cancel`); } catch {}
  }, [activeJobId]);

  const isHuntRunning = jobData?.status === "running";
  const isHuntDone = jobData && ["completed", "cancelled", "error"].includes(jobData.status);

  // Manual attribute editing
  const [editingAttrs, setEditingAttrs] = useState(false);
  const [editGender, setEditGender] = useState<string>("");
  const [editAgeGroup, setEditAgeGroup] = useState<string>("");

  const updateAttrsMut = useMutation({
    mutationFn: (data: { gender?: string; age_group?: string; breed?: string; color?: string; markings?: string; vehicle_type?: string; make?: string }) =>
      api.patch(`/api/training/objects/${id}`, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["object-profile", id] });
      setEditingAttrs(false);
      setEditingPetAttrs(false);
      setEditingVehicleAttrs(false);
    },
  });

  const startEditAttrs = () => {
    setEditGender(attributes?.gender?.value || "");
    setEditAgeGroup(attributes?.age_group?.value || "");
    setEditingAttrs(true);
  };

  const saveAttrs = () => {
    const payload: { gender?: string; age_group?: string } = {};
    if (editGender) payload.gender = editGender;
    if (editAgeGroup) payload.age_group = editAgeGroup;
    if (Object.keys(payload).length > 0) updateAttrsMut.mutate(payload);
    else setEditingAttrs(false);
  };

  // Pet attribute editing
  const [editingPetAttrs, setEditingPetAttrs] = useState(false);
  const [editBreed, setEditBreed] = useState("");
  const [editColor, setEditColor] = useState("");
  const [editMarkings, setEditMarkings] = useState("");

  // Vehicle attribute editing
  const [editingVehicleAttrs, setEditingVehicleAttrs] = useState(false);
  const [editVehicleType, setEditVehicleType] = useState("");
  const [editVehicleColor, setEditVehicleColor] = useState("");
  const [editMake, setEditMake] = useState("");

  const PET_BREEDS = [
    "Persian", "British Shorthair", "Maine Coon", "Siamese", "Ragdoll",
    "Bengal", "Domestic Shorthair", "Domestic Longhair", "Mixed / Unknown",
  ];
  const PET_COLORS = [
    "White", "Black", "Orange / Ginger", "Gray", "Brown", "Cream", "Tortoiseshell", "Calico",
  ];
  const PET_MARKINGS = [
    "Solid", "Tabby Stripes", "Patches", "Bicolor", "Pointed", "Tuxedo", "Van",
  ];
  const VEHICLE_TYPES_LIST = [
    "Car", "SUV", "Truck", "Van", "Motorcycle", "Bicycle", "Bus",
  ];
  const VEHICLE_COLORS_LIST = [
    "White", "Black", "Silver", "Gray", "Red", "Blue", "Green", "Brown", "Beige",
  ];
  const VEHICLE_MAKES_LIST = [
    "Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Volkswagen", "Tesla",
    "Hyundai", "Kia", "Nissan", "Mazda", "Lexus", "Porsche", "Other",
  ];

  const startEditPetAttrs = () => {
    const rawAttrs = obj?.attributes || {};
    setEditBreed(rawAttrs.breed || "");
    setEditColor(rawAttrs.color || "");
    setEditMarkings(rawAttrs.markings || "");
    setEditingPetAttrs(true);
  };

  const savePetAttrs = () => {
    const payload: { breed?: string; color?: string; markings?: string } = {};
    if (editBreed) payload.breed = editBreed;
    if (editColor) payload.color = editColor;
    if (editMarkings) payload.markings = editMarkings;
    if (Object.keys(payload).length > 0) updateAttrsMut.mutate(payload);
    else setEditingPetAttrs(false);
  };

  const startEditVehicleAttrs = () => {
    const rawAttrs = obj?.attributes || {};
    setEditVehicleType(rawAttrs.vehicle_type || "");
    setEditVehicleColor(rawAttrs.color || "");
    setEditMake(rawAttrs.make || "");
    setEditingVehicleAttrs(true);
  };

  const saveVehicleAttrs = () => {
    const payload: { vehicle_type?: string; color?: string; make?: string } = {};
    if (editVehicleType) payload.vehicle_type = editVehicleType;
    if (editVehicleColor) payload.color = editVehicleColor;
    if (editMake) payload.make = editMake;
    if (Object.keys(payload).length > 0) updateAttrsMut.mutate(payload);
    else setEditingVehicleAttrs(false);
  };

  const token = getToken();
  const imgUrl = (url: string) => `${url}?token=${encodeURIComponent(token || "")}`;

  // Live snapshot refresh
  useEffect(() => {
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
    if (profile?.is_live && profile.live_camera_id) {
      const camId = profile.live_camera_id;
      const fetchSnap = async () => {
        try {
          const res = await fetch(`/go2rtc/api/frame.jpeg?src=camera_${camId}`);
          if (res.ok) {
            const blob = await res.blob();
            setLiveSnapshot((prev) => {
              if (prev) URL.revokeObjectURL(prev);
              return URL.createObjectURL(blob);
            });
          }
        } catch { /* ignore */ }
      };
      fetchSnap();
      liveIntervalRef.current = setInterval(fetchSnap, 3000);
    } else {
      setLiveSnapshot((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
    }
    return () => {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
    };
  }, [profile?.is_live, profile?.live_camera_id]);

  const showPreview = (url: string, label: string) => {
    setPreviewUrl(url);
    setPreviewLabel(label);
  };

  if (isLoading) {
    return <div className="p-4 text-center py-12 text-slate-400">Loading profile...</div>;
  }

  if (!profile) {
    return (
      <div className="p-4 space-y-4">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1 text-sm text-slate-400 hover:text-white">
          <ArrowLeft size={16} /> Back
        </button>
        <div className="card text-center py-12 text-slate-400">Object not found</div>
      </div>
    );
  }

  const { object: obj, total_detections, cameras, recent_detections, profile_image_url } = profile;
  const Icon = CATEGORY_ICONS[obj.category] || Box;
  const isLive = profile.is_live;
  const liveCameraName = profile.live_camera_name;
  const attributes = profile.attributes;
  const lastAttrs = profile.last_detection_attrs;

  const ATTR_LABELS: Record<string, string> = {
    gender: "Gender",
    age_group: "Age",
    build: "Build",
    height_estimate: "Height",
  };

  const ATTR_ICONS: Record<string, string> = {
    gender: "👤",
    age_group: "🎂",
    build: "💪",
    height_estimate: "📏",
  };

  const TRANSIENT_LABELS: Record<string, string> = {
    upper_color: "Top",
    lower_color: "Bottom",
    posture: "Posture",
    height_ratio: "Height Ratio",
  };

  return (
    <div className="p-4 space-y-4 pb-24 max-w-2xl mx-auto">
      {/* Header */}
      <button onClick={() => navigate(-1)} className="flex items-center gap-1 text-sm text-slate-400 hover:text-white">
        <ArrowLeft size={16} /> Back
      </button>

      <div className="card flex items-center gap-4">
        {profile_image_url ? (
          <div className="relative shrink-0">
            <img src={imgUrl(profile_image_url)} alt={obj.name}
              className="w-14 h-14 rounded-full object-cover bg-slate-800" />
            {isLive && (
              <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-red-500" />
              </span>
            )}
          </div>
        ) : (
          <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center shrink-0 relative">
            <Icon size={28} className="text-blue-400" />
            {isLive && (
              <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-red-500" />
              </span>
            )}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <h2 className="text-lg font-bold">{obj.name}</h2>
          <p className="text-sm text-slate-400 capitalize">{obj.category}</p>
          <p className="text-xs text-slate-500 mt-0.5">
            {obj.reference_image_count} training images · Created{" "}
            {new Date(obj.created_at).toLocaleDateString()}
          </p>
        </div>
      </div>

      {/* Retrain Alert */}
      {profile.needs_retrain && (
        <button
          onClick={() => navigate(`/profiles/${id}/retrain`)}
          className="w-full bg-amber-900/30 border border-amber-600/40 rounded-xl p-3 text-left hover:border-amber-500/60 transition-colors"
        >
          <div className="flex items-start gap-2.5">
            <AlertTriangle size={16} className="text-amber-400 shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-amber-300">Retraining Recommended</p>
              <ul className="mt-1 space-y-0.5">
                {profile.retrain_reasons.map((r, i) => (
                  <li key={i} className="text-xs text-amber-400/80">· {r}</li>
                ))}
              </ul>
              <p className="text-[10px] text-amber-500/70 mt-1.5">Tap to start Deep Retrain</p>
            </div>
          </div>
        </button>
      )}

      {/* Live Feed */}
      {isLive && liveSnapshot && (
        <div className="card space-y-2">
          <h3 className="font-semibold text-sm flex items-center gap-2">
            <Radio size={14} className="text-red-400" />
            <span className="text-red-400">LIVE</span>
            <span className="text-slate-400 font-normal">on {liveCameraName}</span>
          </h3>
          <img
            src={liveSnapshot}
            alt={`Live: ${obj.name}`}
            className="w-full rounded-lg"
          />
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        <div className="card text-center">
          <p className="text-xl font-bold text-blue-400">{total_detections}</p>
          <p className="text-[10px] text-slate-500">Detections</p>
        </div>
        <div className="card text-center">
          <p className="text-xl font-bold text-blue-400">{cameras.length}</p>
          <p className="text-[10px] text-slate-500">Cameras</p>
        </div>
        <div className="card text-center">
          <p className="text-sm font-medium text-blue-400 truncate">
            {obj.last_seen
              ? new Date(obj.last_seen).toLocaleString(undefined, {
                  day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
                })
              : "—"}
          </p>
          <p className="text-[10px] text-slate-500">Last Seen</p>
        </div>
      </div>

      {/* Attributes */}
      {obj.category === "person" && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm flex items-center gap-1.5">
              <User size={14} className="text-blue-400" /> Physical Profile
            </h3>
            {!editingAttrs ? (
              <button onClick={startEditAttrs} className="text-slate-500 hover:text-blue-400 transition-colors">
                <Pencil size={14} />
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <button onClick={() => setEditingAttrs(false)} className="text-slate-500 hover:text-white text-xs">Cancel</button>
                <button onClick={saveAttrs} disabled={updateAttrsMut.isPending}
                  className="text-blue-400 hover:text-blue-300 text-xs font-medium">
                  {updateAttrsMut.isPending ? "Saving..." : "Save"}
                </button>
              </div>
            )}
          </div>

          {/* Edit mode — gender and age pickers */}
          {editingAttrs && (
            <div className="space-y-3 bg-slate-800/50 rounded-lg p-3">
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Gender</label>
                <div className="flex gap-2">
                  {["male", "female"].map((g) => (
                    <button key={g} onClick={() => setEditGender(editGender === g ? "" : g)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                        editGender === g
                          ? "bg-blue-600 text-white"
                          : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>
                      {g === "male" ? "👨 Male" : "👩 Female"}
                    </button>
                  ))}
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Age Group</label>
                <div className="flex flex-wrap gap-2">
                  {[
                    { value: "child", label: "👶 Child" },
                    { value: "young_adult", label: "🧑 Young Adult" },
                    { value: "adult", label: "🧑‍💼 Adult" },
                    { value: "middle_aged", label: "👨‍🦳 Middle Aged" },
                    { value: "senior", label: "👴 Senior" },
                  ].map((ag) => (
                    <button key={ag.value} onClick={() => setEditAgeGroup(editAgeGroup === ag.value ? "" : ag.value)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                        editAgeGroup === ag.value
                          ? "bg-blue-600 text-white"
                          : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>
                      {ag.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Stable learned attributes */}
          {!editingAttrs && attributes && Object.keys(attributes).length > 0 ? (
            <div className="space-y-2">
              {Object.entries(attributes).map(([key, attr]) => {
                const label = ATTR_LABELS[key] || key;
                const icon = ATTR_ICONS[key] || "";
                const conf = Math.round(attr.confidence * 100);
                const barColor = conf >= 80 ? "bg-emerald-500" : conf >= 50 ? "bg-amber-500" : "bg-slate-600";
                return (
                  <div key={key} className="flex items-center gap-3">
                    <span className="text-sm w-5 text-center">{icon}</span>
                    <span className="text-xs text-slate-400 w-14">{label}</span>
                    <span className="text-sm font-medium capitalize w-24">{attr.value}</span>
                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${barColor}`} style={{ width: `${conf}%` }} />
                    </div>
                    <span className={`text-[10px] w-8 text-right ${conf >= 80 ? "text-emerald-400" : conf >= 50 ? "text-amber-400" : "text-slate-500"}`}>
                      {conf}%
                    </span>
                    <span className="text-[10px] text-slate-600 w-10 text-right">
                      ×{attr.samples}
                    </span>
                    {attr.manual && (
                      <span className="text-[9px] bg-blue-900/40 text-blue-400 px-1.5 py-0.5 rounded font-medium">
                        Manual
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-slate-500 italic">
              No attributes learned yet — will build automatically from detections
            </p>
          )}

          {/* Transient: last-seen clothing + posture */}
          {lastAttrs && (lastAttrs.upper_color || lastAttrs.lower_color || lastAttrs.posture) && (
            <div className="border-t border-slate-800 pt-2 mt-1">
              <p className="text-[10px] text-slate-500 mb-1.5">Last seen wearing</p>
              <div className="flex flex-wrap gap-2">
                {lastAttrs.upper_color && lastAttrs.upper_color !== "unknown" && (
                  <span className="bg-slate-800 rounded-md px-2 py-1 text-xs flex items-center gap-1.5">
                    <span className="text-slate-500">Top</span>
                    <span className="font-medium capitalize">{lastAttrs.upper_color}</span>
                  </span>
                )}
                {lastAttrs.lower_color && lastAttrs.lower_color !== "unknown" && (
                  <span className="bg-slate-800 rounded-md px-2 py-1 text-xs flex items-center gap-1.5">
                    <span className="text-slate-500">Bottom</span>
                    <span className="font-medium capitalize">{lastAttrs.lower_color}</span>
                  </span>
                )}
                {lastAttrs.posture && (
                  <span className="bg-slate-800 rounded-md px-2 py-1 text-xs flex items-center gap-1.5">
                    <span className="text-slate-500">Posture</span>
                    <span className="font-medium capitalize">{String(lastAttrs.posture)}</span>
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Pet Attributes */}
      {obj.category === "pet" && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm flex items-center gap-1.5">
              <Cat size={14} className="text-blue-400" /> Pet Profile
            </h3>
            {!editingPetAttrs ? (
              <button onClick={startEditPetAttrs} className="text-slate-500 hover:text-blue-400 transition-colors">
                <Pencil size={14} />
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <button onClick={() => setEditingPetAttrs(false)} className="text-slate-500 hover:text-white text-xs">Cancel</button>
                <button onClick={savePetAttrs} disabled={updateAttrsMut.isPending}
                  className="text-blue-400 hover:text-blue-300 text-xs font-medium">
                  {updateAttrsMut.isPending ? "Saving..." : "Save"}
                </button>
              </div>
            )}
          </div>

          {editingPetAttrs ? (
            <div className="space-y-3 bg-slate-800/50 rounded-lg p-3">
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Breed</label>
                <div className="flex flex-wrap gap-1.5">
                  {PET_BREEDS.map((b) => (
                    <button key={b} onClick={() => setEditBreed(editBreed === b ? "" : b)}
                      className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                        editBreed === b ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>{b}</button>
                  ))}
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Color</label>
                <div className="flex flex-wrap gap-1.5">
                  {PET_COLORS.map((c) => (
                    <button key={c} onClick={() => setEditColor(editColor === c ? "" : c)}
                      className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                        editColor === c ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>{c}</button>
                  ))}
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Markings</label>
                <div className="flex flex-wrap gap-1.5">
                  {PET_MARKINGS.map((m) => (
                    <button key={m} onClick={() => setEditMarkings(editMarkings === m ? "" : m)}
                      className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                        editMarkings === m ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>{m}</button>
                  ))}
                </div>
              </div>
            </div>
          ) : (() => {
            const rawAttrs = obj.attributes || {};
            const hasAny = rawAttrs.breed || rawAttrs.color || rawAttrs.markings;
            return hasAny ? (
              <div className="space-y-2">
                {rawAttrs.breed && (
                  <div className="flex items-center gap-3">
                    <span className="text-sm w-5 text-center">🐱</span>
                    <span className="text-xs text-slate-400 w-16">Breed</span>
                    <span className="text-sm font-medium">{rawAttrs.breed}</span>
                    {rawAttrs._breed_manual && (
                      <span className="text-[9px] bg-blue-900/40 text-blue-400 px-1.5 py-0.5 rounded font-medium ml-auto">Manual</span>
                    )}
                  </div>
                )}
                {rawAttrs.color && (
                  <div className="flex items-center gap-3">
                    <span className="text-sm w-5 text-center">🎨</span>
                    <span className="text-xs text-slate-400 w-16">Color</span>
                    <span className="text-sm font-medium">{rawAttrs.color}</span>
                  </div>
                )}
                {rawAttrs.markings && (
                  <div className="flex items-center gap-3">
                    <span className="text-sm w-5 text-center">✨</span>
                    <span className="text-xs text-slate-400 w-16">Markings</span>
                    <span className="text-sm font-medium">{rawAttrs.markings}</span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">
                No details set yet — tap the pencil to add breed, color & markings
              </p>
            );
          })()}
        </div>
      )}

      {/* Vehicle Attributes */}
      {obj.category === "vehicle" && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm flex items-center gap-1.5">
              <Car size={14} className="text-blue-400" /> Vehicle Profile
            </h3>
            {!editingVehicleAttrs ? (
              <button onClick={startEditVehicleAttrs} className="text-slate-500 hover:text-blue-400 transition-colors">
                <Pencil size={14} />
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <button onClick={() => setEditingVehicleAttrs(false)} className="text-slate-500 hover:text-white text-xs">Cancel</button>
                <button onClick={saveVehicleAttrs} disabled={updateAttrsMut.isPending}
                  className="text-blue-400 hover:text-blue-300 text-xs font-medium">
                  {updateAttrsMut.isPending ? "Saving..." : "Save"}
                </button>
              </div>
            )}
          </div>

          {editingVehicleAttrs ? (
            <div className="space-y-3 bg-slate-800/50 rounded-lg p-3">
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Type</label>
                <div className="flex flex-wrap gap-1.5">
                  {VEHICLE_TYPES_LIST.map((vt) => (
                    <button key={vt} onClick={() => setEditVehicleType(editVehicleType === vt ? "" : vt)}
                      className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                        editVehicleType === vt ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>{vt}</button>
                  ))}
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Color</label>
                <div className="flex flex-wrap gap-1.5">
                  {VEHICLE_COLORS_LIST.map((vc) => (
                    <button key={vc} onClick={() => setEditVehicleColor(editVehicleColor === vc ? "" : vc)}
                      className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                        editVehicleColor === vc ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>{vc}</button>
                  ))}
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Make</label>
                <div className="flex flex-wrap gap-1.5">
                  {VEHICLE_MAKES_LIST.map((vm) => (
                    <button key={vm} onClick={() => setEditMake(editMake === vm ? "" : vm)}
                      className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                        editMake === vm ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      }`}>{vm}</button>
                  ))}
                </div>
              </div>
            </div>
          ) : (() => {
            const rawAttrs = obj.attributes || {};
            const hasAny = rawAttrs.vehicle_type || rawAttrs.color || rawAttrs.make;
            return hasAny ? (
              <div className="space-y-2">
                {rawAttrs.vehicle_type && (
                  <div className="flex items-center gap-3">
                    <span className="text-sm w-5 text-center">🚗</span>
                    <span className="text-xs text-slate-400 w-16">Type</span>
                    <span className="text-sm font-medium">{rawAttrs.vehicle_type}</span>
                    {rawAttrs._vehicle_type_manual && (
                      <span className="text-[9px] bg-blue-900/40 text-blue-400 px-1.5 py-0.5 rounded font-medium ml-auto">Manual</span>
                    )}
                  </div>
                )}
                {rawAttrs.color && (
                  <div className="flex items-center gap-3">
                    <span className="text-sm w-5 text-center">🎨</span>
                    <span className="text-xs text-slate-400 w-16">Color</span>
                    <span className="text-sm font-medium">{rawAttrs.color}</span>
                  </div>
                )}
                {rawAttrs.make && (
                  <div className="flex items-center gap-3">
                    <span className="text-sm w-5 text-center">🏭</span>
                    <span className="text-xs text-slate-400 w-16">Make</span>
                    <span className="text-sm font-medium">{rawAttrs.make}</span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">
                No details set yet — tap the pencil to add type, color & make
              </p>
            );
          })()}
        </div>
      )}

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => { setRescanResult(null); auditMut.mutate(); }}
          disabled={auditMut.isPending}
          className="card flex items-center justify-center gap-2 py-3 hover:border-amber-500/50 transition-colors text-sm font-medium"
        >
          <ShieldAlert size={16} className={auditMut.isPending ? "animate-pulse text-amber-400" : "text-amber-400"} />
          {auditMut.isPending ? "Checking..." : "Find Mismatches"}
        </button>
        <button
          onClick={() => { setAuditResult(null); rescanMut.mutate(); }}
          disabled={rescanMut.isPending}
          className="card flex items-center justify-center gap-2 py-3 hover:border-blue-500/50 transition-colors text-sm font-medium"
        >
          <RefreshCw size={16} className={rescanMut.isPending ? "animate-spin text-blue-400" : "text-blue-400"} />
          {rescanMut.isPending ? "Scanning..." : "Rescan 12 Hours"}
        </button>
      </div>

      {/* Deep Retrain */}
      <button
        onClick={() => navigate(`/profiles/${id}/retrain`)}
        className="card w-full flex items-center justify-center gap-2 py-3 hover:border-green-500/50 transition-colors text-sm font-medium"
      >
        <Shield size={16} className="text-green-400" />
        Deep Retrain — Full Model Rebuild
      </button>

      {/* Delete Profile */}
      <button
        onClick={() => setShowDeleteConfirm(true)}
        className="card w-full flex items-center justify-center gap-2 py-3 hover:border-red-500/50 transition-colors text-sm font-medium text-red-400"
      >
        <Trash2 size={16} />
        Delete Profile
      </button>

      {/* ── Deep Hunt ── */}
      <div className="card space-y-3">
        <button
          onClick={() => setHuntOpen(!huntOpen)}
          className="w-full flex items-center justify-between"
        >
          <h3 className="font-semibold text-sm flex items-center gap-1.5">
            <Crosshair size={14} className="text-amber-400" /> Deep Hunt
          </h3>
          <span className="text-xs text-slate-500">{huntOpen ? "Hide" : "Scan recordings"}</span>
        </button>

        {huntOpen && (
          <div className="space-y-3">
            <p className="text-xs text-slate-500">
              Scan continuous recordings for <strong className="text-amber-400">{obj.name}</strong> using YOLO + CNN matching.
            </p>

            {/* Camera picker */}
            {allCameras && (
              <div>
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">
                  Cameras <span className="text-slate-600">(empty = all)</span>
                </label>
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {allCameras.filter((c) => c.enabled).map((c) => (
                    <button
                      key={c.id}
                      onClick={() =>
                        setHuntCameras((prev) =>
                          prev.includes(c.id) ? prev.filter((x) => x !== c.id) : [...prev, c.id]
                        )
                      }
                      disabled={isHuntRunning}
                      className={`px-2 py-1 rounded-md text-xs font-medium transition-colors ${
                        huntCameras.includes(c.id)
                          ? "bg-amber-600 text-white"
                          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                      } ${isHuntRunning ? "opacity-50" : ""}`}
                    >
                      <Camera className="w-3 h-3 inline mr-0.5" />
                      {c.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Hours + interval */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Hours back</label>
                <select
                  className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white"
                  value={huntHours}
                  onChange={(e) => setHuntHours(Number(e.target.value))}
                  disabled={isHuntRunning}
                >
                  {[1, 3, 6, 12, 24, 48, 72].map((h) => (
                    <option key={h} value={h}>Last {h}h</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Frame interval</label>
                <select
                  className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white"
                  value={huntInterval}
                  onChange={(e) => setHuntInterval(Number(e.target.value))}
                  disabled={isHuntRunning}
                >
                  <option value={1}>1s (thorough)</option>
                  <option value={2}>2s (balanced)</option>
                  <option value={3}>3s (faster)</option>
                  <option value={5}>5s (quick)</option>
                </select>
              </div>
            </div>

            {/* Start / Cancel */}
            <div className="flex gap-2">
              {!isHuntRunning ? (
                <button
                  onClick={startHunt}
                  className="flex-1 flex items-center justify-center gap-2 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  <Play size={14} /> Start Hunt
                </button>
              ) : (
                <button
                  onClick={cancelHunt}
                  className="flex-1 flex items-center justify-center gap-2 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  <Square size={14} /> Cancel
                </button>
              )}
            </div>

            {/* Progress */}
            {jobData && (
              <div className="space-y-2 bg-slate-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5">
                    {jobData.status === "running" && <Loader2 size={14} className="text-amber-400 animate-spin" />}
                    {jobData.status === "completed" && <CheckCircle2 size={14} className="text-green-400" />}
                    {jobData.status === "cancelled" && <XCircle size={14} className="text-slate-400" />}
                    {jobData.status === "error" && <AlertTriangle size={14} className="text-red-400" />}
                    <span className="text-white font-medium capitalize">{jobData.status}</span>
                  </div>
                  <span className="text-slate-500">
                    {jobData.segments_done}/{jobData.segments_total} segs · {jobData.frames_scanned || 0} frames · {huntSightings.length} found
                  </span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-amber-600 to-amber-400"
                    style={{ width: `${Math.round(jobData.progress * 100)}%` }}
                  />
                </div>
                {jobData.error && <p className="text-xs text-red-400">{jobData.error}</p>}
              </div>
            )}

            {/* Sightings grid */}
            {huntSightings.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-slate-400">
                    Sightings ({huntSightings.length})
                  </h4>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        if (huntSelected.size === huntSightings.length) {
                          setHuntSelected(new Set());
                        } else {
                          setHuntSelected(new Set(huntSightings.map((_, i) => i)));
                        }
                      }}
                      className="text-[10px] text-amber-400 hover:text-amber-300"
                    >
                      {huntSelected.size === huntSightings.length ? "Deselect All" : "Select All"}
                    </button>
                    {huntSelected.size > 0 && (
                      <span className="text-[10px] text-slate-500">{huntSelected.size} selected</span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {huntSightings.map((s, i) => (
                    <div
                      key={i}
                      className={`bg-slate-800 rounded-lg overflow-hidden border transition-colors ${
                        huntSelected.has(i) ? "border-amber-500" : "border-slate-700 hover:border-amber-500/50"
                      }`}
                    >
                      <div className="aspect-[3/4] bg-slate-900 relative cursor-pointer" onClick={() => setHuntPreviewIdx(i)}>
                        <img src={imgUrl(s.thumbnail_url)} alt={`Sighting ${i + 1}`} className="w-full h-full object-contain" loading="lazy" />
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-1.5 py-1">
                          <div className="text-[10px] text-white font-medium">{(s.confidence * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                      <div className="p-1.5 flex items-center gap-1.5">
                        <input
                          type="checkbox"
                          checked={huntSelected.has(i)}
                          onChange={() => {
                            const next = new Set(huntSelected);
                            next.has(i) ? next.delete(i) : next.add(i);
                            setHuntSelected(next);
                          }}
                          className="shrink-0 accent-amber-500"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-[10px] text-slate-300 truncate">{formatCamera(s.camera)}</div>
                          <div className="text-[10px] text-slate-500">{formatHuntTime(s.timestamp)}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Add to Model button */}
                {huntSelected.size > 0 && (
                  <button
                    onClick={() => addToTrainingMut.mutate(Array.from(huntSelected))}
                    disabled={addToTrainingMut.isPending}
                    className="w-full flex items-center justify-center gap-2 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                  >
                    <Plus size={14} />
                    {addToTrainingMut.isPending
                      ? "Adding..."
                      : `Add ${huntSelected.size} Sighting${huntSelected.size !== 1 ? "s" : ""} to Model`}
                  </button>
                )}
              </div>
            )}

            {/* No results */}
            {isHuntDone && huntSightings.length === 0 && (
              <div className="text-center py-4">
                <Crosshair size={24} className="text-slate-600 mx-auto mb-2" />
                <p className="text-xs text-slate-500">No sightings found. Try a longer time range or shorter interval.</p>
              </div>
            )}

            {/* Scan History */}
            {huntHistory && huntHistory.length > 0 && (
              <div className="space-y-2 border-t border-slate-800 pt-3">
                <h4 className="text-xs font-semibold text-slate-400 flex items-center gap-1.5">
                  <History size={12} /> Scan History
                </h4>
                <div className="space-y-1 max-h-[30vh] overflow-y-auto">
                  {huntHistory.map((h) => {
                    const isActive = h.job_id === activeJobId || h.job_id === viewingHistoryJob;
                    const dt = new Date(h.created_at * 1000);
                    return (
                      <button
                        key={h.job_id}
                        onClick={() => {
                          if (h.status === "running" || h.status === "pending") {
                            setActiveJobId(h.job_id);
                            setViewingHistoryJob(null);
                          } else {
                            loadHistoryJob(h.job_id);
                          }
                        }}
                        className={`w-full flex items-center gap-2 p-2 rounded-lg text-left transition-colors ${
                          isActive ? "bg-amber-900/20 border border-amber-700/40" : "bg-slate-800/50 hover:bg-slate-800"
                        }`}
                      >
                        <div className="shrink-0">
                          {h.status === "completed" && <CheckCircle2 size={14} className="text-green-400" />}
                          {h.status === "running" && <Loader2 size={14} className="text-amber-400 animate-spin" />}
                          {h.status === "cancelled" && <XCircle size={14} className="text-slate-400" />}
                          {h.status === "error" && <AlertTriangle size={14} className="text-red-400" />}
                          {h.status === "pending" && <Loader2 size={14} className="text-slate-400" />}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs text-slate-300">
                            {dt.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                          </div>
                          <div className="text-[10px] text-slate-500">
                            {h.frames_scanned} frames · {h.segments_done}/{h.segments_total} segs
                          </div>
                        </div>
                        <div className="shrink-0 text-right">
                          <span className={`text-xs font-medium ${h.sightings_count > 0 ? "text-amber-400" : "text-slate-500"}`}>
                            {h.sightings_count} found
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Hunt Sighting Preview Modal */}
      {huntPreviewIdx !== null && huntSightings[huntPreviewIdx] && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4"
          onClick={() => setHuntPreviewIdx(null)}
        >
          <div className="max-w-lg w-full bg-slate-900 rounded-xl overflow-hidden border border-slate-700" onClick={(e) => e.stopPropagation()}>
            <img src={imgUrl(huntSightings[huntPreviewIdx].thumbnail_url)} alt="Sighting" className="w-full max-h-[50vh] object-contain bg-black" />
            <div className="p-3 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-sm text-white font-medium">Sighting #{huntPreviewIdx + 1}</span>
                <span className="text-amber-400 font-mono text-sm">{(huntSightings[huntPreviewIdx].confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="text-xs text-slate-400">
                {formatCamera(huntSightings[huntPreviewIdx].camera)} · {formatHuntTime(huntSightings[huntPreviewIdx].timestamp)}
              </div>
              <div className="text-xs text-slate-500">
                {huntSightings[huntPreviewIdx].class_name} ({((huntSightings[huntPreviewIdx].det_confidence || 0) * 100).toFixed(0)}% YOLO)
              </div>
              <div className="flex justify-between pt-1">
                <button onClick={() => setHuntPreviewIdx(Math.max(0, huntPreviewIdx - 1))} disabled={huntPreviewIdx === 0}
                  className="px-3 py-1 text-xs bg-slate-800 text-slate-300 rounded hover:bg-slate-700 disabled:opacity-30">Prev</button>
                <button
                  onClick={() => {
                    const next = new Set(huntSelected);
                    if (next.has(huntPreviewIdx)) next.delete(huntPreviewIdx);
                    else next.add(huntPreviewIdx);
                    setHuntSelected(next);
                  }}
                  className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
                    huntSelected.has(huntPreviewIdx)
                      ? "bg-amber-600 text-white"
                      : "bg-slate-800 text-emerald-400 hover:bg-slate-700"
                  }`}
                >
                  {huntSelected.has(huntPreviewIdx) ? "Selected ✓" : "Select"}
                </button>
                <button onClick={() => setHuntPreviewIdx(null)} className="px-3 py-1 text-xs bg-slate-800 text-slate-300 rounded hover:bg-slate-700">Close</button>
                <button onClick={() => setHuntPreviewIdx(Math.min(huntSightings.length - 1, huntPreviewIdx + 1))} disabled={huntPreviewIdx === huntSightings.length - 1}
                  className="px-3 py-1 text-xs bg-slate-800 text-slate-300 rounded hover:bg-slate-700 disabled:opacity-30">Next</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/70 flex items-end sm:items-center justify-center z-50 p-0 sm:p-4">
          <div className="bg-slate-900 rounded-t-2xl sm:rounded-2xl w-full max-w-sm p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-red-400 flex items-center gap-2">
                <TriangleAlert size={18} /> Delete Profile
              </h3>
              <button onClick={() => setShowDeleteConfirm(false)} className="text-slate-400 hover:text-white">
                <X size={20} />
              </button>
            </div>
            <div className="flex items-center gap-3">
              {profile_image_url ? (
                <img src={imgUrl(profile_image_url)} alt={obj.name}
                  className="w-12 h-12 rounded-full object-cover bg-slate-800" />
              ) : (
                <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center">
                  <Icon size={24} className="text-slate-600" />
                </div>
              )}
              <div>
                <p className="font-semibold">{obj.name}</p>
                <p className="text-xs text-slate-400 capitalize">{obj.category}</p>
              </div>
            </div>
            <p className="text-sm text-slate-300">
              This will permanently remove <span className="font-semibold text-white">{obj.name}</span> and
              unlink all <span className="font-semibold text-white">{total_detections}</span> detection{total_detections !== 1 ? "s" : ""}.
              This cannot be undone.
            </p>
            <div className="flex gap-2">
              <button onClick={() => setShowDeleteConfirm(false)}
                className="btn-secondary flex-1 text-sm py-2.5">
                Cancel
              </button>
              <button
                onClick={() => deleteMut.mutate()}
                disabled={deleteMut.isPending}
                className="flex-1 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm py-2.5 font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-1.5"
              >
                <Trash2 size={14} />
                {deleteMut.isPending ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Audit Results ── */}
      {auditResult && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm flex items-center gap-1.5">
              <ShieldAlert size={14} className="text-amber-400" /> Audit Results
            </h3>
            <button onClick={() => setAuditResult(null)} className="text-slate-500 hover:text-white">
              <X size={16} />
            </button>
          </div>
          <p className="text-sm text-slate-300">{auditResult.summary}</p>
          <div className="text-xs text-slate-500">
            Avg similarity: {(auditResult.mean_similarity * 100).toFixed(1)}% · Threshold: {(auditResult.threshold * 100).toFixed(1)}%
          </div>

          {auditResult.detections.length > 0 && (
            <>
              <div className="space-y-1 max-h-[40vh] overflow-y-auto">
                {auditResult.detections.map((d) => (
                  <div
                    key={d.event_id}
                    className={`flex items-center gap-2 p-2 rounded-lg text-sm ${
                      d.flagged ? "bg-red-900/20 border border-red-800/40" : "bg-slate-800/50"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={auditSelected.has(d.event_id)}
                      onChange={() => {
                        const next = new Set(auditSelected);
                        next.has(d.event_id) ? next.delete(d.event_id) : next.add(d.event_id);
                        setAuditSelected(next);
                      }}
                      className="shrink-0 accent-red-500"
                    />
                    {d.thumbnail_url && (
                      <button
                        onClick={() => showPreview(d.thumbnail_url, `${d.camera_name} · ${new Date(d.timestamp).toLocaleString()}`)}
                        className="w-10 h-8 rounded overflow-hidden bg-slate-800 shrink-0"
                      >
                        <img src={imgUrl(d.thumbnail_url)} alt="" className="w-full h-full object-contain" loading="lazy" />
                      </button>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs truncate">{d.camera_name}</p>
                      <p className="text-[10px] text-slate-500">{new Date(d.timestamp).toLocaleString()}</p>
                    </div>
                    <div className="shrink-0 text-right">
                      <span className={`text-xs font-mono ${d.flagged ? "text-red-400" : "text-emerald-400"}`}>
                        {(d.similarity * 100).toFixed(1)}%
                      </span>
                      {d.flagged && <AlertTriangle size={12} className="text-red-400 ml-1 inline" />}
                    </div>
                  </div>
                ))}
              </div>

              {auditSelected.size > 0 && (
                <button
                  onClick={() => removeDetMut.mutate(Array.from(auditSelected))}
                  disabled={removeDetMut.isPending}
                  className="btn-danger w-full text-sm flex items-center justify-center gap-2"
                >
                  <Trash2 size={14} />
                  {removeDetMut.isPending
                    ? "Removing..."
                    : `Remove ${auditSelected.size} Selected & Recompute Model`}
                </button>
              )}
            </>
          )}
        </div>
      )}

      {/* ── Rescan Results ── */}
      {rescanResult && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm flex items-center gap-1.5">
              <RefreshCw size={14} className="text-blue-400" /> Rescan Results
            </h3>
            <button onClick={() => setRescanResult(null)} className="text-slate-500 hover:text-white">
              <X size={16} />
            </button>
          </div>
          <p className="text-sm text-slate-400">
            Scanned {rescanResult.scanned} events — {rescanResult.candidates.length} potential match{rescanResult.candidates.length !== 1 ? "es" : ""}
          </p>

          {rescanResult.candidates.length > 0 ? (
            <>
              <div className="flex justify-between items-center">
                <button
                  onClick={() => {
                    if (rescanSelected.size === rescanResult.candidates.length) {
                      setRescanSelected(new Set());
                    } else {
                      setRescanSelected(new Set(rescanResult.candidates.map((c) => c.event_id)));
                    }
                  }}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  {rescanSelected.size === rescanResult.candidates.length ? "Deselect All" : "Select All"}
                </button>
                <span className="text-xs text-slate-500">{rescanSelected.size} selected</span>
              </div>

              <div className="space-y-1 max-h-[40vh] overflow-y-auto">
                {rescanResult.candidates.map((c) => (
                  <div
                    key={c.event_id}
                    className={`flex items-center gap-2 p-2 rounded-lg text-sm ${
                      rescanSelected.has(c.event_id) ? "bg-blue-900/20 border border-blue-800/40" : "bg-slate-800/50"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={rescanSelected.has(c.event_id)}
                      onChange={() => {
                        const next = new Set(rescanSelected);
                        next.has(c.event_id) ? next.delete(c.event_id) : next.add(c.event_id);
                        setRescanSelected(next);
                      }}
                      className="shrink-0 accent-blue-500"
                    />
                    {c.thumbnail_url && (
                      <button
                        onClick={() => showPreview(
                          c.snapshot_url || c.thumbnail_url,
                          `${c.camera_name} · ${new Date(c.timestamp).toLocaleString()}`
                        )}
                        className="w-10 h-8 rounded overflow-hidden bg-slate-800 shrink-0"
                      >
                        <img src={imgUrl(c.thumbnail_url)} alt="" className="w-full h-full object-contain" loading="lazy" />
                      </button>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs truncate">{c.camera_name}</p>
                      <p className="text-[10px] text-slate-500">{new Date(c.timestamp).toLocaleString()}</p>
                    </div>
                    <span className={`text-xs font-mono shrink-0 ${
                      c.similarity >= 0.85 ? "text-emerald-400" : c.similarity >= 0.80 ? "text-blue-400" : "text-amber-400"
                    }`}>
                      {(c.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>

              {rescanSelected.size > 0 && (
                <button
                  onClick={() => confirmMut.mutate(Array.from(rescanSelected))}
                  disabled={confirmMut.isPending}
                  className="btn-primary w-full text-sm flex items-center justify-center gap-2"
                >
                  <CheckCheck size={14} />
                  {confirmMut.isPending
                    ? "Confirming..."
                    : `Confirm ${rescanSelected.size} Match${rescanSelected.size !== 1 ? "es" : ""}`}
                </button>
              )}
            </>
          ) : (
            <p className="text-sm text-slate-500">No matches found in the last 12 hours.</p>
          )}
        </div>
      )}

      {/* Camera Breakdown */}
      {cameras.length > 0 && (
        <div className="card space-y-2">
          <h3 className="font-semibold text-sm flex items-center gap-1.5">
            <Camera size={14} className="text-slate-400" /> Camera Breakdown
          </h3>
          {cameras.map((cam) => {
            const pct = total_detections > 0 ? (cam.count / total_detections) * 100 : 0;
            return (
              <div key={cam.camera_id} className="flex items-center gap-2">
                <span className="text-sm flex-1 truncate">{cam.camera_name}</span>
                <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-600 rounded-full" style={{ width: `${pct}%` }} />
                </div>
                <span className="text-xs text-slate-500 w-8 text-right">{cam.count}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Recent Activity */}
      <div className="card space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-sm flex items-center gap-1.5">
            <Clock size={14} className="text-slate-400" /> Recent Activity ({recent_detections.length})
          </h3>
          <button
            onClick={() => { setRemoveMode(!removeMode); setRemoveSelected(new Set()); }}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              removeMode ? "bg-red-600/20 text-red-400" : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {removeMode ? "Cancel" : "Manage"}
          </button>
        </div>

        {recent_detections.length === 0 ? (
          <p className="text-sm text-slate-500">No detections yet</p>
        ) : (
          <div className="space-y-1 max-h-[50vh] overflow-y-auto">
            {recent_detections.map((det) => {
              const date = new Date(det.timestamp);
              return (
                <div
                  key={det.event_id}
                  className={`flex items-center gap-2 p-2 rounded-lg transition-colors group ${
                    removeSelected.has(det.event_id) ? "bg-red-900/20 border border-red-800/40" : "hover:bg-slate-800/50"
                  }`}
                >
                  {removeMode && (
                    <input
                      type="checkbox"
                      checked={removeSelected.has(det.event_id)}
                      onChange={() => {
                        const next = new Set(removeSelected);
                        next.has(det.event_id) ? next.delete(det.event_id) : next.add(det.event_id);
                        setRemoveSelected(next);
                      }}
                      className="shrink-0 accent-red-500"
                    />
                  )}
                  {det.thumbnail_url ? (
                    <button
                      onClick={() => showPreview(
                        det.snapshot_url || det.thumbnail_url!,
                        `${det.camera_name} · ${date.toLocaleString()}`
                      )}
                      className="w-12 h-9 rounded overflow-hidden bg-slate-800 shrink-0 relative"
                    >
                      <img src={imgUrl(det.thumbnail_url)} alt="" className="w-full h-full object-contain" loading="lazy" />
                      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                        <Eye size={12} className="text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </button>
                  ) : (
                    <div className="w-12 h-9 rounded bg-slate-800 shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm truncate">{det.camera_name}</p>
                    {det.narrative ? (
                      <p className="text-xs text-slate-400 italic truncate">{det.narrative}</p>
                    ) : (
                      <p className="text-xs text-slate-500">
                        {det.confidence != null && `${(det.confidence * 100).toFixed(0)}%`}
                      </p>
                    )}
                  </div>
                  <div className="text-right shrink-0">
                    <p className="text-sm font-medium">
                      {date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
                    </p>
                    <p className="text-[10px] text-slate-500">
                      {date.toLocaleDateString(undefined, { day: "numeric", month: "short" })}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {removeMode && removeSelected.size > 0 && (
          <button
            onClick={() => removeDetMut.mutate(Array.from(removeSelected))}
            disabled={removeDetMut.isPending}
            className="btn-danger w-full text-sm flex items-center justify-center gap-2 mt-2"
          >
            <Trash2 size={14} />
            {removeDetMut.isPending
              ? "Removing..."
              : `Remove ${removeSelected.size} Detection${removeSelected.size !== 1 ? "s" : ""}`}
          </button>
        )}
      </div>

      {/* Preview Modal */}
      {previewUrl && (
        <div
          className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-4"
          onClick={() => setPreviewUrl(null)}
        >
          <div className="max-w-lg w-full space-y-2" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <p className="text-sm text-slate-300">{previewLabel}</p>
              <button onClick={() => setPreviewUrl(null)} className="text-slate-400 hover:text-white">
                <X size={20} />
              </button>
            </div>
            <img src={imgUrl(previewUrl)} alt="Detection" className="w-full rounded-lg" />
          </div>
        </div>
      )}
    </div>
  );
}
