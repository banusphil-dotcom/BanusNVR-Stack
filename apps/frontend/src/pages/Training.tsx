import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, getToken } from "../api";
import {
  Plus, Upload, Trash2, X, User, Cat, Car, Box, Check, Tag,
  ChevronLeft, ChevronRight, Eye, Layers, Camera, Radio, Video, AlertTriangle,
  ShieldAlert, Loader2, ExternalLink, Crosshair, ArrowRight, SkipForward, RotateCcw,
} from "lucide-react";

interface NamedObjectInfo {
  id: number;
  name: string;
  category: string;
  reference_image_count: number;
  last_seen: string | null;
  last_camera: string | null;
  created_at: string;
}

interface NamedObjectStatus {
  id: number;
  name: string;
  category: string;
  reference_image_count: number;
  is_live: boolean;
  live_camera_id: number | null;
  live_camera_name: string | null;
  last_camera_id: number | null;
  last_camera_name: string | null;
  last_seen_at: string | null;
  snapshot_url: string | null;
  thumbnail_url: string | null;
  needs_retrain: boolean;
  retrain_reasons: string[];
  attributes: Record<string, any> | null;
}

interface UnrecognizedDetection {
  event_id: number;
  camera_id: number;
  camera_name: string;
  object_type: string;
  confidence: number | null;
  thumbnail_url: string;
  snapshot_url: string | null;
  timestamp: string;
}

interface UnrecognizedPage {
  items: UnrecognizedDetection[];
  total: number;
  page: number;
  page_size: number;
}

interface ClusterEvent {
  event_id: number;
  camera_id: number;
  camera_name: string;
  object_type: string;
  confidence: number | null;
  thumbnail_url: string;
  snapshot_url: string | null;
  timestamp: string;
}

interface Cluster {
  cluster_id: number;
  object_type: string;
  size: number;
  representative: ClusterEvent;
  events: ClusterEvent[];
}

interface ClustersResponse {
  clusters: Cluster[];
  total_clustered: number;
  unclustered_count: number;
}

interface IntegrityResult {
  object_id: number;
  name: string;
  category: string;
  consistent: boolean;
  outlier_indices: number[];
  confidence: number;
  reasoning: string;
  checked_images: number;
  error?: string;
}

interface IntegrityResponse {
  total_checked: number;
  flagged: number;
  results: IntegrityResult[];
}

interface CrossAuditProfile {
  id: number;
  name: string;
  category: string;
  thumbnail_url: string;
  attributes: Record<string, any>;
}

interface FlaggedDetection {
  event_id: number;
  thumbnail_url: string;
  assigned_to_id: number;
  assigned_to_name: string;
  similarity: number;
  camera_name: string;
  timestamp: string;
  threshold: number;
}

interface CrossAuditResponse {
  profiles: CrossAuditProfile[];
  flagged: FlaggedDetection[];
  total_audited: number;
}

interface ReviewAction {
  event_id: number;
  original_name: string;
  new_object_id: number | null;
  new_name: string | null;
}

const CATEGORIES = [
  { value: "person", label: "Person", icon: User },
  { value: "pet", label: "Pet", icon: Cat },
  { value: "vehicle", label: "Vehicle", icon: Car },
  { value: "other", label: "Other", icon: Box },
];

const CATEGORY_ICONS: Record<string, typeof User> = {
  person: User, pet: Cat, vehicle: Car, other: Box,
};

export default function Training() {
  const [tab, setTab] = useState<"profiles" | "unrecognized" | "clusters">("profiles");

  return (
    <div className="p-4 space-y-4 pb-24">
      <h2 className="text-lg font-bold">Profiles</h2>

      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
        {[
          { key: "profiles" as const, label: "All Profiles" },
          { key: "unrecognized" as const, label: "Unrecognized" },
          { key: "clusters" as const, label: "Clusters" },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
              tab === key ? "bg-blue-600 text-white" : "text-slate-400 hover:text-white"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "profiles" ? <ProfilesGrid /> : tab === "unrecognized" ? <UnrecognizedTab /> : <ClustersTab />}
    </div>
  );
}

/* ======================= Profiles Grid (Primary View) ======================= */

function ProfilesGrid() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const token = getToken();
  const [showCreate, setShowCreate] = useState(false);
  const [showMismatchReview, setShowMismatchReview] = useState(false);
  const [catFilter, setCatFilter] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [liveSnapshots, setLiveSnapshots] = useState<Record<number, string>>({});
  const [integrityResult, setIntegrityResult] = useState<IntegrityResponse | null>(null);
  const [integrityRunning, setIntegrityRunning] = useState(false);

  const { data: objectsStatus, isLoading } = useQuery({
    queryKey: ["named-objects-status"],
    queryFn: () => api.get<NamedObjectStatus[]>("/api/search/named-objects-status"),
    refetchInterval: 10000,
  });

  // Refresh live snapshots
  useEffect(() => {
    if (!objectsStatus) return;
    const liveIds = objectsStatus
      .filter((o) => o.is_live && o.live_camera_id)
      .map((o) => o.live_camera_id!);
    if (liveIds.length === 0) return;
    let cancelled = false;
    const refresh = async () => {
      const snaps: Record<number, string> = {};
      for (const camId of liveIds) {
        try {
          const r = await fetch(`/go2rtc/api/frame.jpeg?src=camera_${camId}&_=${Date.now()}`);
          if (r.ok) {
            const blob = await r.blob();
            snaps[camId] = URL.createObjectURL(blob);
          }
        } catch { /* */ }
      }
      if (!cancelled) setLiveSnapshots((prev) => {
        Object.values(prev).forEach((u) => URL.revokeObjectURL(u));
        return snaps;
      });
    };
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => { cancelled = true; clearInterval(iv); };
  }, [objectsStatus]);

  const imgUrl = (url: string) => `${url}?token=${encodeURIComponent(token || "")}`;

  const formatTimeAgo = (iso: string) => {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  const filtered = objectsStatus?.filter(
    (o) => !catFilter || o.category === catFilter
  );

  return (
    <>
      {/* Filter + Create */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2 flex-wrap">
          {[
            { value: "", label: "All" },
            { value: "person", label: "People" },
            { value: "pet", label: "Pets" },
            { value: "vehicle", label: "Vehicles" },
          ].map((f) => (
            <button
              key={f.value}
              onClick={() => setCatFilter(f.value)}
              className={`px-3 py-1.5 text-xs rounded-full transition-colors ${
                catFilter === f.value
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:text-white"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <button onClick={() => setShowCreate(true)} className="btn-primary text-sm py-2 flex items-center gap-1.5 shrink-0">
          <Plus size={16} /> New
        </button>
      </div>

      {/* Profile Integrity Check */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={async () => {
            setIntegrityRunning(true);
            setIntegrityResult(null);
            try {
              const res = await api.post<IntegrityResponse>("/api/training/integrity-check-all", {});
              setIntegrityResult(res);
            } catch { /* */ }
            setIntegrityRunning(false);
          }}
          disabled={integrityRunning}
          className="card flex items-center gap-2 px-3 py-2 hover:border-amber-500/50 transition-colors text-sm font-medium"
        >
          {integrityRunning
            ? <Loader2 size={14} className="animate-spin text-amber-400" />
            : <ShieldAlert size={14} className="text-amber-400" />}
          {integrityRunning ? "Checking profiles..." : "Profile Integrity Check"}
        </button>
        <button
          onClick={() => setShowMismatchReview(true)}
          className="card flex items-center gap-2 px-3 py-2 hover:border-blue-500/50 transition-colors text-sm font-medium"
        >
          <Crosshair size={14} className="text-blue-400" />
          Review Mismatches
        </button>
        {integrityResult && (
          <span className="text-xs text-slate-400">
            {integrityResult.total_checked} checked · {integrityResult.flagged > 0
              ? <span className="text-red-400 font-medium">{integrityResult.flagged} flagged</span>
              : <span className="text-emerald-400">All clean</span>}
          </span>
        )}
      </div>

      {/* Integrity Results Panel */}
      {integrityResult && integrityResult.flagged > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm flex items-center gap-1.5">
              <ShieldAlert size={14} className="text-amber-400" /> Profiles with Mixed Identities
            </h3>
            <button onClick={() => setIntegrityResult(null)} className="text-slate-500 hover:text-white">
              <X size={16} />
            </button>
          </div>
          <p className="text-xs text-slate-400">
            These profiles may have images of the wrong person or animal mixed in.
            Tap a profile to review and remove bad entries with AI Audit.
          </p>
          <div className="space-y-2">
            {integrityResult.results
              .filter((r) => !r.consistent)
              .map((r) => {
                const Icon = CATEGORY_ICONS[r.category] || Box;
                return (
                  <button
                    key={r.object_id}
                    onClick={() => navigate(`/profiles/${r.object_id}`)}
                    className="w-full flex items-center gap-3 p-3 rounded-lg bg-red-900/20 border border-red-800/40 hover:border-red-600/60 transition-colors text-left"
                  >
                    <Icon size={20} className="text-slate-400 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{r.name}</p>
                      <p className="text-xs text-slate-400 truncate">{r.reasoning}</p>
                      <p className="text-[10px] text-slate-500 mt-0.5">
                        {r.outlier_indices.length} outlier{r.outlier_indices.length !== 1 ? "s" : ""} in {r.checked_images} images · {r.confidence}% confidence
                      </p>
                    </div>
                    <ExternalLink size={14} className="text-slate-500 shrink-0" />
                  </button>
                );
              })}
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="text-center py-12 text-slate-400">Loading profiles...</div>
      ) : !filtered?.length ? (
        <div className="card text-center py-12 text-slate-400">
          <User size={48} className="mx-auto mb-3 opacity-50" />
          <p>{catFilter ? "No matching profiles" : "No profiles yet"}</p>
          <p className="text-xs mt-1">Create a profile or assign detections from the Unrecognized tab</p>
        </div>
      ) : (
        (() => {
          // Group filtered profiles by category
          const order = ["person", "pet", "vehicle", "other"];
          const labels: Record<string, string> = { person: "People", pet: "Pets", vehicle: "Vehicles", other: "Other" };
          const groups: Record<string, NamedObjectStatus[]> = {};
          for (const o of filtered) {
            const key = order.includes(o.category) ? o.category : "other";
            (groups[key] ||= []).push(o);
          }
          return (
            <div className="space-y-3">
              {order.map((cat) => {
                const items = groups[cat];
                if (!items || items.length === 0) return null;
                const Icon = CATEGORY_ICONS[cat] || Box;
                const isCollapsed = collapsed[cat];
                const liveCount = items.filter((o) => o.is_live).length;
                return (
                  <div key={cat} className="card p-0 overflow-hidden">
                    <button
                      onClick={() => setCollapsed((c) => ({ ...c, [cat]: !c[cat] }))}
                      className="w-full flex items-center gap-2 px-3 py-3 hover:bg-slate-800/40 transition-colors"
                    >
                      {isCollapsed
                        ? <ChevronRight size={16} className="text-slate-400" />
                        : <ChevronLeft size={16} className="text-slate-400 rotate-[-90deg]" />}
                      <Icon size={16} className="text-blue-400" />
                      <span className="font-semibold text-sm">{labels[cat]}</span>
                      <span className="text-xs text-slate-500">({items.length})</span>
                      {liveCount > 0 && (
                        <span className="ml-auto flex items-center gap-1 text-[10px] font-bold bg-red-600/20 text-red-300 px-2 py-0.5 rounded-full uppercase tracking-wider">
                          <Radio size={9} className="animate-pulse" /> {liveCount} live
                        </span>
                      )}
                    </button>
                    {!isCollapsed && (
                      <div className="p-3 pt-0">
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                          {items.map((obj) => {
                            const ItemIcon = CATEGORY_ICONS[obj.category] || Box;
                            const liveSnap = obj.is_live && obj.live_camera_id
                              ? liveSnapshots[obj.live_camera_id] : null;
                            const imageUrl = liveSnap
                              || (obj.thumbnail_url ? imgUrl(obj.thumbnail_url) : null)
                              || (obj.snapshot_url ? imgUrl(obj.snapshot_url) : null);
                            return (
                              <button
                                key={obj.id}
                                onClick={() => navigate(`/profiles/${obj.id}`)}
                                className="card p-0 overflow-hidden hover:border-blue-500/50 transition-colors text-left group relative"
                              >
                                <div className="aspect-[3/4] bg-slate-800 relative">
                                  {imageUrl ? (
                                    <img src={imageUrl} alt={obj.name}
                                      className="w-full h-full object-contain" loading="lazy" />
                                  ) : (
                                    <div className="w-full h-full flex items-center justify-center">
                                      <ItemIcon size={40} className="text-slate-600" />
                                    </div>
                                  )}
                                  {obj.is_live && (
                                    <div className="absolute top-1.5 left-1.5 flex items-center gap-1 bg-red-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider">
                                      <Radio size={10} className="animate-pulse" /> Live
                                    </div>
                                  )}
                                  {!obj.is_live && obj.needs_retrain && (
                                    <div className="absolute top-1.5 left-1.5 flex items-center gap-1 bg-amber-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider">
                                      <AlertTriangle size={10} /> Retrain
                                    </div>
                                  )}
                                  <div className="absolute top-1.5 right-1.5 bg-black/60 p-1 rounded">
                                    <ItemIcon size={12} className="text-slate-300" />
                                  </div>
                                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent pt-8 pb-2 px-2.5">
                                    <p className="font-semibold text-sm truncate text-white">{obj.name}</p>
                                    {obj.category === "pet" && obj.attributes?.breed && (
                                      <p className="text-[10px] text-blue-300 truncate">{obj.attributes.breed}{obj.attributes.color ? ` · ${obj.attributes.color}` : ""}</p>
                                    )}
                                    {obj.category === "vehicle" && (obj.attributes?.vehicle_type || obj.attributes?.make) && (
                                      <p className="text-[10px] text-blue-300 truncate">{[obj.attributes.vehicle_type, obj.attributes.make, obj.attributes.color].filter(Boolean).join(" · ")}</p>
                                    )}
                                    {obj.category === "person" && obj.attributes?.gender && (
                                      <p className="text-[10px] text-blue-300 truncate">{obj.attributes.gender === "male" ? "👨" : "👩"} {obj.attributes.age_group ? obj.attributes.age_group.replace("_", " ") : ""}</p>
                                    )}
                                    <div className="flex items-center gap-1 mt-0.5 text-[11px] text-slate-300">
                                      {obj.is_live ? (
                                        <>
                                          <Video size={10} className="text-green-400 shrink-0" />
                                          <span className="text-green-300 truncate">{obj.live_camera_name}</span>
                                        </>
                                      ) : obj.last_camera_name ? (
                                        <>
                                          <Camera size={10} className="shrink-0" />
                                          <span className="truncate">{obj.last_camera_name}</span>
                                          <span className="text-slate-400 shrink-0 ml-auto">
                                            {obj.last_seen_at ? formatTimeAgo(obj.last_seen_at) : ""}
                                          </span>
                                        </>
                                      ) : (
                                        <span className="text-slate-500 italic">Not seen yet</span>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })()
      )}

      {showCreate && <CreateObjectModal onClose={() => setShowCreate(false)} />}
      {showMismatchReview && (
        <MismatchReview
          onClose={() => setShowMismatchReview(false)}
          onDone={() => {
            setShowMismatchReview(false);
            qc.invalidateQueries({ queryKey: ["named-objects-status"] });
            qc.invalidateQueries({ queryKey: ["named-objects"] });
          }}
        />
      )}
    </>
  );
}

/* ======================= Unrecognized Detections Tab ======================= */

function UnrecognizedTab() {
  const qc = useQueryClient();
  const token = getToken();
  const [typeFilter, setTypeFilter] = useState("");
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [page, setPage] = useState(1);
  const [showAssign, setShowAssign] = useState(false);
  const [showCreateAndTrain, setShowCreateAndTrain] = useState(false);
  const [previewEvent, setPreviewEvent] = useState<UnrecognizedDetection | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["unrecognized", typeFilter, page],
    queryFn: () =>
      api.get<UnrecognizedPage>(
        `/api/training/unrecognized?page=${page}&page_size=50${typeFilter ? `&object_type=${typeFilter}` : ""}`
      ),
  });

  const toggleSelect = (eventId: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(eventId)) next.delete(eventId);
      else next.add(eventId);
      return next;
    });
  };

  const selectAll = () => {
    if (!data) return;
    if (selected.size === data.items.length) setSelected(new Set());
    else setSelected(new Set(data.items.map((d) => d.event_id)));
  };

  const imgUrl = (url: string) => `${url}?token=${encodeURIComponent(token || "")}`;

  const onDone = () => {
    setSelected(new Set());
    setShowAssign(false);
    setShowCreateAndTrain(false);
    qc.invalidateQueries({ queryKey: ["unrecognized"] });
    qc.invalidateQueries({ queryKey: ["named-objects"] });
    qc.invalidateQueries({ queryKey: ["named-objects-status"] });
  };

  return (
    <>
      <p className="text-sm text-slate-400">
        Detected people and pets that haven't been identified. Select to assign or create new profiles.
      </p>

      <div className="flex gap-2 flex-wrap">
        {[
          { value: "", label: "All" },
          { value: "person", label: "People" },
          { value: "pet", label: "Pets" },
          { value: "vehicle", label: "Vehicles" },
          { value: "other", label: "Other" },
        ].map((f) => (
          <button
            key={f.value}
            onClick={() => { setTypeFilter(f.value); setPage(1); setSelected(new Set()); }}
            className={`px-3 py-1.5 text-xs rounded-full transition-colors ${
              typeFilter === f.value ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="text-center py-12 text-slate-400">Loading...</div>
      ) : !data?.items.length ? (
        <div className="card text-center py-12 text-slate-400">
          <Check size={48} className="mx-auto mb-3 opacity-50" />
          <p>No unrecognized detections</p>
          <p className="text-xs mt-1">All recent people and pets have been identified</p>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between">
            <button onClick={selectAll} className="text-xs text-blue-400 hover:text-blue-300">
              {selected.size === data.items.length ? "Deselect All" : "Select All"}
            </button>
            <span className="text-xs text-slate-500">{data.total} unrecognized</span>
          </div>

          <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
            {data.items.map((det) => (
              <div key={det.event_id} className="relative group">
                <button
                  onClick={() => toggleSelect(det.event_id)}
                  className={`w-full relative rounded-lg overflow-hidden border-2 transition-all ${
                    selected.has(det.event_id) ? "border-blue-500 ring-1 ring-blue-500/50" : "border-transparent"
                  }`}
                >
                  <img src={imgUrl(det.thumbnail_url)} alt={det.object_type}
                    className="w-full aspect-[3/4] object-contain bg-slate-950" loading="lazy"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
                  {selected.has(det.event_id) && (
                    <div className="absolute top-1 right-1 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
                      <Check size={12} className="text-white" />
                    </div>
                  )}
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-1.5">
                    <p className="text-[10px] text-white font-medium truncate capitalize">{det.object_type}</p>
                    <p className="text-[9px] text-slate-300 truncate">{det.camera_name}</p>
                  </div>
                </button>
                {det.snapshot_url && (
                  <button
                    onClick={(e) => { e.stopPropagation(); setPreviewEvent(det); }}
                    className="absolute top-1 left-1 w-6 h-6 bg-black/60 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Eye size={12} className="text-white" />
                  </button>
                )}
              </div>
            ))}
          </div>

          {data.total > data.page_size && (() => {
            const totalPages = Math.ceil(data.total / data.page_size);
            return (
              <div className="flex items-center justify-center gap-2">
                <button className="btn-secondary text-sm py-1.5" disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
                  <ChevronLeft size={16} />
                </button>
                <span className="text-sm text-slate-400">{page} / {totalPages}</span>
                <button className="btn-secondary text-sm py-1.5" disabled={page === totalPages} onClick={() => setPage((p) => p + 1)}>
                  <ChevronRight size={16} />
                </button>
              </div>
            );
          })()}
        </>
      )}

      {selected.size > 0 && (
        <div className="fixed bottom-16 left-0 right-0 bg-slate-900/95 backdrop-blur border-t border-slate-700 p-3 flex gap-2 z-40">
          <button onClick={() => setShowAssign(true)}
            className="btn-primary flex-1 text-sm py-2.5 flex items-center justify-center gap-1.5">
            <Tag size={14} /> Assign ({selected.size})
          </button>
          <button onClick={() => setShowCreateAndTrain(true)}
            className="btn-secondary flex-1 text-sm py-2.5 flex items-center justify-center gap-1.5">
            <Plus size={14} /> New Profile
          </button>
        </div>
      )}

      {showAssign && <AssignModal eventIds={Array.from(selected)} onClose={() => setShowAssign(false)} onDone={onDone} />}
      {showCreateAndTrain && (
        <CreateAndTrainModal eventIds={Array.from(selected)} onClose={() => setShowCreateAndTrain(false)} onDone={onDone} />
      )}
      {previewEvent && <SnapshotPreview detection={previewEvent} onClose={() => setPreviewEvent(null)} />}
    </>
  );
}

/* ======================= Clusters Tab ======================= */

function ClustersTab() {
  const qc = useQueryClient();
  const token = getToken();
  const [typeFilter, setTypeFilter] = useState("");
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);
  const [showAssign, setShowAssign] = useState(false);
  const [showCreateAndTrain, setShowCreateAndTrain] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["clusters", typeFilter],
    queryFn: () =>
      api.get<ClustersResponse>(
        `/api/training/clusters?max_events=500${typeFilter ? `&object_type=${typeFilter}` : ""}`
      ),
  });

  const imgUrl = (url: string) => `${url}?token=${encodeURIComponent(token || "")}`;
  const selectedEventIds = selectedCluster?.events.map((e) => e.event_id) ?? [];

  const onDone = () => {
    setSelectedCluster(null);
    setShowAssign(false);
    setShowCreateAndTrain(false);
    qc.invalidateQueries({ queryKey: ["clusters"] });
    qc.invalidateQueries({ queryKey: ["unrecognized"] });
    qc.invalidateQueries({ queryKey: ["named-objects"] });
    qc.invalidateQueries({ queryKey: ["named-objects-status"] });
  };

  const catIcon = (type: string) => {
    const c = CATEGORIES.find((cat) => {
      if (cat.value === "pet" && ["cat", "dog"].includes(type)) return true;
      if (cat.value === "vehicle" && ["car", "truck", "bus", "motorcycle", "bicycle"].includes(type)) return true;
      return cat.value === type;
    });
    return c?.icon || Box;
  };

  return (
    <>
      <p className="text-sm text-slate-400">
        Auto-grouped detections that look similar. Assign a cluster to name all its detections at once.
      </p>

      <div className="flex gap-2 flex-wrap">
        {[
          { value: "", label: "All" },
          { value: "person", label: "People" },
          { value: "pet", label: "Pets" },
          { value: "vehicle", label: "Vehicles" },
        ].map((f) => (
          <button
            key={f.value}
            onClick={() => setTypeFilter(f.value)}
            className={`px-3 py-1.5 text-xs rounded-full transition-colors ${
              typeFilter === f.value ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="text-center py-12 text-slate-400">Loading...</div>
      ) : !data?.clusters.length ? (
        <div className="card text-center py-12 text-slate-400">
          <Layers size={48} className="mx-auto mb-3 opacity-50" />
          <p>No clusters found</p>
          <p className="text-xs mt-1">Clusters form as unnamed detections accumulate</p>
        </div>
      ) : (
        <div className="space-y-4">
          {data.clusters.filter((c) => c.size > 1).map((cluster) => {
            const Icon = catIcon(cluster.object_type);
            const isSelected = selectedCluster?.cluster_id === cluster.cluster_id;
            return (
              <div key={cluster.cluster_id}
                className={`card transition-colors ${isSelected ? "border-blue-500 ring-1 ring-blue-500/30" : ""}`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-7 h-7 rounded-full bg-slate-800 flex items-center justify-center shrink-0">
                    <Icon size={14} className="text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold capitalize">{cluster.object_type}</p>
                    <p className="text-[10px] text-slate-500">{cluster.size} detections</p>
                  </div>
                  <button onClick={() => setSelectedCluster(isSelected ? null : cluster)}
                    className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${
                      isSelected ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                    }`}>
                    {isSelected ? "Selected" : "Select"}
                  </button>
                </div>
                <div className="flex gap-2 overflow-x-auto pb-1 -mx-1 px-1 scrollbar-thin">
                  {cluster.events.map((ev) => (
                    <div key={ev.event_id} className="shrink-0 w-20">
                      <img src={imgUrl(ev.thumbnail_url)} alt={ev.object_type}
                        className="w-20 h-20 object-cover rounded-lg bg-slate-800" loading="lazy"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
                      <p className="text-[9px] text-slate-500 truncate mt-0.5">{ev.camera_name}</p>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
          {data.unclustered_count > 0 && (
            <p className="text-xs text-slate-500 text-center">
              + {data.unclustered_count} unique detections (no matches found)
            </p>
          )}
        </div>
      )}

      {selectedCluster && (
        <div className="fixed bottom-16 left-0 right-0 bg-slate-900/95 backdrop-blur border-t border-slate-700 p-3 flex gap-2 z-40">
          <button onClick={() => setShowAssign(true)}
            className="btn-primary flex-1 text-sm py-2.5 flex items-center justify-center gap-1.5">
            <Tag size={14} /> Assign ({selectedCluster.size})
          </button>
          <button onClick={() => setShowCreateAndTrain(true)}
            className="btn-secondary flex-1 text-sm py-2.5 flex items-center justify-center gap-1.5">
            <Plus size={14} /> New Profile
          </button>
        </div>
      )}

      {showAssign && <AssignModal eventIds={selectedEventIds} onClose={() => setShowAssign(false)} onDone={onDone} />}
      {showCreateAndTrain && (
        <CreateAndTrainModal eventIds={selectedEventIds} onClose={() => setShowCreateAndTrain(false)} onDone={onDone} />
      )}
    </>
  );
}

/* ======================= Assign to Existing Modal ======================= */

function AssignModal({ eventIds, onClose, onDone }: { eventIds: number[]; onClose: () => void; onDone: () => void }) {
  const { data: objects, isLoading } = useQuery({
    queryKey: ["named-objects"],
    queryFn: () => api.get<NamedObjectInfo[]>("/api/training/objects"),
  });

  const assignMut = useMutation({
    mutationFn: (objectId: number) =>
      api.post(`/api/training/objects/${objectId}/train-from-events`, { event_ids: eventIds }),
    onSuccess: onDone,
  });

  return (
    <div className="fixed inset-0 bg-black/70 flex items-end sm:items-center justify-center z-50 p-0 sm:p-4">
      <div className="bg-slate-900 rounded-t-2xl sm:rounded-2xl w-full max-w-sm p-5 space-y-4 max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between shrink-0">
          <h3 className="font-bold">Assign {eventIds.length} Detection{eventIds.length > 1 ? "s" : ""}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
        </div>
        <p className="text-sm text-slate-400 shrink-0">Select a profile:</p>
        {isLoading ? (
          <div className="text-center py-8 text-slate-400">Loading...</div>
        ) : !objects?.length ? (
          <p className="text-sm text-slate-500 py-4 text-center">No profiles yet. Use "New Profile" instead.</p>
        ) : (
          <div className="space-y-2 overflow-y-auto flex-1 min-h-0">
            {objects.map((obj) => {
              const catInfo = CATEGORIES.find((c) => c.value === obj.category);
              const Icon = catInfo?.icon || Box;
              return (
                <button key={obj.id} onClick={() => assignMut.mutate(obj.id)} disabled={assignMut.isPending}
                  className="w-full card flex items-center gap-3 hover:border-blue-500 transition-colors text-left">
                  <div className="w-9 h-9 rounded-full bg-slate-800 flex items-center justify-center shrink-0">
                    <Icon size={16} className="text-blue-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="font-medium text-sm truncate">{obj.name}</p>
                    <p className="text-xs text-slate-500">{catInfo?.label} . {obj.reference_image_count} images</p>
                  </div>
                </button>
              );
            })}
          </div>
        )}
        {assignMut.isPending && <p className="text-sm text-blue-400 text-center animate-pulse shrink-0">Training model...</p>}
      </div>
    </div>
  );
}

/* ======================= Create & Train Modal ======================= */

function CreateAndTrainModal({ eventIds, onClose, onDone }: { eventIds: number[]; onClose: () => void; onDone: () => void }) {
  const [name, setName] = useState("");
  const [category, setCategory] = useState("person");
  const [error, setError] = useState("");

  const createMut = useMutation({
    mutationFn: (data: { name: string; category: string; event_ids: number[] }) =>
      api.post("/api/training/create-and-train", data),
    onSuccess: onDone,
    onError: (err: any) => setError(err.message),
  });

  return (
    <div className="fixed inset-0 bg-black/70 flex items-end sm:items-center justify-center z-50 p-0 sm:p-4">
      <div className="bg-slate-900 rounded-t-2xl sm:rounded-2xl w-full max-w-sm p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-bold">New Profile</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
        </div>
        <p className="text-sm text-slate-400">
          Create a profile and train from {eventIds.length} selected detection{eventIds.length > 1 ? "s" : ""}.
        </p>
        <div>
          <label className="label">Name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Fred, Buddy" autoFocus />
        </div>
        <div>
          <label className="label">Category</label>
          <div className="grid grid-cols-2 gap-2">
            {CATEGORIES.map((cat) => (
              <button key={cat.value} onClick={() => setCategory(cat.value)}
                className={`card text-center py-3 cursor-pointer transition-colors ${
                  category === cat.value ? "border-blue-500" : ""
                }`}>
                <cat.icon size={20} className="mx-auto mb-1 text-blue-400" />
                <span className="text-sm">{cat.label}</span>
              </button>
            ))}
          </div>
        </div>
        {error && <p className="text-red-400 text-sm">{error}</p>}
        <button onClick={() => createMut.mutate({ name, category, event_ids: eventIds })}
          className="btn-primary w-full" disabled={!name.trim() || createMut.isPending}>
          {createMut.isPending ? "Creating & Training..." : "Create & Train"}
        </button>
      </div>
    </div>
  );
}

/* ======================= Snapshot Preview ======================= */

function SnapshotPreview({ detection, onClose }: { detection: UnrecognizedDetection; onClose: () => void }) {
  const token = getToken();
  const imgUrl = detection.snapshot_url ? `${detection.snapshot_url}?token=${encodeURIComponent(token || "")}` : null;
  if (!imgUrl) return null;
  return (
    <div className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="max-w-lg w-full space-y-2" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-300 capitalize">
            {detection.object_type} . {detection.camera_name} . {new Date(detection.timestamp).toLocaleString()}
          </p>
          <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
        </div>
        <img src={imgUrl} alt="Snapshot" className="w-full rounded-lg" />
      </div>
    </div>
  );
}

/* ======================= Create Profile Wizard (5-step) ======================= */

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

const PERSON_GENDERS = [
  { value: "male", label: "👨 Male" },
  { value: "female", label: "👩 Female" },
];

const PERSON_AGE_GROUPS = [
  { value: "child", label: "👶 Child" },
  { value: "young_adult", label: "🧑 Young Adult" },
  { value: "adult", label: "🧑‍💼 Adult" },
  { value: "middle_aged", label: "👨‍🦳 Middle Aged" },
  { value: "senior", label: "👴 Senior" },
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

const WIZARD_STEPS = [
  { num: 1, label: "Category" },
  { num: 2, label: "Details" },
  { num: 3, label: "Select" },
  { num: 4, label: "Review" },
  { num: 5, label: "Confirm" },
];

function CreateObjectModal({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const token = getToken();
  const [step, setStep] = useState(1);

  // Step 1
  const [category, setCategory] = useState("");

  // Step 2: Details + attributes
  const [name, setName] = useState("");
  const [gender, setGender] = useState("");
  const [ageGroup, setAgeGroup] = useState("");
  const [breed, setBreed] = useState("");
  const [color, setColor] = useState("");
  const [markings, setMarkings] = useState("");
  const [vehicleType, setVehicleType] = useState("");
  const [vehicleColor, setVehicleColor] = useState("");
  const [make, setMake] = useState("");

  // Step 3: Select training images (paginated)
  const [trainingPage, setTrainingPage] = useState(1);
  const [selectedDetections, setSelectedDetections] = useState<Map<number, UnrecognizedDetection>>(new Map());

  const [error, setError] = useState("");

  const objectType = category === "pet" ? "cat" : category === "vehicle" ? "car" : category;
  const { data: unrecognized, isLoading: unrecLoading } = useQuery({
    queryKey: ["wizard-unrecognized", objectType, trainingPage],
    queryFn: () =>
      api.get<UnrecognizedPage>(
        `/api/training/unrecognized?page=${trainingPage}&page_size=20${objectType ? `&object_type=${objectType}` : ""}`
      ),
    enabled: step === 3 && !!category,
  });

  const hasMorePages = unrecognized ? trainingPage * unrecognized.page_size < unrecognized.total : false;

  const createMut = useMutation({
    mutationFn: async () => {
      const eventIds = Array.from(selectedDetections.keys());
      let profileId: number;

      if (eventIds.length > 0) {
        const res = await api.post<{ id: number }>("/api/training/create-and-train", {
          name, category, event_ids: eventIds,
        });
        profileId = res.id;
      } else {
        const res = await api.post<{ id: number }>("/api/training/objects", { name, category });
        profileId = res.id;
      }

      // Apply attributes via PATCH
      const patchData: Record<string, string> = {};
      if (category === "person") {
        if (gender) patchData.gender = gender;
        if (ageGroup) patchData.age_group = ageGroup;
      } else if (category === "pet") {
        if (breed) patchData.breed = breed;
        if (color) patchData.color = color;
        if (markings) patchData.markings = markings;
      } else if (category === "vehicle") {
        if (vehicleType) patchData.vehicle_type = vehicleType;
        if (vehicleColor) patchData.color = vehicleColor;
        if (make) patchData.make = make;
      }
      if (Object.keys(patchData).length > 0) {
        await api.patch(`/api/training/objects/${profileId}`, patchData);
      }

      return profileId;
    },
    onSuccess: (profileId) => {
      qc.invalidateQueries({ queryKey: ["named-objects"] });
      qc.invalidateQueries({ queryKey: ["named-objects-status"] });
      onClose();
      navigate(`/profiles/${profileId}`);
    },
    onError: (err: any) => setError(err.message),
  });

  const imgUrl = (url: string) => `${url}?token=${encodeURIComponent(token || "")}`;

  const toggleDetection = (det: UnrecognizedDetection) => {
    setSelectedDetections((prev) => {
      const next = new Map(prev);
      if (next.has(det.event_id)) next.delete(det.event_id);
      else next.set(det.event_id, det);
      return next;
    });
  };

  // Build summary tags for step 5
  const summaryTags = (): string[] => {
    const tags: string[] = [];
    if (category === "person") {
      if (gender) tags.push(gender === "male" ? "👨 Male" : "👩 Female");
      if (ageGroup) tags.push(ageGroup.replace("_", " "));
    } else if (category === "pet") {
      if (breed) tags.push(breed);
      if (color) tags.push(color);
      if (markings) tags.push(markings);
    } else if (category === "vehicle") {
      if (vehicleType) tags.push(vehicleType);
      if (vehicleColor) tags.push(vehicleColor);
      if (make) tags.push(make);
    }
    return tags;
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-end sm:items-center justify-center z-50 p-0 sm:p-4">
      <div className="bg-slate-900 rounded-t-2xl sm:rounded-2xl w-full max-w-md p-5 space-y-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="font-bold">New Profile</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-1">
          {WIZARD_STEPS.map((s, i) => {
            const completed = step > s.num;
            const active = step === s.num;
            return (
              <div key={s.num} className="flex items-center gap-1 flex-1">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                  completed ? "bg-green-600 text-white" :
                  active ? "bg-blue-600 text-white" :
                  "bg-slate-700 text-slate-400"
                }`}>
                  {completed ? <Check size={12} /> : s.num}
                </div>
                <span className={`text-[9px] truncate ${active ? "text-white" : "text-slate-500"}`}>{s.label}</span>
                {i < WIZARD_STEPS.length - 1 && (
                  <div className={`h-px flex-1 ${completed ? "bg-green-600" : "bg-slate-700"}`} />
                )}
              </div>
            );
          })}
        </div>

        {/* Step 1: Category */}
        {step === 1 && (
          <div className="space-y-3">
            <p className="text-sm text-slate-400">What are you tracking?</p>
            <div className="grid grid-cols-2 gap-3">
              {CATEGORIES.map((cat) => (
                <button key={cat.value}
                  onClick={() => { setCategory(cat.value); setStep(2); }}
                  className="card text-center py-5 cursor-pointer hover:border-blue-500/50 transition-colors">
                  <cat.icon size={28} className="mx-auto mb-2 text-blue-400" />
                  <span className="text-sm font-medium">{cat.label}</span>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    {cat.value === "person" ? "People & visitors" :
                     cat.value === "pet" ? "Cats, dogs & animals" :
                     cat.value === "vehicle" ? "Cars, bikes & trucks" :
                     "Other objects"}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Step 2: Details + Attributes */}
        {step === 2 && (
          <div className="space-y-3">
            <div>
              <label className="label">Name</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)}
                placeholder={category === "pet" ? "e.g. Whiskers, Luna" : category === "person" ? "e.g. Fred, Sarah" : category === "vehicle" ? "e.g. Red Tesla, Family Car" : "e.g. Package, Ball"}
                autoFocus />
            </div>

            {/* Person attributes */}
            {category === "person" && (
              <>
                <div>
                  <label className="label">Gender (optional)</label>
                  <div className="flex gap-2">
                    {PERSON_GENDERS.map((g) => (
                      <button key={g.value} onClick={() => setGender(gender === g.value ? "" : g.value)}
                        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                          gender === g.value ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {g.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="label">Age Group (optional)</label>
                  <div className="flex flex-wrap gap-1.5">
                    {PERSON_AGE_GROUPS.map((ag) => (
                      <button key={ag.value} onClick={() => setAgeGroup(ageGroup === ag.value ? "" : ag.value)}
                        className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                          ageGroup === ag.value ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {ag.label}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Pet attributes */}
            {category === "pet" && (
              <>
                <div>
                  <label className="label">Breed (optional)</label>
                  <div className="flex flex-wrap gap-1.5">
                    {PET_BREEDS.map((b) => (
                      <button key={b} onClick={() => setBreed(breed === b ? "" : b)}
                        className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                          breed === b ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {b}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="label">Color (optional)</label>
                  <div className="flex flex-wrap gap-1.5">
                    {PET_COLORS.map((c) => (
                      <button key={c} onClick={() => setColor(color === c ? "" : c)}
                        className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                          color === c ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {c}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="label">Markings (optional)</label>
                  <div className="flex flex-wrap gap-1.5">
                    {PET_MARKINGS.map((m) => (
                      <button key={m} onClick={() => setMarkings(markings === m ? "" : m)}
                        className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                          markings === m ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Vehicle attributes */}
            {category === "vehicle" && (
              <>
                <div>
                  <label className="label">Type (optional)</label>
                  <div className="flex flex-wrap gap-1.5">
                    {VEHICLE_TYPES_LIST.map((vt) => (
                      <button key={vt} onClick={() => setVehicleType(vehicleType === vt ? "" : vt)}
                        className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                          vehicleType === vt ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {vt}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="label">Color (optional)</label>
                  <div className="flex flex-wrap gap-1.5">
                    {VEHICLE_COLORS_LIST.map((vc) => (
                      <button key={vc} onClick={() => setVehicleColor(vehicleColor === vc ? "" : vc)}
                        className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                          vehicleColor === vc ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {vc}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="label">Make (optional)</label>
                  <div className="flex flex-wrap gap-1.5">
                    {VEHICLE_MAKES_LIST.map((vm) => (
                      <button key={vm} onClick={() => setMake(make === vm ? "" : vm)}
                        className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                          make === vm ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}>
                        {vm}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}

            <div className="flex gap-2 pt-1">
              <button onClick={() => { setStep(1); setCategory(""); }}
                className="btn-secondary flex-1 text-sm py-2.5">Back</button>
              <button onClick={() => setStep(3)} disabled={!name.trim()}
                className="btn-primary flex-1 text-sm py-2.5">Next</button>
            </div>
          </div>
        )}

        {/* Step 3: Select Training Images */}
        {step === 3 && (
          <div className="space-y-3">
            <p className="text-sm text-slate-400">
              Select images of <span className="text-white font-medium">{name}</span> from recent detections.
            </p>

            {unrecLoading ? (
              <div className="text-center py-8 text-slate-400 text-sm">Loading detections...</div>
            ) : unrecognized?.items.length ? (
              <>
                <div className="grid grid-cols-4 gap-1.5 max-h-[40vh] overflow-y-auto">
                  {unrecognized.items.map((det) => {
                    const isSelected = selectedDetections.has(det.event_id);
                    return (
                      <button key={det.event_id} onClick={() => toggleDetection(det)}
                        className={`relative rounded-lg overflow-hidden border-2 transition-all ${
                          isSelected ? "border-blue-500 ring-1 ring-blue-500/50" : "border-transparent"
                        }`}>
                        <img src={imgUrl(det.thumbnail_url)} alt=""
                          className="w-full aspect-[3/4] object-contain bg-slate-950" loading="lazy" />
                        {isSelected && (
                          <div className="absolute top-0.5 right-0.5 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
                            <Check size={12} className="text-white" />
                          </div>
                        )}
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-1">
                          <p className="text-[8px] text-slate-300 truncate">{det.camera_name}</p>
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Not Present / Load More */}
                {hasMorePages && (
                  <button
                    onClick={() => setTrainingPage((p) => p + 1)}
                    className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors flex items-center justify-center gap-2"
                  >
                    <ChevronRight size={14} />
                    {name} not here? Load more...
                  </button>
                )}

                <p className="text-xs text-slate-500 text-center">
                  Page {trainingPage} of {Math.ceil((unrecognized?.total || 1) / (unrecognized?.page_size || 20))}
                  {selectedDetections.size > 0 && <> · <span className="text-blue-400">{selectedDetections.size} selected</span></>}
                </p>
              </>
            ) : (
              <div className="card text-center py-8 text-slate-500 text-sm">
                No unrecognized {category === "pet" ? "pet" : category} detections found
              </div>
            )}

            <div className="flex gap-2 pt-1">
              <button onClick={() => setStep(2)} className="btn-secondary flex-1 text-sm py-2.5">Back</button>
              <button onClick={() => setStep(selectedDetections.size > 0 ? 4 : 5)}
                className="btn-primary flex-1 text-sm py-2.5">
                {selectedDetections.size > 0 ? "Review Selection" : "Skip"}
              </button>
            </div>
          </div>
        )}

        {/* Step 4: Review Matches */}
        {step === 4 && (
          <div className="space-y-3">
            <p className="text-sm text-slate-300">
              Do these all look like <span className="text-white font-semibold">{name}</span>?
            </p>
            <p className="text-xs text-slate-500">Tap any image to remove it if it doesn't match.</p>

            <div className="grid grid-cols-3 gap-2 max-h-[45vh] overflow-y-auto">
              {Array.from(selectedDetections.values()).map((det) => (
                <button key={det.event_id} onClick={() => toggleDetection(det)}
                  className="relative rounded-lg overflow-hidden border-2 border-green-500/50 group">
                  <img src={imgUrl(det.thumbnail_url)} alt=""
                    className="w-full aspect-[3/4] object-contain bg-slate-950" loading="lazy" />
                  <div className="absolute inset-0 bg-red-900/0 group-hover:bg-red-900/40 transition-colors flex items-center justify-center">
                    <X size={20} className="text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                  <div className="absolute top-0.5 right-0.5 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                    <Check size={12} className="text-white" />
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-1">
                    <p className="text-[8px] text-slate-300 truncate">{det.camera_name}</p>
                  </div>
                </button>
              ))}
            </div>

            <p className="text-xs text-center text-slate-400">
              {selectedDetections.size} match{selectedDetections.size !== 1 ? "es" : ""} confirmed
            </p>

            <div className="flex gap-2 pt-1">
              <button onClick={() => setStep(3)} className="btn-secondary flex-1 text-sm py-2.5">
                Back
              </button>
              <button onClick={() => setStep(5)} disabled={selectedDetections.size === 0}
                className="btn-primary flex-1 text-sm py-2.5 flex items-center justify-center gap-1.5">
                <Check size={14} /> Yes, these match
              </button>
            </div>
          </div>
        )}

        {/* Step 5: Summary + Create */}
        {step === 5 && (
          <div className="space-y-3">
            <div className="card bg-slate-800/50 space-y-2">
              <div className="flex items-center gap-3">
                {(() => { const Icon = CATEGORY_ICONS[category] || Box; return <Icon size={24} className="text-blue-400" />; })()}
                <div>
                  <p className="font-semibold">{name}</p>
                  <p className="text-xs text-slate-400 capitalize">{category}</p>
                </div>
              </div>

              {summaryTags().length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {summaryTags().map((t) => (
                    <span key={t} className="bg-slate-700 rounded-full px-2 py-0.5 text-[10px] capitalize">{t}</span>
                  ))}
                </div>
              )}

              {selectedDetections.size > 0 && (
                <p className="text-xs text-slate-400">
                  {selectedDetections.size} detection{selectedDetections.size !== 1 ? "s" : ""} will be used for initial training
                </p>
              )}
            </div>

            {error && <p className="text-red-400 text-sm">{error}</p>}

            <div className="flex gap-2 pt-1">
              <button onClick={() => setStep(selectedDetections.size > 0 ? 4 : 3)}
                className="btn-secondary flex-1 text-sm py-2.5">Back</button>
              <button onClick={() => createMut.mutate()}
                disabled={createMut.isPending}
                className="btn-primary flex-1 text-sm py-2.5">
                {createMut.isPending ? "Creating..." : "Create Profile"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


/* ======================= Mismatch Review (Cross-Profile Audit) ======================= */

function MismatchReview({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const token = getToken();
  const [phase, setPhase] = useState<"scanning" | "review" | "summary" | "applying">("scanning");
  const [auditData, setAuditData] = useState<CrossAuditResponse | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [actions, setActions] = useState<ReviewAction[]>([]);
  const [error, setError] = useState("");
  const [applyError, setApplyError] = useState("");

  const imgUrl = (url: string) => `${url}?token=${encodeURIComponent(token || "")}`;

  // Run cross-audit on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await api.post<CrossAuditResponse>("/api/training/cross-audit", {});
        if (!cancelled) {
          setAuditData(res);
          if (res.flagged.length === 0) {
            setPhase("summary");
          } else {
            setPhase("review");
          }
        }
      } catch (err: any) {
        if (!cancelled) setError(err.message || "Audit failed");
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const currentFlagged = auditData?.flagged[currentIdx] ?? null;
  const profiles = auditData?.profiles ?? [];

  // Get the profile thumbnail for a given profile id
  const profileById = (id: number) => profiles.find((p) => p.id === id);

  const handleChoice = (newObjectId: number | null, newName: string | null) => {
    if (!currentFlagged) return;
    // Only record if the choice differs from current assignment
    if (newObjectId !== currentFlagged.assigned_to_id) {
      setActions((prev) => [
        ...prev,
        {
          event_id: currentFlagged.event_id,
          original_name: currentFlagged.assigned_to_name,
          new_object_id: newObjectId,
          new_name: newName,
        },
      ]);
    }
    advance();
  };

  const handleSkip = () => {
    advance();
  };

  const advance = () => {
    if (!auditData) return;
    if (currentIdx + 1 < auditData.flagged.length) {
      setCurrentIdx((i) => i + 1);
    } else {
      setPhase("summary");
    }
  };

  const handleApply = async () => {
    if (actions.length === 0) {
      onDone();
      return;
    }
    setPhase("applying");
    setApplyError("");
    try {
      await api.post("/api/training/reassign-detections", {
        actions: actions.map((a) => ({
          event_id: a.event_id,
          new_object_id: a.new_object_id,
        })),
      });
      onDone();
    } catch (err: any) {
      setApplyError(err.message || "Failed to apply changes");
      setPhase("summary");
    }
  };

  // Describe attributes briefly
  const describeAttrs = (attrs: Record<string, any>, category: string) => {
    if (!attrs || Object.keys(attrs).length === 0) return "";
    if (category === "person") {
      const parts: string[] = [];
      if (attrs.age_group) parts.push(attrs.age_group.replace("_", " "));
      if (attrs.gender) parts.push(attrs.gender);
      if (attrs.hair_color) parts.push(`${attrs.hair_color} hair`);
      return parts.join(", ");
    }
    if (category === "pet") {
      const parts: string[] = [];
      if (attrs.color) parts.push(attrs.color);
      if (attrs.breed) parts.push(attrs.breed);
      if (attrs.species) parts.push(attrs.species);
      return parts.join(" ");
    }
    return "";
  };

  return (
    <div className="fixed inset-0 bg-black/90 z-50 flex flex-col h-[100dvh]">
      {/* Header — compact */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700 shrink-0">
        <h2 className="font-bold text-sm flex items-center gap-1.5">
          <Crosshair size={14} className="text-blue-400" />
          Review Mismatches
        </h2>
        {phase === "review" && auditData && (
          <span className="text-[10px] text-slate-500">
            {currentIdx + 1}/{auditData.flagged.length}
            {actions.length > 0 && <span className="text-blue-400 ml-1">· {actions.length} fix{actions.length !== 1 ? "es" : ""}</span>}
          </span>
        )}
        <button onClick={onClose} className="text-slate-400 hover:text-white p-1">
          <X size={18} />
        </button>
      </div>

      {/* Scanning Phase */}
      {phase === "scanning" && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 p-6">
          {error ? (
            <>
              <AlertTriangle size={36} className="text-red-400" />
              <p className="text-red-400 text-center text-sm">{error}</p>
              <button onClick={onClose} className="btn-secondary text-sm">Close</button>
            </>
          ) : (
            <>
              <Loader2 size={36} className="text-blue-400 animate-spin" />
              <p className="text-slate-300 text-center text-sm">Scanning all profiles...</p>
            </>
          )}
        </div>
      )}

      {/* Review Phase — single screen layout */}
      {phase === "review" && currentFlagged && auditData && (
        <div className="flex-1 flex flex-col min-h-0 p-3 gap-2">
          {/* Progress bar */}
          <div className="bg-slate-800 rounded-full h-1 overflow-hidden shrink-0">
            <div
              className="bg-blue-500 h-full rounded-full transition-all duration-300"
              style={{ width: `${((currentIdx + 1) / auditData.flagged.length) * 100}%` }}
            />
          </div>

          {/* Top row: detection image + current assignment side by side */}
          <div className="flex gap-2 shrink-0">
            {/* Detection thumbnail */}
            <div className="relative rounded-lg overflow-hidden border-2 border-amber-500/50 w-24 shrink-0">
              <img
                src={imgUrl(currentFlagged.thumbnail_url)}
                alt="Detection"
                className="w-full aspect-[3/4] object-contain bg-slate-950"
              />
            </div>
            {/* Info + current assignment */}
            <div className="flex-1 min-w-0 flex flex-col justify-center gap-1.5">
              <p className="text-amber-300 text-xs font-semibold">Who is this?</p>
              <p className="text-[10px] text-slate-500 truncate">
                {currentFlagged.camera_name} · {(currentFlagged.similarity * 100).toFixed(0)}% match
              </p>
              {(() => {
                const assigned = profileById(currentFlagged.assigned_to_id);
                if (!assigned) return null;
                const attrDesc = describeAttrs(assigned.attributes, assigned.category);
                return (
                  <div className="flex items-center gap-2 bg-slate-800/60 rounded-lg p-1.5 mt-0.5">
                    <img
                      src={imgUrl(assigned.thumbnail_url)}
                      alt={assigned.name}
                      className="w-8 h-8 rounded object-cover bg-slate-700 shrink-0"
                      onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                    />
                    <div className="min-w-0">
                      <p className="text-[11px] text-slate-300 truncate">
                        Assigned: <span className="text-white font-medium">{assigned.name}</span>
                      </p>
                      {attrDesc && <p className="text-[9px] text-slate-500 truncate">{attrDesc}</p>}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>

          {/* Profile grid — takes remaining space, scrollable if many profiles */}
          <div className="flex-1 min-h-0 overflow-y-auto">
            <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1.5 sticky top-0 bg-black/90 py-0.5">Select correct profile</p>
            <div className="grid grid-cols-4 gap-1.5">
              {profiles
                .filter((p) => {
                  const assignedProfile = profileById(currentFlagged.assigned_to_id);
                  return assignedProfile ? p.category === assignedProfile.category : true;
                })
                .map((p) => {
                  const isCurrentlyAssigned = p.id === currentFlagged.assigned_to_id;
                  const attrDesc = describeAttrs(p.attributes, p.category);
                  return (
                    <button
                      key={p.id}
                      onClick={() => handleChoice(p.id, p.name)}
                      className={`relative rounded-lg overflow-hidden border transition-all text-left ${
                        isCurrentlyAssigned
                          ? "border-slate-700 opacity-40"
                          : "border-slate-700 hover:border-blue-500 active:border-blue-400 active:scale-95"
                      }`}
                    >
                      <div className="aspect-square bg-slate-800 relative">
                        <img
                          src={imgUrl(p.thumbnail_url)}
                          alt={p.name}
                          className="w-full h-full object-cover"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                        />
                        {isCurrentlyAssigned && (
                          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                            <span className="text-[8px] text-slate-400">current</span>
                          </div>
                        )}
                      </div>
                      <div className="px-1 py-0.5 bg-slate-900">
                        <p className="text-[10px] font-medium truncate">{p.name}</p>
                      </div>
                    </button>
                  );
                })}
            </div>
          </div>

          {/* Bottom action bar — fixed at bottom */}
          <div className="flex gap-1.5 shrink-0 pt-1">
            <button
              onClick={() => handleChoice(null, "Removed")}
              className="flex-1 flex items-center justify-center gap-1 py-2 rounded-lg bg-red-900/30 border border-red-800/40 text-red-300 text-xs hover:bg-red-900/50 transition-colors"
            >
              <Trash2 size={12} /> Remove
            </button>
            <button
              onClick={handleSkip}
              className="flex-1 flex items-center justify-center gap-1 py-2 rounded-lg bg-slate-800 text-slate-300 text-xs hover:bg-slate-700 transition-colors"
            >
              <SkipForward size={12} /> Skip
            </button>
            <button
              onClick={() => handleChoice(currentFlagged.assigned_to_id, currentFlagged.assigned_to_name)}
              className="flex-1 flex items-center justify-center gap-1 py-2 rounded-lg bg-emerald-900/30 border border-emerald-800/40 text-emerald-300 text-xs hover:bg-emerald-900/50 transition-colors"
            >
              <Check size={12} /> Correct
            </button>
          </div>
        </div>
      )}

      {/* Summary Phase */}
      {phase === "summary" && (
        <div className="flex-1 flex flex-col min-h-0 p-3">
          {auditData && auditData.flagged.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-3">
              <Check size={36} className="text-emerald-400" />
              <p className="font-semibold">All profiles look clean</p>
              <p className="text-xs text-slate-400">
                No suspicious detections across {auditData.total_audited} profiles
              </p>
              <button onClick={onClose} className="btn-primary mt-2 text-sm">Done</button>
            </div>
          ) : actions.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-3">
              <Check size={36} className="text-emerald-400" />
              <p className="font-semibold">No changes needed</p>
              <p className="text-xs text-slate-400">All flagged detections confirmed or skipped</p>
              <button onClick={onClose} className="btn-primary mt-2 text-sm">Done</button>
            </div>
          ) : (
            <>
              <div className="text-center shrink-0 pb-2">
                <h3 className="font-bold">Review Summary</h3>
                <p className="text-xs text-slate-400">
                  {actions.length} correction{actions.length !== 1 ? "s" : ""} to apply
                </p>
              </div>

              <div className="flex-1 overflow-y-auto min-h-0 space-y-1.5">
                {actions.map((action, i) => (
                  <div key={i} className="card flex items-center gap-2 p-2">
                    <img
                      src={imgUrl(`/api/events/${action.event_id}/crop`)}
                      alt="Detection"
                      className="w-8 h-8 rounded object-cover bg-slate-800 shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1 text-xs">
                        <span className="text-red-400 line-through truncate">{action.original_name}</span>
                        <ArrowRight size={10} className="text-slate-500 shrink-0" />
                        <span className={`font-medium truncate ${
                          action.new_object_id === null ? "text-red-300" : "text-emerald-300"
                        }`}>
                          {action.new_name || "Removed"}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => setActions((prev) => prev.filter((_, idx) => idx !== i))}
                      className="text-slate-500 hover:text-red-400 shrink-0 p-1"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>

              {applyError && <p className="text-red-400 text-xs text-center pt-1">{applyError}</p>}

              <div className="flex gap-2 pt-2 shrink-0">
                <button onClick={onClose} className="btn-secondary flex-1 text-sm py-2">Cancel</button>
                <button
                  onClick={handleApply}
                  className="btn-primary flex-1 text-sm py-2 flex items-center justify-center gap-1"
                >
                  <Check size={12} /> Apply {actions.length}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Applying Phase */}
      {phase === "applying" && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 p-6">
          <Loader2 size={36} className="text-blue-400 animate-spin" />
          <p className="text-slate-300 text-center text-sm">Applying corrections...</p>
        </div>
      )}
    </div>
  );
}
