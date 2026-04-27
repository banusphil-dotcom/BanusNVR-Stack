import { useState, useEffect, memo, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api, getToken } from "../api";
import { Play, Trash2, ChevronDown, Camera, RefreshCw, GraduationCap, User, Cat, Car, Box, ArrowRightLeft, Sparkles, RotateCcw, ImageUp, Eye, EyeOff, Users, Bell } from "lucide-react";
import { useReExtractModal, ReExtractModal } from "../components/ReExtractModal";
import { useReclassifyModal, ReclassifyModal } from "../components/ReclassifyModal";

interface Suggestion {
  named_object_id: number;
  name: string;
  confidence: number;
  face_similarity: number | null;
  body_similarity: number | null;
}

interface EventAnnotation {
  name: string;
  class_name: string;
  bbox: Record<string, number>;
  confidence: number | null;
  primary: boolean;
}

interface EventObj {
  id: number;
  camera_id: number;
  camera_name: string;
  event_type: string;
  object_type: string | null;
  named_object_id: number | null;
  named_object_name: string | null;
  confidence: number | null;
  snapshot_path: string | null;
  thumbnail_path: string | null;
  bbox: Record<string, number> | null;
  recording_path: string | null;
  started_at: string;
  ended_at: string | null;
  duration: number | null;
  gif_url: string | null;
  annotations: EventAnnotation[];
  narrative: string | null;
}

interface EventGroup {
  group_key: string;
  camera_id: number;
  camera_name: string;
  camera_names: string[];
  started_at: string;
  ended_at: string | null;
  duration: number | null;
  narrative: string;
  names: string[];
  object_count: number;
  primary_event_id: number;
  events: EventObj[];
}

interface EventGroupsPage {
  groups: EventGroup[];
  total_groups: number;
  page: number;
  page_size: number;
}

const OBJECT_TYPE_TO_CATEGORIES: Record<string, string[]> = {
  person: ["person"],
  cat: ["pet"],
  dog: ["pet"],
  bird: ["pet"],
  car: ["vehicle"],
  truck: ["vehicle"],
  bus: ["vehicle"],
  motorcycle: ["vehicle"],
  bicycle: ["vehicle"],
  boat: ["vehicle"],
};

function allowedCategoriesForEvent(ev: EventObj): string[] {
  if (!ev.object_type) return ["person", "pet", "vehicle", "other"];
  return OBJECT_TYPE_TO_CATEGORIES[ev.object_type] || ["other"];
}

function groupDisplayLabel(group: EventGroup): string {
  if (group.names.length > 0) {
    if (group.names.length === 1 && group.object_count === 1) {
      return group.names[0];
    }
    const others = group.object_count - group.names.length;
    if (others > 0) {
      return `${group.names.join(", ")} and ${others} other${others > 1 ? "s" : ""}`;
    }
    if (group.names.length === 2) return `${group.names[0]} and ${group.names[1]}`;
    if (group.names.length > 2) return `${group.names.slice(0, -1).join(", ")} and ${group.names[group.names.length - 1]}`;
    return group.names[0];
  }
  // No recognized names
  if (group.object_count === 1) {
    return group.events[0]?.object_type || "Detection";
  }
  return `${group.object_count} detections`;
}

function eventMemberLabel(ev: EventObj): string {
  if (ev.named_object_name) return ev.named_object_name;
  return ev.object_type || ev.event_type;
}

export default function Events() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [typeFilter, setTypeFilter] = useState("");
  const [recogTab, setRecogTab] = useState<"all" | "named" | "unknown">("all");
  const [expandedGroupKey, setExpandedGroupKey] = useState<string | null>(null);
  const [expandedMemberId, setExpandedMemberId] = useState<number | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const token = getToken();

  const imgUrl = (eventId: number, type: "snapshot" | "thumbnail" | "crop") =>
    `/api/events/${eventId}/${type}?token=${encodeURIComponent(token || "")}${type === "snapshot" ? `&annotated=${showBoxes}` : ""}`;

  // Mark notifications read when visiting this page
  useEffect(() => {
    api.post("/api/notifications/mark-read").catch(() => {});
    qc.invalidateQueries({ queryKey: ["unread-count"] });
  }, []);

  const { data, isLoading } = useQuery({
    queryKey: ["events-grouped", page, typeFilter, recogTab],
    queryFn: () => {
      const recogParam = recogTab === "named" ? "&recognised=true" : recogTab === "unknown" ? "&recognised=false" : "";
      return api.get<EventGroupsPage>(
        `/api/events/grouped?page=${page}&page_size=20${typeFilter ? `&object_type=${typeFilter}` : ""}${recogParam}`,
      );
    },
  });

  const deleteMut = useMutation({
    mutationFn: (id: number) => api.delete(`/api/events/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["events-grouped"] }),
  });

  const reclassify = useReclassifyModal();
  const reExtract = useReExtractModal();

  const [reanalyseResult, setReanalyseResult] = useState<{ eventId: number; name: string | null; method: string | null; objectType: string } | null>(null);
  const [reanalysePickerId, setReanalysePickerId] = useState<number | null>(null);
  const reanalyseMut = useMutation({
    mutationFn: ({ id, objectType }: { id: number; objectType: string }) =>
      api.post<{ named_object: string | null; match_method: string | null; object_type: string }>(
        `/api/events/${id}/reanalyse?object_type=${encodeURIComponent(objectType)}`
      ),
    onSuccess: (data, vars) => {
      setReanalyseResult({ eventId: vars.id, name: data.named_object, method: data.match_method, objectType: data.object_type });
      setReanalysePickerId(null);
      qc.invalidateQueries({ queryKey: ["events-grouped"] });
    },
  });

  // Train from event
  const [trainEventId, setTrainEventId] = useState<number | null>(null);
  const [trainName, setTrainName] = useState("");
  const [trainCategory, setTrainCategory] = useState("pet");

  const CATEGORY_OPTIONS = [
    { value: "person", label: "Person", icon: User },
    { value: "pet", label: "Pet", icon: Cat },
    { value: "vehicle", label: "Vehicle", icon: Car },
    { value: "other", label: "Other", icon: Box },
  ];

  const trainMut = useMutation({
    mutationFn: (payload: { name: string; category: string; event_ids: number[] }) =>
      api.post<{ id: number; name: string; trained: number }>("/api/training/create-and-train", payload),
    onSuccess: () => {
      setTrainEventId(null);
      setTrainName("");
      qc.invalidateQueries({ queryKey: ["events-grouped"] });
    },
  });

  // Reassign event to different named object
  const [reassignEventId, setReassignEventId] = useState<number | null>(null);

  // Suggestions for unrecognized person events
  const [suggestionsMap, setSuggestionsMap] = useState<Record<number, Suggestion[]>>({});
  const [loadingSuggestions, setLoadingSuggestions] = useState<number | null>(null);
  const [confirmAssign, setConfirmAssign] = useState<{ eventId: number; suggestion: Suggestion } | null>(null);

  interface NamedObj { id: number; name: string; category: string; }
  const { data: namedObjects } = useQuery({
    queryKey: ["named-objects"],
    queryFn: () => api.get<NamedObj[]>("/api/training/objects"),
    enabled: reassignEventId != null,
  });

  const [reassignError, setReassignError] = useState<string | null>(null);
  const reassignMut = useMutation({
    mutationFn: ({ eventId, namedObjectId }: { eventId: number; namedObjectId: number }) =>
      api.post(`/api/events/${eventId}/label`, { named_object_id: namedObjectId }),
    onSuccess: (_, vars) => {
      setReassignEventId(null);
      setReassignError(null);
      setConfirmAssign(null);
      setSuggestionsMap((prev) => { const next = { ...prev }; delete next[vars.eventId]; return next; });
      qc.invalidateQueries({ queryKey: ["events-grouped"] });
    },
    onError: (err: Error) => {
      setReassignError(err.message || "Reassign failed");
    },
  });

  // Fetch suggestions when expanding an unrecognized person event
  const fetchSuggestions = async (eventId: number) => {
    if (suggestionsMap[eventId]) return;
    setLoadingSuggestions(eventId);
    try {
      const res = await api.get<{ suggestions: Suggestion[] }>(`/api/events/${eventId}/suggestions`);
      setSuggestionsMap((prev) => ({ ...prev, [eventId]: res.suggestions }));
    } catch { /* ignore */ }
    setLoadingSuggestions(null);
  };

  const relativeTime = (iso: string) => {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return days === 1 ? "Yesterday" : `${days}d ago`;
  };

  const formatTime = (iso: string) =>
    new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const formatDuration = (secs: number) => {
    if (secs < 60) return `${Math.round(secs)}s`;
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m ${Math.round(secs % 60)}s`;
    const hrs = Math.floor(mins / 60);
    return `${hrs}h ${mins % 60}m`;
  };

  const gifUrl = (eventId: number) =>
    `/api/events/${eventId}/gif?token=${encodeURIComponent(token || "")}`;

  const SuggestionsBar = ({ eventId, suggestions, loading, onFetch, onSelect }: {
    eventId: number;
    suggestions?: Suggestion[];
    loading: boolean;
    onFetch: () => void;
    onSelect: (s: Suggestion) => void;
  }) => {
    if (loading) return <p className="text-xs text-slate-400">Analysing...</p>;
    if (!suggestions) return null;
    if (suggestions.length === 0) return <p className="text-xs text-slate-500">No matches found</p>;

    return (
      <div className="bg-slate-800/60 rounded-lg p-2.5 space-y-1.5">
        <p className="text-[10px] text-slate-400 font-medium flex items-center gap-1">
          <Sparkles size={10} /> I think this is...
        </p>
        <div className="flex flex-wrap gap-1.5">
          {suggestions.map((s) => (
            <button
              key={s.named_object_id}
              onClick={() => onSelect(s)}
              className={`text-xs py-1.5 px-3 rounded-full font-medium transition-colors flex items-center gap-1.5 ${
                s.confidence >= 40
                  ? "bg-green-900/50 text-green-300 hover:bg-green-800/60 border border-green-700/50"
                  : s.confidence >= 25
                  ? "bg-yellow-900/40 text-yellow-300 hover:bg-yellow-800/50 border border-yellow-700/40"
                  : "bg-slate-700/60 text-slate-300 hover:bg-slate-600/60 border border-slate-600/40"
              }`}
            >
              <User size={11} />
              {s.name}
              <span className="opacity-70">{s.confidence}%</span>
            </button>
          ))}
        </div>
      </div>
    );
  };

  /** Render per-member action buttons and panels */
  const MemberActions = ({ ev }: { ev: EventObj }) => (
    <div className="space-y-2">
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => navigate(`/camera/${ev.camera_id}?time=${encodeURIComponent(ev.started_at)}`)}
          className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
        >
          <Play size={12} /> Recording
        </button>
        <button
          onClick={() => setReanalysePickerId(reanalysePickerId === ev.id ? null : ev.id)}
          disabled={reanalyseMut.isPending && reanalyseMut.variables?.id === ev.id}
          className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
        >
          <RefreshCw size={12} className={reanalyseMut.isPending && reanalyseMut.variables?.id === ev.id ? "animate-spin" : ""} />
          Reanalyse
        </button>
        <button
          onClick={() => deleteMut.mutate(ev.id)}
          className="btn-danger text-xs py-1 px-2.5 flex items-center gap-1"
        >
          <Trash2 size={12} /> Delete
        </button>
        {!ev.named_object_name && (
          <button
            onClick={() => {
              setTrainEventId(trainEventId === ev.id ? null : ev.id);
              setTrainName("");
              const objType = ev.object_type || "";
              if (objType === "person") setTrainCategory("person");
              else if (["cat", "dog", "bird"].includes(objType)) setTrainCategory("pet");
              else if (["car", "truck", "bus", "motorcycle", "bicycle"].includes(objType)) setTrainCategory("vehicle");
              else setTrainCategory("other");
            }}
            className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
          >
            <GraduationCap size={12} /> Train
          </button>
        )}
        {ev.object_type && (
          <button
            onClick={() => {
              setReassignEventId(reassignEventId === ev.id ? null : ev.id);
              setTrainEventId(null);
            }}
            className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
          >
            <ArrowRightLeft size={12} /> Reassign
          </button>
        )}
      </div>
      {/* Reanalyse picker */}
      {reanalysePickerId === ev.id && (
        <div className="bg-slate-800 rounded-lg p-2.5 space-y-2">
          <p className="text-xs font-medium">Reanalyse as:</p>
          <div className="flex flex-wrap gap-1.5">
            {["person", "cat", "dog", "car", "truck", "motorcycle"].map((t) => (
              <button
                key={t}
                onClick={() => reanalyseMut.mutate({ id: ev.id, objectType: t })}
                disabled={reanalyseMut.isPending}
                className="btn-secondary text-xs py-1 px-2.5 capitalize flex items-center gap-1"
              >
                {t === "person" && <User size={12} />}
                {(t === "cat" || t === "dog") && <Cat size={12} />}
                {(t === "car" || t === "truck" || t === "motorcycle") && <Car size={12} />}
                {t}
              </button>
            ))}
          </div>
          <button onClick={() => setReanalysePickerId(null)} className="btn-secondary text-xs py-1 px-2.5">Cancel</button>
        </div>
      )}
      {/* Train form */}
      {trainEventId === ev.id && (
        <div className="bg-slate-800 rounded-lg p-2.5 space-y-2">
          <p className="text-xs font-medium">Create Named Object</p>
          <input
            type="text"
            placeholder="Name (e.g. Luna, My Car)"
            value={trainName}
            onChange={(e) => setTrainName(e.target.value)}
            className="input text-sm w-full"
            autoFocus
          />
          <div className="flex gap-1.5">
            {CATEGORY_OPTIONS.map((cat) => {
              const Icon = cat.icon;
              return (
                <button
                  key={cat.value}
                  onClick={() => setTrainCategory(cat.value)}
                  className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded text-[10px] font-medium transition-colors ${
                    trainCategory === cat.value
                      ? "bg-blue-600 text-white"
                      : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                  }`}
                >
                  <Icon size={12} /> {cat.label}
                </button>
              );
            })}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                if (trainName.trim()) {
                  trainMut.mutate({ name: trainName.trim(), category: trainCategory, event_ids: [ev.id] });
                }
              }}
              disabled={!trainName.trim() || trainMut.isPending}
              className="btn-primary text-xs py-1.5 px-4 flex-1 disabled:opacity-40"
            >
              {trainMut.isPending ? "Training..." : "Create & Train"}
            </button>
            <button onClick={() => setTrainEventId(null)} className="btn-secondary text-xs py-1 px-2.5">Cancel</button>
          </div>
        </div>
      )}
      {/* Reassign panel */}
      {reassignEventId === ev.id && (
        <div className="bg-slate-800 rounded-lg p-2.5 space-y-2">
          <p className="text-xs font-medium">Reassign to</p>
          {!namedObjects ? (
            <p className="text-xs text-slate-400">Loading...</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {(() => {
                const allowed = new Set(allowedCategoriesForEvent(ev));
                const candidates = namedObjects
                  .filter((o) => allowed.has(o.category) && o.id !== ev.named_object_id);
                if (candidates.length === 0) {
                  return <p className="text-xs text-slate-400">No matching named objects for this event type</p>;
                }
                return candidates.map((o) => (
                  <button
                    key={o.id}
                    onClick={() => { setReassignError(null); reassignMut.mutate({ eventId: ev.id, namedObjectId: o.id }); }}
                    disabled={reassignMut.isPending}
                    className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
                  >
                    {o.category === "person" && <User size={12} />}
                    {o.category === "pet" && <Cat size={12} />}
                    {o.category === "vehicle" && <Car size={12} />}
                    {o.category === "other" && <Box size={12} />}
                    {o.name}
                  </button>
                ));
              })()}
            </div>
          )}
          {reassignError && <p className="text-xs text-red-400">{reassignError}</p>}
          {reassignMut.isPending && <p className="text-xs text-blue-400">Reassigning...</p>}
          <button onClick={() => { setReassignEventId(null); setReassignError(null); }} className="btn-secondary text-xs py-1 px-2.5">Cancel</button>
        </div>
      )}
      {/* Suggestions for unrecognized person events */}
      {!ev.named_object_name && ev.object_type === "person" && (
        <SuggestionsBar
          eventId={ev.id}
          suggestions={suggestionsMap[ev.id]}
          loading={loadingSuggestions === ev.id}
          onFetch={() => fetchSuggestions(ev.id)}
          onSelect={(s) => setConfirmAssign({ eventId: ev.id, suggestion: s })}
        />
      )}
      {reanalyseResult?.eventId === ev.id && (
        <p className="text-xs text-green-400">
          Reanalysis as {reanalyseResult.objectType} — {reanalyseResult.name ? `matched: ${reanalyseResult.name} (${reanalyseResult.method})` : "no match found"}
        </p>
      )}
    </div>
  );

  return (
    <div className="p-4 space-y-3">
      {/* Confirm assign dialog */}
      {confirmAssign && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 rounded-xl p-5 max-w-sm w-full space-y-4 border border-slate-700">
            <p className="text-sm font-medium text-center">
              Assign this event to <span className="text-blue-400">{confirmAssign.suggestion.name}</span>?
            </p>
            <p className="text-xs text-slate-400 text-center">
              Confidence: {confirmAssign.suggestion.confidence}%
              {confirmAssign.suggestion.face_similarity != null && ` · Face: ${(confirmAssign.suggestion.face_similarity * 100).toFixed(0)}%`}
              {confirmAssign.suggestion.body_similarity != null && ` · Body: ${(confirmAssign.suggestion.body_similarity * 100).toFixed(0)}%`}
            </p>
            <p className="text-[10px] text-slate-500 text-center">
              This will train the model with this image to improve future recognition.
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => reassignMut.mutate({ eventId: confirmAssign.eventId, namedObjectId: confirmAssign.suggestion.named_object_id })}
                disabled={reassignMut.isPending}
                className="btn-primary text-sm py-2 flex-1"
              >
                {reassignMut.isPending ? "Assigning..." : "Yes, assign"}
              </button>
              <button
                onClick={() => setConfirmAssign(null)}
                className="btn-secondary text-sm py-2 flex-1"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Reclassify modal */}
      <ReclassifyModal
        open={reclassify.open} phase={reclassify.phase}
        progress={reclassify.progress} error={reclassify.error}
        onClose={() => { reclassify.close(); qc.invalidateQueries({ queryKey: ["events-grouped"] }); }}
      />
      {/* Re-extract modal */}
      <ReExtractModal
        open={reExtract.open} phase={reExtract.phase} scan={reExtract.scan}
        progress={reExtract.progress} result={reExtract.result} error={reExtract.error}
        onClose={() => { reExtract.close(); qc.invalidateQueries({ queryKey: ["events-grouped"] }); }}
      />
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold">Events</h2>
          <div className="flex items-center gap-2">
            <select
              className="input w-auto text-sm"
              value={typeFilter}
              onChange={(e) => { setTypeFilter(e.target.value); setPage(1); }}
            >
            <option value="">All</option>
            <option value="person">Person</option>
            <option value="cat">Cat</option>
            <option value="dog">Dog</option>
            <option value="car">Car</option>
            <option value="truck">Truck</option>
            </select>
            <button
              onClick={() => setShowBoxes(!showBoxes)}
              className={`p-1.5 rounded transition-colors ${showBoxes ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-400"}`}
              title={showBoxes ? "Hide bounding boxes" : "Show bounding boxes"}
            >
              {showBoxes ? <Eye size={16} /> : <EyeOff size={16} />}
            </button>
          </div>
        </div>
        {/* Recognised / Unrecognised tabs */}
        <div className="flex gap-1 bg-slate-900 rounded-lg p-1">
          {([
            { key: "all" as const, label: "All", icon: Bell },
            { key: "named" as const, label: "Recognised", icon: User },
            { key: "unknown" as const, label: "Unrecognised", icon: Users },
          ]).map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => { setRecogTab(key); setPage(1); }}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-sm font-medium transition-colors ${
                recogTab === key ? "bg-slate-800 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Icon size={14} /> {label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => reExtract.start()}
            disabled={reExtract.phase !== "idle" && reExtract.phase !== "done" && reExtract.phase !== "error"}
            className="btn btn-sm bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3 py-1.5 rounded flex items-center gap-1.5"
            title="Re-extract HD thumbnails from recordings for better face detection"
          >
            <ImageUp size={14} />
            Re-extract HD
          </button>
          <button
            onClick={() => reclassify.start()}
            disabled={reclassify.phase === "running"}
            className="btn btn-sm bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3 py-1.5 rounded flex items-center gap-1.5"
            title="Re-run face detection on last 24h of person events"
          >
            <RotateCcw size={14} className={reclassify.phase === "running" ? "animate-spin" : ""} />
            {reclassify.phase === "running" ? "Reclassifying…" : "Reclassify 24h"}
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="text-center py-12 text-slate-400">Loading...</div>
      ) : !data?.groups.length ? (
        <div className="text-slate-500 text-center py-16 space-y-2">
          <Camera size={40} className="mx-auto opacity-30" />
          <p>No events yet</p>
          <p className="text-xs">Events will appear here when objects are detected</p>
        </div>
      ) : (
        <div className="space-y-2">
          {data.groups.map((group) => {
            const isExpanded = expandedGroupKey === group.group_key;
            const isSingle = group.events.length === 1;
            const primaryEvent = group.events.find((e) => e.id === group.primary_event_id) || group.events[0];

            return (
              <div key={group.group_key} className="card p-0 overflow-hidden">
                {/* Group header row */}
                <button
                  className="w-full flex items-center gap-3 p-3 text-left"
                  onClick={() => {
                    const newKey = isExpanded ? null : group.group_key;
                    setExpandedGroupKey(newKey);
                    setExpandedMemberId(null);
                    // Fetch suggestions for unrecognized person events
                    if (newKey && isSingle && !primaryEvent.named_object_name && primaryEvent.object_type === "person") {
                      fetchSuggestions(primaryEvent.id);
                    }
                  }}
                >
                  {/* Group thumbnail — animated GIF when available, else static crop */}
                  {(primaryEvent.gif_url || primaryEvent.thumbnail_path || primaryEvent.snapshot_path) ? (
                    <div className="relative shrink-0">
                      <img
                        src={primaryEvent.gif_url ? gifUrl(primaryEvent.id) : imgUrl(primaryEvent.id, "crop")}
                        alt=""
                        loading="lazy"
                        className="w-16 h-12 rounded object-contain bg-slate-800"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                      />
                      {!isSingle && (
                        <span className="absolute -top-1 -right-1 bg-blue-600 text-white text-[10px] font-bold rounded-full w-5 h-5 flex items-center justify-center">
                          {group.object_count}
                        </span>
                      )}
                    </div>
                  ) : (
                    <div className="w-16 h-12 rounded bg-slate-800 shrink-0 flex items-center justify-center">
                      {!isSingle ? <Users size={16} className="text-slate-600" /> : <Camera size={16} className="text-slate-600" />}
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">
                      {groupDisplayLabel(group)}
                    </p>
                    {group.names.length > 1 && (
                      <div className="mt-0.5 flex flex-wrap gap-1">
                        {group.names.map((name) => (
                          <span key={name} className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300">
                            {name}
                          </span>
                        ))}
                      </div>
                    )}
                    <p className="text-xs text-slate-500">
                      {group.camera_names && group.camera_names.length > 1
                        ? group.camera_names.join(" → ")
                        : group.camera_name} · {group.ended_at
                        ? `${formatTime(group.started_at)} – ${formatTime(group.ended_at)}`
                        : relativeTime(group.started_at)}
                      {group.duration != null && group.duration > 0 && (
                        <span className="ml-1 text-blue-400">({formatDuration(group.duration)})</span>
                      )}
                    </p>
                    <p className="text-xs text-slate-400 italic mt-0.5 truncate">{group.narrative}</p>
                  </div>
                  <ChevronDown
                    size={16}
                    className={`text-slate-500 shrink-0 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                  />
                </button>

                {/* Expanded group detail */}
                {isExpanded && (
                  <div className="border-t border-slate-800">
                    {isSingle ? (
                      /* Single event — show full detail like before */
                      <div className="p-3 space-y-3">
                        {primaryEvent.gif_url ? (
                          <img src={gifUrl(primaryEvent.id)} alt="Timelapse" className="w-full rounded-lg" />
                        ) : primaryEvent.snapshot_path ? (
                          <img src={imgUrl(primaryEvent.id, "snapshot")} alt="Snapshot" className="w-full rounded-lg" />
                        ) : null}
                        {primaryEvent.narrative && (
                          <p className="text-sm text-slate-300 italic">{primaryEvent.narrative}</p>
                        )}
                        <p className="text-xs text-slate-500">
                          {new Date(primaryEvent.started_at).toLocaleString()}
                        </p>
                        <MemberActions ev={primaryEvent} />
                      </div>
                    ) : (
                      /* Multi-event group — show member cards */
                      <div className="p-3 space-y-2">
                        <p className="text-sm text-slate-300 italic mb-2">{group.narrative}</p>
                        {group.events.map((ev) => {
                          const isMemberExpanded = expandedMemberId === ev.id;
                          return (
                            <div key={ev.id} className="bg-slate-800/50 rounded-lg overflow-hidden">
                              {/* Member row */}
                              <button
                                className="w-full flex items-center gap-2.5 p-2.5 text-left"
                                onClick={() => {
                                  const newId = isMemberExpanded ? null : ev.id;
                                  setExpandedMemberId(newId);
                                  if (newId && !ev.named_object_name && ev.object_type === "person") {
                                    fetchSuggestions(ev.id);
                                  }
                                }}
                              >
                                {(ev.thumbnail_path || ev.snapshot_path) ? (
                                  <img
                                    src={imgUrl(ev.id, "crop")}
                                    alt=""
                                    loading="lazy"
                                    className="w-14 h-10 rounded object-contain bg-slate-900 shrink-0"
                                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                                  />
                                ) : (
                                  <div className="w-14 h-10 rounded bg-slate-900 shrink-0 flex items-center justify-center">
                                    <Camera size={14} className="text-slate-600" />
                                  </div>
                                )}
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium truncate">
                                    {eventMemberLabel(ev)}
                                    {ev.confidence != null && (
                                      <span className="text-xs text-slate-500 ml-1">
                                        ({(ev.confidence * 100).toFixed(0)}%)
                                      </span>
                                    )}
                                  </p>
                                  <p className="text-[10px] text-slate-500">
                                    {ev.camera_name && group.camera_names && group.camera_names.length > 1 && (
                                      <span className="text-blue-400 mr-1">{ev.camera_name}</span>
                                    )}
                                    {formatTime(ev.started_at)}
                                    {ev.duration != null && ` · ${formatDuration(ev.duration)}`}
                                  </p>
                                </div>
                                <ChevronDown
                                  size={14}
                                  className={`text-slate-500 shrink-0 transition-transform ${isMemberExpanded ? "rotate-180" : ""}`}
                                />
                              </button>
                              {/* Expanded member detail */}
                              {isMemberExpanded && (
                                <div className="border-t border-slate-700 p-2.5 space-y-2">
                                  {ev.snapshot_path && (
                                    <img src={imgUrl(ev.id, "snapshot")} alt="Snapshot" className="w-full rounded-lg" />
                                  )}
                                  {ev.narrative && (
                                    <p className="text-xs text-slate-400 italic">{ev.narrative}</p>
                                  )}
                                  <MemberActions ev={ev} />
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination */}
      {data && data.total_groups > data.page_size && (() => {
        const totalPages = Math.ceil(data.total_groups / data.page_size);
        return (
          <div className="flex items-center justify-center gap-2">
            <button className="btn-secondary text-sm py-1.5" disabled={page === 1} onClick={() => setPage(page - 1)}>Prev</button>
            <span className="text-sm text-slate-400">{page} / {totalPages}</span>
            <button className="btn-secondary text-sm py-1.5" disabled={page === totalPages} onClick={() => setPage(page + 1)}>Next</button>
          </div>
        );
      })()}
    </div>
  );
}
