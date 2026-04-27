import { useState, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api, getToken } from "../api";
import {
  MapPin,
  Search as SearchIcon,
  Camera,
  ChevronDown,
  ChevronRight,
  User,
  Cat,
  Car,
  Box,
  Eye,
  X,
  Radio,
  Video,
} from "lucide-react";

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
}

interface TimelineEntry {
  event_id: number;
  camera_id: number;
  camera_name: string;
  timestamp: string;
  confidence: number | null;
  snapshot_url: string | null;
  narrative: string | null;
}

const CATEGORY_ICONS: Record<string, typeof User> = {
  person: User,
  pet: Cat,
  vehicle: Car,
  other: Box,
};

const CATEGORY_ORDER = ["person", "pet", "vehicle", "other"];
const CATEGORY_LABELS: Record<string, string> = {
  person: "People",
  pet: "Pets",
  vehicle: "Vehicles",
  other: "Other",
};

export default function Search() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [previewEvent, setPreviewEvent] = useState<TimelineEntry | null>(null);
  const [liveSnapshots, setLiveSnapshots] = useState<Record<number, string>>({});
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [expandedObject, setExpandedObject] = useState<number | null>(null);

  const { data: objectsStatus } = useQuery({
    queryKey: ["named-objects-status"],
    queryFn: () => api.get<NamedObjectStatus[]>("/api/search/named-objects-status"),
    refetchInterval: 5000,
  });

  // Refresh live snapshots from go2rtc for currently-live objects
  useEffect(() => {
    if (!objectsStatus) return;
    const camIds = new Set<number>();
    for (const o of objectsStatus) {
      if (o.is_live && o.live_camera_id) camIds.add(o.live_camera_id);
    }
    if (camIds.size === 0) return;

    let cancelled = false;
    const refreshSnaps = async () => {
      const snaps: Record<number, string> = {};
      for (const camId of camIds) {
        try {
          const r = await fetch(`/go2rtc/api/frame.jpeg?src=camera_${camId}&_=${Date.now()}`);
          if (r.ok) {
            const blob = await r.blob();
            snaps[camId] = URL.createObjectURL(blob);
          }
        } catch { /* ignore */ }
      }
      if (!cancelled) setLiveSnapshots((prev) => {
        Object.values(prev).forEach((u) => URL.revokeObjectURL(u));
        return snaps;
      });
    };
    refreshSnaps();
    const interval = setInterval(refreshSnaps, 5000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [objectsStatus]);

  // Timeline for the inline-expanded object
  const expandedObj = expandedObject != null
    ? objectsStatus?.find((o) => o.id === expandedObject)
    : undefined;

  const { data: timeline, isLoading: loadingTimeline } = useQuery({
    queryKey: ["timeline-name", expandedObj?.name],
    queryFn: () => api.get<TimelineEntry[]>(
      `/api/search/timeline?object_name=${encodeURIComponent(expandedObj!.name)}&limit=50`
    ),
    enabled: !!expandedObj?.name,
    retry: false,
  });

  const token = getToken();
  const imgUrl = (url: string) => `${url}?token=${encodeURIComponent(token || "")}`;

  // Filter + group objects by category
  const grouped = useMemo(() => {
    if (!objectsStatus) return {} as Record<string, NamedObjectStatus[]>;
    const q = query.trim().toLowerCase();
    const filtered = q
      ? objectsStatus.filter((o) => o.name.toLowerCase().includes(q))
      : objectsStatus;
    const out: Record<string, NamedObjectStatus[]> = {};
    for (const o of filtered) {
      const cat = CATEGORY_ORDER.includes(o.category) ? o.category : "other";
      (out[cat] ||= []).push(o);
    }
    // Sort each group: live first, then by name
    for (const k of Object.keys(out)) {
      out[k].sort((a, b) => {
        if (a.is_live !== b.is_live) return a.is_live ? -1 : 1;
        return a.name.localeCompare(b.name);
      });
    }
    return out;
  }, [objectsStatus, query]);

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

  const toggleCategory = (cat: string) => setCollapsed((c) => ({ ...c, [cat]: !c[cat] }));

  const totalCount = objectsStatus?.length ?? 0;
  const visibleCount = Object.values(grouped).reduce((s, arr) => s + arr.length, 0);

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-bold">Discover</h2>
        <span className="text-xs text-slate-500">
          {visibleCount} of {totalCount}
        </span>
      </div>

      {/* Search box */}
      <div className="relative">
        <SearchIcon size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          className="input pl-10"
          placeholder="Search by name…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {query && (
          <button
            onClick={() => setQuery("")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Empty state */}
      {objectsStatus && objectsStatus.length === 0 && (
        <div className="card text-center py-12 text-slate-400">
          <Eye size={48} className="mx-auto mb-3 opacity-50" />
          <p>No named objects yet</p>
          <p className="text-xs mt-1">Train objects from the Profiles page to track them here</p>
        </div>
      )}

      {/* Grouped accordions */}
      {CATEGORY_ORDER.map((cat) => {
        const items = grouped[cat];
        if (!items || items.length === 0) return null;
        const Icon = CATEGORY_ICONS[cat] || Box;
        const isCollapsed = collapsed[cat];
        const liveCount = items.filter((o) => o.is_live).length;

        return (
          <div key={cat} className="card p-0 overflow-hidden">
            <button
              onClick={() => toggleCategory(cat)}
              className="w-full flex items-center gap-2 px-3 py-3 hover:bg-slate-800/40 transition-colors"
            >
              {isCollapsed
                ? <ChevronRight size={16} className="text-slate-400" />
                : <ChevronDown size={16} className="text-slate-400" />}
              <Icon size={16} className="text-blue-400" />
              <span className="font-semibold text-sm">{CATEGORY_LABELS[cat]}</span>
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
                    const liveSnap = obj.is_live && obj.live_camera_id
                      ? liveSnapshots[obj.live_camera_id] : null;
                    const imageUrl = liveSnap
                      || (obj.thumbnail_url ? imgUrl(obj.thumbnail_url) : null)
                      || (obj.snapshot_url ? imgUrl(obj.snapshot_url) : null);
                    const isExpanded = expandedObject === obj.id;

                    return (
                      <div key={obj.id} className="space-y-1.5">
                        <button
                          onClick={() => setExpandedObject(isExpanded ? null : obj.id)}
                          className={`w-full card p-0 overflow-hidden text-left transition-colors ${
                            isExpanded ? "border-blue-500" : "hover:border-blue-500/50"
                          }`}
                        >
                          <div className="aspect-[3/4] bg-slate-800 relative">
                            {imageUrl ? (
                              <img src={imageUrl} alt={obj.name} className="w-full h-full object-contain" loading="lazy" />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center">
                                <Icon size={32} className="text-slate-600" />
                              </div>
                            )}
                            {obj.is_live && (
                              <div className="absolute top-1.5 left-1.5 flex items-center gap-1 bg-red-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider">
                                <Radio size={10} className="animate-pulse" /> Live
                              </div>
                            )}
                          </div>
                          <div className="p-2">
                            <p className="font-semibold text-sm truncate">{obj.name}</p>
                            <div className="flex items-center gap-1 mt-0.5 text-[11px] text-slate-400">
                              {obj.is_live ? (
                                <>
                                  <Video size={11} className="text-green-400 shrink-0" />
                                  <span className="text-green-400 truncate">{obj.live_camera_name}</span>
                                </>
                              ) : obj.last_camera_name ? (
                                <>
                                  <MapPin size={11} className="shrink-0" />
                                  <span className="truncate">{obj.last_camera_name}</span>
                                  <span className="text-slate-500 shrink-0 ml-auto">
                                    {obj.last_seen_at ? formatTimeAgo(obj.last_seen_at) : ""}
                                  </span>
                                </>
                              ) : (
                                <span className="text-slate-500 italic">Not seen yet</span>
                              )}
                            </div>
                          </div>
                        </button>
                      </div>
                    );
                  })}
                </div>

                {/* Inline expanded timeline (full-width below grid) */}
                {expandedObj && items.some((o) => o.id === expandedObj.id) && (
                  <div className="mt-3 card bg-slate-900/60 space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-sm">
                        <span className="text-blue-400">{expandedObj.name}</span> recent activity
                      </h3>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => navigate(`/profiles/${expandedObj.id}`)}
                          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                        >
                          View Profile <ChevronRight size={12} />
                        </button>
                        <button
                          onClick={() => setExpandedObject(null)}
                          className="text-slate-400 hover:text-white"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    </div>
                    {loadingTimeline ? (
                      <p className="text-xs text-slate-400">Loading…</p>
                    ) : !timeline || timeline.length === 0 ? (
                      <p className="text-xs text-slate-500 italic">No recent activity</p>
                    ) : (
                      <div className="space-y-1.5 max-h-[40vh] overflow-y-auto">
                        {timeline.map((entry) => {
                          const date = new Date(entry.timestamp);
                          return (
                            <div
                              key={entry.event_id}
                              className="flex items-center gap-2 p-1.5 rounded hover:bg-slate-800/50 transition-colors"
                            >
                              {entry.snapshot_url ? (
                                <button
                                  onClick={() => setPreviewEvent(entry)}
                                  className="w-12 h-9 rounded overflow-hidden bg-slate-800 shrink-0"
                                >
                                  <img src={imgUrl(entry.snapshot_url)} alt="" className="w-full h-full object-contain" loading="lazy" />
                                </button>
                              ) : (
                                <div className="w-12 h-9 rounded bg-slate-800 shrink-0 flex items-center justify-center">
                                  <Camera size={12} className="text-slate-600" />
                                </div>
                              )}
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-medium truncate">{entry.camera_name}</p>
                                {entry.narrative && (
                                  <p className="text-[10px] text-slate-400 italic truncate">{entry.narrative}</p>
                                )}
                              </div>
                              <div className="text-right shrink-0">
                                <p className="text-xs font-medium">
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
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}

      {/* No matches */}
      {objectsStatus && objectsStatus.length > 0 && visibleCount === 0 && (
        <div className="card text-center py-8 text-slate-400 text-sm">
          No matches for "{query}"
        </div>
      )}

      {/* Snapshot Preview Modal */}
      {previewEvent && (
        <div
          className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-4"
          onClick={() => setPreviewEvent(null)}
        >
          <div className="max-w-lg w-full space-y-2" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <p className="text-sm text-slate-300">
                {previewEvent.camera_name} · {new Date(previewEvent.timestamp).toLocaleString()}
              </p>
              <button
                onClick={() => setPreviewEvent(null)}
                className="text-slate-400 hover:text-white"
              >
                <X size={20} />
              </button>
            </div>
            {previewEvent.snapshot_url && (
              <img src={imgUrl(previewEvent.snapshot_url)} alt="Snapshot" className="w-full rounded-lg" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
