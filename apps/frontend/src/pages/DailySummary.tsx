import { useState, type ReactNode } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, getToken } from "../api";
import {
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  Users,
  PawPrint,
  Car,
  Activity,
  RefreshCw,
  Calendar,
  Sparkles,
  Zap,
  Clock,
  Camera,
  Eye,
} from "lucide-react";

interface SnapshotEntry {
  event_id: number;
  time: string;
  camera: string;
  description: string;
}

interface PersonEntry {
  name: string;
  first_seen: string;
  last_seen: string;
  locations: string[];
  event_count: number;
  snapshot_event_id: number | null;
  species?: string;
  snapshots?: SnapshotEntry[];
  by_period?: Record<string, number>;
}

interface TimelineEntry {
  hour: number;
  label: string;
  counts: Record<string, number>;
}

interface ActivityPeriod {
  period: string;
  label: string;
  total_events: number;
  people: string[];
  pets: string[];
}

interface SummaryData {
  greeting: string;
  date: string;
  summary_type: string;
  total_events: number;
  total_people_events: number;
  total_pet_events: number;
  people: PersonEntry[];
  pets: PersonEntry[];
  vehicles: PersonEntry[];
  unknown_counts: Record<string, number>;
  timeline: TimelineEntry[];
  activity_periods?: ActivityPeriod[];
  narrative: string;
  narrative_source?: "local" | "deep" | "fallback";
}

interface SummaryRecord {
  id: number;
  date: string;
  summary_type: string;
  data: SummaryData;
  generated_at: string;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + "T12:00:00");
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  if (dateStr === today.toISOString().split("T")[0]) return "Today";
  if (dateStr === yesterday.toISOString().split("T")[0]) return "Yesterday";
  return d.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" });
}

/** Safely render markdown-style **bold** text as React elements (no dangerouslySetInnerHTML). */
function renderNarrative(text: string): ReactNode[] {
  return text.split("\n\n").map((para, pi) => {
    const parts: ReactNode[] = [];
    const lines = para.split("\n");
    lines.forEach((line, li) => {
      if (li > 0) parts.push(<br key={`br-${pi}-${li}`} />);
      // Split on **bold** markers
      const segments = line.split(/(\*\*[^*]+\*\*)/g);
      segments.forEach((seg, si) => {
        const boldMatch = seg.match(/^\*\*(.+)\*\*$/);
        if (boldMatch) {
          parts.push(<strong key={`b-${pi}-${li}-${si}`} className="text-white">{boldMatch[1]}</strong>);
        } else if (seg) {
          parts.push(seg);
        }
      });
    });
    return <p key={pi}>{parts}</p>;
  });
}

function StatCard({ icon: Icon, label, value, color }: {
  icon: typeof Users;
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className={`flex items-center gap-3 bg-slate-800/60 rounded-2xl p-4 border border-slate-700/50`}>
      <div className={`p-2.5 rounded-xl ${color}`}>
        <Icon size={20} className="text-white" />
      </div>
      <div>
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-xs text-slate-400">{label}</div>
      </div>
    </div>
  );
}

function PersonCard({ entry, token }: { entry: PersonEntry; token: string | null }) {
  const [expanded, setExpanded] = useState(false);
  const hasSnapshots = entry.snapshots && entry.snapshots.length > 0;
  const activePeriods = entry.by_period
    ? Object.entries(entry.by_period).filter(([, v]) => v > 0).map(([k, v]) => ({ period: k, count: v }))
    : [];

  return (
    <div className="bg-slate-800/40 rounded-xl border border-slate-700/30 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 p-3 text-left"
      >
        {entry.snapshot_event_id ? (
          <img
            src={`/api/events/${entry.snapshot_event_id}/thumbnail?token=${encodeURIComponent(token || "")}`}
            alt={entry.name}
            className="w-12 h-12 rounded-full object-cover border-2 border-slate-600 shrink-0"
          />
        ) : (
          <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center text-lg shrink-0">
            {entry.name[0]}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="font-medium text-white truncate">{entry.name}</div>
          <div className="text-xs text-slate-400">
            {entry.first_seen} — {entry.last_seen} · {entry.locations.join(", ")}
          </div>
          {activePeriods.length > 0 && (
            <div className="flex gap-1.5 mt-1">
              {activePeriods.map(({ period, count }) => (
                <span key={period} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-400">
                  {period} ×{count}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="text-right shrink-0">
          <div className="text-lg font-semibold text-blue-400">{entry.event_count}</div>
          <div className="text-[10px] text-slate-500">events</div>
          {hasSnapshots && (
            <Camera size={12} className={`mx-auto mt-1 transition-transform ${expanded ? "text-blue-400" : "text-slate-600"}`} />
          )}
        </div>
      </button>

      {/* Expanded snapshots with AI descriptions */}
      {expanded && hasSnapshots && (
        <div className="border-t border-slate-700/30 p-3 space-y-3">
          {entry.snapshots!.map((snap, i) => (
            <div key={i} className="flex gap-3">
              <img
                src={`/api/events/${snap.event_id}/snapshot?token=${encodeURIComponent(token || "")}`}
                alt={`${entry.name} at ${snap.time}`}
                className="w-28 h-20 rounded-lg object-contain border border-slate-600 shrink-0"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 text-[10px] text-slate-500 mb-1">
                  <Clock size={10} /> {snap.time} · {snap.camera}
                </div>
                {snap.description ? (
                  <p className="text-xs text-slate-300 leading-relaxed">
                    <Eye size={10} className="inline mr-1 text-blue-400" />
                    {snap.description}
                  </p>
                ) : (
                  <p className="text-xs text-slate-500 italic">No AI description available</p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ActivityTimeline({ timeline }: { timeline: TimelineEntry[] }) {
  if (!timeline.length) return null;

  const maxCount = Math.max(...timeline.map((t) => {
    return Object.values(t.counts).reduce((a, b) => a + b, 0);
  }), 1);

  return (
    <div className="bg-slate-800/40 rounded-2xl p-4 border border-slate-700/30">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <Activity size={14} /> Activity Timeline
      </h3>
      <div className="flex items-end gap-1 h-24">
        {Array.from({ length: 24 }, (_, hour) => {
          const entry = timeline.find((t) => t.hour === hour);
          const total = entry
            ? Object.values(entry.counts).reduce((a, b) => a + b, 0)
            : 0;
          const height = total > 0 ? Math.max(8, (total / maxCount) * 100) : 2;
          const people = entry?.counts?.person || 0;
          const pets = (entry?.counts?.cat || 0) + (entry?.counts?.dog || 0);

          return (
            <div key={hour} className="flex-1 flex flex-col items-center gap-0.5" title={`${hour}:00 — ${total} events`}>
              <div
                className="w-full rounded-t transition-all"
                style={{
                  height: `${height}%`,
                  background: total === 0
                    ? "rgb(51 65 85 / 0.5)"
                    : people > 0 && pets > 0
                    ? "linear-gradient(to top, #3b82f6, #a855f7)"
                    : people > 0
                    ? "#3b82f6"
                    : pets > 0
                    ? "#a855f7"
                    : "#64748b",
                }}
              />
              {hour % 6 === 0 && (
                <span className="text-[9px] text-slate-500">{hour}</span>
              )}
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-4 mt-2 text-[10px] text-slate-500">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" /> People</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500" /> Pets</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-slate-500" /> Other</span>
      </div>
    </div>
  );
}

export default function DailySummary() {
  const qc = useQueryClient();
  const token = getToken();

  // Date navigation
  const today = new Date().toISOString().split("T")[0];
  const [selectedDate, setSelectedDate] = useState(today);
  const [showDatePicker, setShowDatePicker] = useState(false);

  // Available dates
  const { data: datesData } = useQuery({
    queryKey: ["summary-dates"],
    queryFn: () => api.get<{ dates: string[] }>("/api/summaries"),
  });

  // Summary for selected date
  const { data: summaryData, isLoading } = useQuery({
    queryKey: ["summary", selectedDate],
    queryFn: () => api.get<{ summaries: SummaryRecord[] }>(`/api/summaries/${selectedDate}`).catch(() => null),
  });

  // Generate summary
  const generateMut = useMutation({
    mutationFn: (type: string) =>
      api.post(`/api/summaries/${selectedDate}/generate?summary_type=${type}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["summary", selectedDate] });
      qc.invalidateQueries({ queryKey: ["summary-dates"] });
    },
  });

  // Deep generate (ml.banusphotos.com)
  const deepMut = useMutation({
    mutationFn: (type: string) =>
      api.post(`/api/summaries/${selectedDate}/generate-deep?summary_type=${type}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["summary", selectedDate] });
      qc.invalidateQueries({ queryKey: ["summary-dates"] });
    },
  });

  const summaries = summaryData?.summaries || [];
  const latestSummary = summaries[0]?.data;

  const navigateDate = (delta: number) => {
    const d = new Date(selectedDate + "T12:00:00");
    d.setDate(d.getDate() + delta);
    const newDate = d.toISOString().split("T")[0];
    if (newDate <= today) setSelectedDate(newDate);
  };

  return (
    <div className="max-w-lg mx-auto px-4 py-4 space-y-4">
      {/* Date Header */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigateDate(-1)}
          className="p-2 rounded-xl bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
        >
          <ChevronLeft size={20} />
        </button>

        <button
          onClick={() => setShowDatePicker(!showDatePicker)}
          className="flex flex-col items-center"
        >
          <span className="text-lg font-bold text-white">{formatDate(selectedDate)}</span>
          <span className="text-xs text-slate-400">{selectedDate}</span>
        </button>

        <button
          onClick={() => navigateDate(1)}
          disabled={selectedDate >= today}
          className="p-2 rounded-xl bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition-colors disabled:opacity-30"
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* Date picker dropdown */}
      {showDatePicker && datesData?.dates && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-2 max-h-48 overflow-y-auto">
          {datesData.dates.map((d) => (
            <button
              key={d}
              onClick={() => { setSelectedDate(d); setShowDatePicker(false); }}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                d === selectedDate
                  ? "bg-blue-600 text-white"
                  : "text-slate-300 hover:bg-slate-700"
              }`}
            >
              {formatDate(d)} <span className="text-xs text-slate-400 ml-1">{d}</span>
            </button>
          ))}
        </div>
      )}

      {/* Summary Type Tabs */}
      {summaries.length > 1 && (
        <div className="flex gap-2">
          {summaries.map((s) => (
            <button
              key={s.summary_type}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm ${
                s === summaries[0]
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              {s.summary_type === "morning" ? <Sun size={14} /> : <Moon size={14} />}
              {s.summary_type === "morning" ? "Morning" : "Evening"}
            </button>
          ))}
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center py-20 text-slate-400">
          <RefreshCw size={20} className="animate-spin mr-2" /> Loading...
        </div>
      ) : latestSummary ? (
        <>
          {/* AI Narrative */}
          {latestSummary.narrative && (
            <div className={`rounded-2xl p-5 border ${
              latestSummary.narrative_source === "deep"
                ? "bg-gradient-to-br from-amber-600/20 to-orange-600/20 border-amber-500/20"
                : "bg-gradient-to-br from-blue-600/20 to-purple-600/20 border-blue-500/20"
            }`}>
              <div className="flex items-center gap-2 mb-3">
                {latestSummary.narrative_source === "deep" ? (
                  <Zap size={16} className="text-amber-400" />
                ) : (
                  <Sparkles size={16} className="text-blue-400" />
                )}
                <span className={`text-xs font-semibold uppercase tracking-wider ${
                  latestSummary.narrative_source === "deep" ? "text-amber-400" : "text-blue-400"
                }`}>
                  {latestSummary.narrative_source === "deep" ? "Deep AI Summary" : "AI Summary"}
                </span>
              </div>
              <div className="text-sm leading-relaxed text-slate-200 space-y-3">
                {renderNarrative(latestSummary.narrative)}
              </div>
              <p className="text-[10px] text-slate-500 mt-3">
                Generated {new Date(summaries[0].generated_at).toLocaleString()}
              </p>
            </div>
          )}

          {/* Key Snapshots Gallery (top snapshots from all entities) */}
          {(() => {
            const allSnaps = [
              ...latestSummary.people,
              ...latestSummary.pets,
            ].flatMap((e) =>
              (e.snapshots || []).filter((s) => s.description).map((s) => ({ ...s, entity: e.name }))
            ).slice(0, 6);
            if (!allSnaps.length) return null;
            return (
              <div>
                <h3 className="text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
                  <Camera size={14} /> Key Moments ({allSnaps.length})
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {allSnaps.map((snap, i) => (
                    <div key={i} className="bg-slate-800/40 rounded-xl overflow-hidden border border-slate-700/30">
                      <img
                        src={`/api/events/${snap.event_id}/snapshot?token=${encodeURIComponent(token || "")}`}
                        alt={`${snap.entity} at ${snap.time}`}
                        className="w-full h-24 object-contain bg-slate-900"
                      />
                      <div className="p-2">
                        <div className="text-[10px] text-slate-500 flex items-center gap-1">
                          <Clock size={9} /> {snap.time} · {snap.camera}
                        </div>
                        <div className="text-[11px] text-slate-300 font-medium mt-0.5">{snap.entity}</div>
                        {snap.description && (
                          <p className="text-[10px] text-slate-400 mt-1 line-clamp-2">{snap.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}

          {/* Fallback greeting when no narrative */}
          {!latestSummary.narrative && latestSummary.greeting && (
            <div className="bg-gradient-to-br from-blue-600/20 to-purple-600/20 rounded-2xl p-5 border border-blue-500/20">
              <p className="text-base text-slate-200">{latestSummary.greeting}</p>
              <p className="text-xs text-slate-400 mt-1">
                Generated {new Date(summaries[0].generated_at).toLocaleString()}
              </p>
            </div>
          )}

          {/* Stats Grid */}
          <div className="grid grid-cols-3 gap-2">
            <StatCard icon={Activity} label="Events" value={latestSummary.total_events} color="bg-slate-600" />
            <StatCard icon={Users} label="People" value={latestSummary.total_people_events} color="bg-blue-600" />
            <StatCard icon={PawPrint} label="Pets" value={latestSummary.total_pet_events} color="bg-purple-600" />
          </div>

          {/* Activity Periods Breakdown */}
          {latestSummary.activity_periods && latestSummary.activity_periods.length > 0 && (
            <div className="bg-slate-800/40 rounded-2xl p-4 border border-slate-700/30">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <Clock size={14} /> Day Breakdown
              </h3>
              <div className="space-y-2">
                {latestSummary.activity_periods.filter(p => p.total_events > 0).map((p) => (
                  <div key={p.period} className="flex items-center gap-3 py-1.5 border-b border-slate-700/20 last:border-0">
                    <span className="text-xs font-medium text-slate-400 w-20">{p.label}</span>
                    <div className="flex-1">
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                          style={{ width: `${Math.min(100, (p.total_events / latestSummary.total_events) * 100)}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-xs text-slate-400 w-8 text-right">{p.total_events}</span>
                    <div className="flex gap-1">
                      {p.people.length > 0 && (
                        <span className="text-[10px] px-1 rounded bg-blue-500/20 text-blue-400">
                          {p.people.length} <Users size={8} className="inline" />
                        </span>
                      )}
                      {p.pets.length > 0 && (
                        <span className="text-[10px] px-1 rounded bg-purple-500/20 text-purple-400">
                          {p.pets.length} <PawPrint size={8} className="inline" />
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* People Section */}
          {latestSummary.people.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
                <Users size={14} /> People ({latestSummary.people.length})
              </h3>
              <div className="space-y-2">
                {latestSummary.people.map((p) => (
                  <PersonCard key={p.name} entry={p} token={token} />
                ))}
              </div>
            </div>
          )}

          {/* Pets Section */}
          {latestSummary.pets.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
                <PawPrint size={14} /> Pets ({latestSummary.pets.length})
              </h3>
              <div className="space-y-2">
                {latestSummary.pets.map((p) => (
                  <PersonCard key={p.name} entry={p} token={token} />
                ))}
              </div>
            </div>
          )}

          {/* Vehicles Section */}
          {latestSummary.vehicles.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
                <Car size={14} /> Vehicles ({latestSummary.vehicles.length})
              </h3>
              <div className="space-y-2">
                {latestSummary.vehicles.map((v) => (
                  <PersonCard key={v.name} entry={v} token={token} />
                ))}
              </div>
            </div>
          )}

          {/* Activity Timeline */}
          <ActivityTimeline timeline={latestSummary.timeline} />

          {/* Action Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => generateMut.mutate(latestSummary.summary_type || "morning")}
              disabled={generateMut.isPending || deepMut.isPending}
              className="flex-1 py-2.5 rounded-xl bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 text-sm transition-colors flex items-center justify-center gap-2"
            >
              <RefreshCw size={14} className={generateMut.isPending ? "animate-spin" : ""} />
              {generateMut.isPending ? "Generating..." : "Regenerate"}
            </button>
            <button
              onClick={() => deepMut.mutate(latestSummary.summary_type || "morning")}
              disabled={generateMut.isPending || deepMut.isPending}
              className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-amber-600/80 to-orange-600/80 text-white hover:from-amber-500 hover:to-orange-500 text-sm transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Zap size={14} className={deepMut.isPending ? "animate-pulse" : ""} />
              {deepMut.isPending ? "Deep generating..." : "Deep Generate"}
            </button>
          </div>
        </>
      ) : (
        /* No summary yet */
        <div className="text-center py-16 space-y-4">
          <Calendar size={48} className="mx-auto text-slate-600" />
          <p className="text-slate-400">No summary for this date yet</p>
          <div className="flex gap-2 justify-center">
            <button
              onClick={() => generateMut.mutate("morning")}
              disabled={generateMut.isPending}
              className="px-4 py-2 rounded-xl bg-blue-600 text-white text-sm hover:bg-blue-500 transition-colors flex items-center gap-2"
            >
              <Sun size={14} />
              {generateMut.isPending ? "Generating..." : "Generate Full Day"}
            </button>
            {selectedDate === today && (
              <button
                onClick={() => generateMut.mutate("evening")}
                disabled={generateMut.isPending}
                className="px-4 py-2 rounded-xl bg-purple-600 text-white text-sm hover:bg-purple-500 transition-colors flex items-center gap-2"
              >
                <Moon size={14} />
                Today So Far
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
