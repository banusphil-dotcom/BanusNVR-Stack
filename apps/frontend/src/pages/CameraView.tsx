import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useParams, useSearchParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, getToken } from "../api";
import Hls from "hls.js";
import CameraStream from "../components/CameraStream";
import { ArrowLeft, Radio, Calendar, Download, SkipBack, SkipForward, Play, ZoomIn, ZoomOut, User, Cat, Car, Box, Save, SlidersHorizontal, Camera as CameraIcon, Eye, GraduationCap, Sparkles, BarChart3, List, Maximize2, Minimize2 } from "lucide-react";

interface ObjectSettings {
  confidence?: number;
  min_area?: number;
}

interface DetectionData {
  detection_objects: string[];
  detection_confidence: number;
  detection_settings: Record<string, ObjectSettings>;
}

interface CameraInfo {
  id: number;
  name: string;
  camera_type: string;
  enabled: boolean;
}

interface TimelineHour {
  hour: number;
  has_recordings: boolean;
  segments: number;
}

interface TimelineResponse {
  camera_id: number;
  date: string;
  hours: TimelineHour[];
}

interface TimelineEvent {
  id: number;
  time: string;
  event_type: string;
  object_type: string | null;
  confidence: number | null;
  named_object_id: number | null;
  named_object_name: string | null;
  motion_score?: number | null;
  thumbnail_url: string | null;
}

interface PresenceBar {
  start: string;
  end: string;
  event_id: number;
  cameras: number[];
}

interface PresenceRow {
  named_object_id: number;
  name: string;
  category: string;
  bars: PresenceBar[];
}

export default function CameraView() {
  const { id } = useParams<{ id: string }>();
  const cameraId = Number(id);
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const paramTime = searchParams.get("time");
  const qc = useQueryClient();

  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);

  // Detection settings panel state
  const [showDetSettings, setShowDetSettings] = useState(false);
  // View mode: events list vs presence timeline
  const [viewMode, setViewMode] = useState<"events" | "timeline">("events");

  const [isLive, setIsLive] = useState(!paramTime);
  const [date, setDate] = useState(() => {
    if (paramTime) {
      const d = new Date(paramTime);
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    }
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  });
  const [scrubMinutes, setScrubMinutes] = useState<number>(() => {
    if (paramTime) {
      const d = new Date(paramTime);
      return d.getHours() * 60 + d.getMinutes();
    }
    const n = new Date();
    return n.getHours() * 60 + n.getMinutes();
  });

  // Camera info
  const { data: camera } = useQuery({
    queryKey: ["camera", cameraId],
    queryFn: async () => {
      const cameras = await api.get<CameraInfo[]>("/api/cameras");
      return cameras.find((c) => c.id === cameraId) ?? null;
    },
  });

  // Recording timeline (always fetched for the timeline bar)
  const { data: timeline } = useQuery({
    queryKey: ["rec-timeline", cameraId, date],
    queryFn: () => api.get<TimelineResponse>(`/api/recordings/${cameraId}/timeline?date=${date}`),
  });

  // Events for timeline markers
  const { data: timelineEvents } = useQuery({
    queryKey: ["events-timeline", cameraId, date],
    queryFn: () => api.get<TimelineEvent[]>(`/api/events/camera-timeline?camera_id=${cameraId}&date=${date}`),
  });

  // Presence bars for timeline view
  const { data: presenceData } = useQuery({
    queryKey: ["presence-timeline", cameraId, date],
    queryFn: () => api.get<PresenceRow[]>(`/api/events/presence-timeline?camera_id=${cameraId}&date=${date}`),
    enabled: viewMode === "timeline",
  });

  // Hours with recordings for the bar — convert UTC directory hours to local
  const recordingHours = useMemo(() => {
    if (!timeline?.hours) return new Set<number>();
    return new Set(timeline.hours
      .filter((h) => h.has_recordings)
      .map((h) => {
        const utcDate = new Date(`${date}T${String(h.hour).padStart(2, "0")}:30:00Z`);
        return utcDate.getHours();
      })
    );
  }, [timeline, date]);

  const destroyHls = useCallback(() => {
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }
  }, []);

  const startPlayback = useCallback(
    (mins: number, seekSeconds?: number) => {
      const video = videoRef.current;
      if (!video) return;
      destroyHls();

      const h = Math.floor(mins / 60);
      const m = mins % 60;
      const startDate = new Date(`${date}T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00`);
      const endDate = new Date(startDate.getTime() + 5 * 60 * 1000);
      const url = `/api/recordings/${cameraId}/playlist.m3u8?start=${encodeURIComponent(startDate.toISOString())}&end=${encodeURIComponent(endDate.toISOString())}`;

      if (Hls.isSupported()) {
        const hls = new Hls({
          maxBufferLength: 5,
          maxMaxBufferLength: 30,
          startFragPrefetch: true,
          xhrSetup: (xhr) => {
            const t = getToken();
            if (t) xhr.setRequestHeader("Authorization", `Bearer ${t}`);
          },
        });
        hlsRef.current = hls;
        hls.loadSource(url);
        hls.attachMedia(video);
        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          if (seekSeconds && seekSeconds > 0) {
            video.currentTime = seekSeconds;
          }
          video.play().catch(() => {});
        });
        hls.on(Hls.Events.ERROR, (_event, data) => {
          if (data.fatal) {
            if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
              hls.startLoad();
            } else {
              hls.destroy();
              hlsRef.current = null;
            }
          }
        });
      } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = url;
        video.addEventListener("loadedmetadata", () => {
          if (seekSeconds && seekSeconds > 0) {
            video.currentTime = seekSeconds;
          }
          video.play().catch(() => {});
        }, { once: true });
      }
    },
    [cameraId, date, destroyHls],
  );

  // Auto-start playback if navigated with ?time=
  useEffect(() => {
    if (paramTime && !isLive) {
      const eventDate = new Date(paramTime);
      const eventSeconds = eventDate.getSeconds();
      const t = setTimeout(() => startPlayback(scrubMinutes, eventSeconds), 150);
      return () => clearTimeout(t);
    }
  }, [paramTime]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => () => destroyHls(), [destroyHls]);

  const goLive = () => {
    destroyHls();
    setIsLive(true);
  };

  const scrubTo = (mins: number, seekSeconds?: number) => {
    const clamped = Math.max(0, Math.min(1439, mins));
    setScrubMinutes(clamped);
    setIsLive(false);
    startPlayback(clamped, seekSeconds);
  };

  const handleTimelineTouchEnd = () => {
    setIsLive(false);
    startPlayback(scrubMinutes);
  };

  const exportClip = async () => {
    const h = Math.floor(scrubMinutes / 60);
    const m = scrubMinutes % 60;
    const startDate = new Date(`${date}T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00`);
    const endDate = new Date(startDate.getTime() + 5 * 60 * 1000);
    const res = await fetch(
      `/api/recordings/${cameraId}/export?start=${encodeURIComponent(startDate.toISOString())}&end=${encodeURIComponent(endDate.toISOString())}`,
      { method: "POST", headers: { Authorization: `Bearer ${localStorage.getItem("banusnas_token")}` } },
    );
    if (res.ok) {
      const blob = await res.blob();
      const u = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = u;
      a.download = `clip_${cameraId}_${date}.mp4`;
      a.click();
      URL.revokeObjectURL(u);
    }
  };

  const getEventColor = (ev: TimelineEvent) => {
    if (ev.named_object_name) return "#a855f7"; // purple — recognized
    switch (ev.object_type) {
      case "person": return "#22c55e";
      case "car": case "truck": case "bus": case "motorcycle": return "#3b82f6";
      case "cat": case "dog": case "bird": return "#f97316";
      default: return "#6b7280";
    }
  };

  const getEventIcon = (ev: TimelineEvent) => {
    if (ev.named_object_name) return User;
    switch (ev.object_type) {
      case "person": return User;
      case "cat": case "dog": case "bird": return Cat;
      case "car": case "truck": case "bus": case "motorcycle": return Car;
      default: return Box;
    }
  };

  const getEventMinutes = (ev: TimelineEvent) => {
    const d = new Date(ev.time);
    return d.getHours() * 60 + d.getMinutes();
  };

  // Object events for skip navigation
  const objectEvents = useMemo(() => {
    if (!timelineEvents) return [];
    return timelineEvents;
  }, [timelineEvents]);

  const getEventSeconds = (ev: TimelineEvent) => {
    return new Date(ev.time).getSeconds();
  };

  // Skip to prev/next event
  const skipPrev = () => {
    if (!objectEvents.length) return;
    const cur = scrubMinutes;
    for (let i = objectEvents.length - 1; i >= 0; i--) {
      const mins = getEventMinutes(objectEvents[i]);
      if (mins < cur - 0.5) { scrubTo(mins, getEventSeconds(objectEvents[i])); return; }
    }
    const last = objectEvents[objectEvents.length - 1];
    scrubTo(getEventMinutes(last), getEventSeconds(last));
  };

  const skipNext = () => {
    if (!objectEvents.length) return;
    const cur = scrubMinutes;
    for (const ev of objectEvents) {
      const mins = getEventMinutes(ev);
      if (mins > cur + 0.5) { scrubTo(mins, getEventSeconds(ev)); return; }
    }
    const first = objectEvents[0];
    scrubTo(getEventMinutes(first), getEventSeconds(first));
  };

  // Zoom state: 1 = full day, 2 = 12h, 4 = 6h, 8 = 3h, 16 = 1.5h
  const [zoomLevel, setZoomLevel] = useState(1);
  const [zoomCenter, setZoomCenter] = useState(scrubMinutes);

  const zoomIn = () => setZoomLevel((z) => Math.min(z * 2, 16));
  const zoomOut = () => setZoomLevel((z) => Math.max(z / 2, 1));

  // Visible window in minutes
  const windowMinutes = 1440 / zoomLevel;
  const halfWindow = windowMinutes / 2;
  const viewStart = Math.max(0, Math.min(1440 - windowMinutes, zoomCenter - halfWindow));
  const viewEnd = viewStart + windowMinutes;

  // Keep zoom centered on scrub position
  useEffect(() => {
    setZoomCenter(scrubMinutes);
  }, [scrubMinutes]);

  // Vertical timeline: position = percentage from top
  const getPositionInView = (minutes: number) => {
    return ((minutes - viewStart) / windowMinutes) * 100;
  };

  // Click on vertical timeline bar
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = (e.clientY - rect.top) / rect.height;
    const mins = viewStart + fraction * windowMinutes;
    scrubTo(Math.round(Math.max(0, Math.min(1439, mins))));
  };

  // Touch-drag on vertical timeline
  const handleTimelineTouch = (e: React.TouchEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (e.touches[0].clientY - rect.top) / rect.height));
    const mins = viewStart + fraction * windowMinutes;
    setScrubMinutes(Math.round(Math.max(0, Math.min(1439, mins))));
  };

  // Pinch-to-zoom on timeline
  const lastPinchDist = useRef<number | null>(null);
  const handleTimelinePinch = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length !== 2) { lastPinchDist.current = null; return; }
    e.preventDefault();
    const dy = Math.abs(e.touches[0].clientY - e.touches[1].clientY);
    const dx = Math.abs(e.touches[0].clientX - e.touches[1].clientX);
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (lastPinchDist.current !== null) {
      const delta = dist - lastPinchDist.current;
      if (Math.abs(delta) > 8) {
        setZoomLevel((z) => {
          const next = delta > 0 ? z * 1.5 : z / 1.5;
          return Math.max(1, Math.min(16, next));
        });
        lastPinchDist.current = dist;
      }
    } else {
      lastPinchDist.current = dist;
    }
  }, []);
  const handleTimelinePinchEnd = useCallback(() => { lastPinchDist.current = null; }, []);

  // Pinch-to-zoom on video
  const videoWrapRef = useRef<HTMLDivElement>(null);
  const [videoScale, setVideoScale] = useState(1);
  const [videoTranslate, setVideoTranslate] = useState({ x: 0, y: 0 });
  const videoPinchRef = useRef<{ dist: number; scale: number } | null>(null);
  const videoPanRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  // Fullscreen toggle
  const [isFullscreen, setIsFullscreen] = useState(false);
  useEffect(() => {
    const onFs = () => setIsFullscreen(document.fullscreenElement === videoWrapRef.current);
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else if (videoWrapRef.current) {
      videoWrapRef.current.requestFullscreen().catch(() => {});
    }
  }, []);

  const handleVideoTouchStart = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      videoPinchRef.current = { dist, scale: videoScale };
    } else if (e.touches.length === 1 && videoScale > 1) {
      videoPanRef.current = {
        x: e.touches[0].clientX, y: e.touches[0].clientY,
        tx: videoTranslate.x, ty: videoTranslate.y,
      };
    }
  }, [videoScale, videoTranslate]);

  const handleVideoTouchMove = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length === 2 && videoPinchRef.current) {
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const newScale = Math.max(1, Math.min(5, videoPinchRef.current.scale * (dist / videoPinchRef.current.dist)));
      setVideoScale(newScale);
      if (newScale <= 1) setVideoTranslate({ x: 0, y: 0 });
    } else if (e.touches.length === 1 && videoPanRef.current && videoScale > 1) {
      const dx = e.touches[0].clientX - videoPanRef.current.x;
      const dy = e.touches[0].clientY - videoPanRef.current.y;
      setVideoTranslate({
        x: videoPanRef.current.tx + dx,
        y: videoPanRef.current.ty + dy,
      });
    }
  }, [videoScale]);

  const handleVideoTouchEnd = useCallback(() => {
    videoPinchRef.current = null;
    videoPanRef.current = null;
  }, []);

  // Double-tap to reset zoom
  const lastTapRef = useRef(0);
  const handleVideoDoubleTap = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length !== 1) return;
    const now = Date.now();
    if (now - lastTapRef.current < 300) {
      setVideoScale(1);
      setVideoTranslate({ x: 0, y: 0 });
    }
    lastTapRef.current = now;
  }, []);

  const [hoveredEvent, setHoveredEvent] = useState<TimelineEvent | null>(null);
  const token = getToken();

  const h = Math.floor(scrubMinutes / 60);
  const m = scrubMinutes % 60;
  const timeLabel = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;

  const todayStr = useMemo(() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  }, []);

  // Generate hour labels visible in current zoom window
  const visibleHourLabels = useMemo(() => {
    const step = zoomLevel >= 8 ? 1 : zoomLevel >= 4 ? 2 : zoomLevel >= 2 ? 3 : 3;
    const labels: number[] = [];
    for (let hr = Math.ceil(viewStart / 60); hr <= Math.floor(viewEnd / 60); hr++) {
      if (hr % step === 0 && hr >= 0 && hr <= 23) labels.push(hr);
    }
    return labels;
  }, [viewStart, viewEnd, zoomLevel]);

  // Visible non-motion events for the vertical timeline list
  const visibleObjectEvents = useMemo(() => {
    if (!timelineEvents) return [];
    return timelineEvents
      .filter((ev) => {
        const mins = getEventMinutes(ev);
        return mins >= viewStart && mins <= viewEnd;
      })
      .sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime());
  }, [timelineEvents, viewStart, viewEnd]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 shrink-0">
        <button onClick={() => navigate(-1)} className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors">
          <ArrowLeft size={20} />
        </button>
        <h2 className="text-sm font-semibold flex-1 truncate">{camera?.name ?? `Camera ${cameraId}`}</h2>
        <button
          onClick={goLive}
          className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-bold transition-colors ${
            isLive ? "bg-red-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
          }`}
        >
          <Radio size={12} /> LIVE
        </button>
      </div>

      {/* Video area with pinch-to-zoom */}
      <div
        ref={videoWrapRef}
        className="relative bg-black aspect-video shrink-0 overflow-hidden touch-none"
        onTouchStart={(e) => { handleVideoDoubleTap(e); handleVideoTouchStart(e); }}
        onTouchMove={handleVideoTouchMove}
        onTouchEnd={handleVideoTouchEnd}
      >
        <div
          style={{
            transform: `scale(${videoScale}) translate(${videoTranslate.x / videoScale}px, ${videoTranslate.y / videoScale}px)`,
            transformOrigin: "center center",
            width: "100%",
            height: "100%",
          }}
        >
          {isLive ? (
            <CameraStream cameraId={cameraId} cameraName={camera?.name ?? ""} className="w-full h-full" onSettingsToggle={setShowDetSettings} />
          ) : (
            <video ref={videoRef} controls playsInline className="w-full h-full object-contain" />
          )}
        </div>
        {videoScale > 1 && (
          <div className="absolute top-2 right-2 bg-black/60 text-white text-[10px] px-2 py-0.5 rounded-full">
            {videoScale.toFixed(1)}x
          </div>
        )}
        <button
          onClick={toggleFullscreen}
          className="absolute bottom-2 right-2 p-1.5 bg-black/60 hover:bg-black/90 rounded text-white z-10 transition-colors"
          aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
        >
          {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
        </button>
      </div>

      {/* Controls + Vertical Timeline side-by-side */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left panel: controls + event list */}
        <div className="flex-1 px-3 py-2 space-y-2 overflow-y-auto">
          {/* Date + time + zoom */}
          <div className="flex items-center gap-2">
            <Calendar size={14} className="text-slate-400 shrink-0" />
            <input type="date" className="input text-sm flex-1" value={date} onChange={(e) => setDate(e.target.value)} />
            <span className="text-sm font-mono font-semibold text-blue-400 shrink-0">{timeLabel}</span>
          </div>

          {/* Skip + play controls */}
          {!isLive && (
            <div className="flex gap-1.5">
              <button onClick={skipPrev} disabled={!objectEvents.length} className="btn-secondary flex-1 flex items-center justify-center gap-1 text-xs py-2 disabled:opacity-30">
                <SkipBack size={14} />
              </button>
              <button onClick={() => startPlayback(scrubMinutes)} className="btn-primary flex-1 flex items-center justify-center gap-1 text-xs py-2">
                <Play size={14} /> Play
              </button>
              <button onClick={skipNext} disabled={!objectEvents.length} className="btn-secondary flex-1 flex items-center justify-center gap-1 text-xs py-2 disabled:opacity-30">
                <SkipForward size={14} />
              </button>
              <button onClick={exportClip} className="btn-secondary flex items-center justify-center text-xs py-2 px-2">
                <Download size={14} />
              </button>
            </div>
          )}

          {/* Zoom controls */}
          <div className="flex items-center gap-2">
            <button onClick={zoomOut} disabled={zoomLevel <= 1} className="p-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-30 transition-colors">
              <ZoomOut size={14} />
            </button>
            <div className="flex-1 text-center">
              <span className="text-[10px] text-slate-500">
                {zoomLevel <= 1 ? "24h" : zoomLevel <= 2 ? "12h" : zoomLevel <= 4 ? "6h" : zoomLevel <= 8 ? "3h" : "1.5h"} view
              </span>
            </div>
            <button onClick={zoomIn} disabled={zoomLevel >= 16} className="p-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-30 transition-colors">
              <ZoomIn size={14} />
            </button>
          </div>

          {/* View mode tabs */}
          <div className="flex gap-1">
            <button
              onClick={() => { setViewMode("events"); setShowDetSettings(false); }}
              className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded text-[10px] font-medium transition-colors ${
                viewMode === "events" && !showDetSettings ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"
              }`}
            >
              <List size={12} /> Events
            </button>
            <button
              onClick={() => { setViewMode("timeline"); setShowDetSettings(false); }}
              className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded text-[10px] font-medium transition-colors ${
                viewMode === "timeline" ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"
              }`}
            >
              <BarChart3 size={12} /> Timeline
            </button>
          </div>

          {/* Event list / Detection settings / Presence Timeline (scrollable) */}
          {showDetSettings ? (
            <DetectionSettingsPanel cameraId={cameraId} />
          ) : viewMode === "timeline" ? (
            <PresenceTimelineView rows={presenceData ?? []} date={date} navigate={navigate} />
          ) : (
          <div className="space-y-1">
            {visibleObjectEvents.length === 0 && (
              <p className="text-xs text-slate-600 text-center py-4">No object events in this window</p>
            )}
            {visibleObjectEvents.map((ev) => {
              const Icon = getEventIcon(ev);
              const evTime = new Date(ev.time);
              const evMins = getEventMinutes(ev);
              const isActive = Math.abs(evMins - scrubMinutes) < 1;
              return (
                <button
                  key={ev.id}
                  onClick={() => scrubTo(evMins, new Date(ev.time).getSeconds())}
                  className={`w-full flex items-center gap-2 p-1.5 rounded-lg text-left transition-colors ${
                    isActive ? "bg-blue-600/20 border border-blue-500/40" : "hover:bg-slate-800"
                  }`}
                >
                  {ev.thumbnail_url ? (
                    <img
                      src={`${ev.thumbnail_url}?token=${encodeURIComponent(token || "")}`}
                      alt=""
                      className="w-8 h-8 rounded object-cover shrink-0 bg-slate-800"
                      style={{ borderLeft: `3px solid ${getEventColor(ev)}` }}
                      onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                    />
                  ) : (
                    <div
                      className="w-8 h-8 rounded flex items-center justify-center shrink-0"
                      style={{ backgroundColor: getEventColor(ev) + "30", borderLeft: `3px solid ${getEventColor(ev)}` }}
                    >
                      <Icon size={14} style={{ color: getEventColor(ev) }} />
                    </div>
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="text-[11px] font-medium truncate capitalize">
                      {ev.named_object_name || ev.object_type || ev.event_type}
                    </p>
                    <p className="text-[9px] text-slate-500">
                      {evTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                      {ev.confidence ? ` · ${(ev.confidence * 100).toFixed(0)}%` : ""}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
          )}
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate-500 pt-1">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500 inline-block" /> Person</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500 inline-block" /> Vehicle</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-orange-500 inline-block" /> Animal</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500 inline-block" /> Recognized</span>
          </div>
        </div>

        {/* Right: Vertical Timeline Bar */}
        <div
          className="w-12 shrink-0 relative bg-slate-900 border-l border-slate-700 cursor-pointer select-none"
          onClick={handleTimelineClick}
          onTouchMove={(e) => { handleTimelineTouch(e); handleTimelinePinch(e); }}
          onTouchEnd={(e) => { handleTimelineTouchEnd(); handleTimelinePinchEnd(); }}
        >
          {/* Recording blocks */}
          {Array.from({ length: 24 }, (_, hr) => {
            const hrStart = hr * 60;
            const hrEnd = hrStart + 60;
            if (hrEnd < viewStart || hrStart > viewEnd) return null;
            const top = getPositionInView(Math.max(hrStart, viewStart));
            const bottom = getPositionInView(Math.min(hrEnd, viewEnd));
            return (
              <div
                key={hr}
                className={`absolute left-0 right-0 ${recordingHours.has(hr) ? "bg-blue-600/30" : ""} ${
                  hrStart > viewStart ? "border-t border-slate-700/50" : ""
                }`}
                style={{ top: `${top}%`, height: `${bottom - top}%` }}
              />
            );
          })}

          {/* Hour labels */}
          {visibleHourLabels.map((hr) => (
            <span
              key={hr}
              className="absolute left-1 text-[9px] text-slate-600 -translate-y-1/2 pointer-events-none"
              style={{ top: `${getPositionInView(hr * 60)}%` }}
            >
              {String(hr).padStart(2, "0")}
            </span>
          ))}

          {/* Event dots on the timeline */}
          {timelineEvents
            ?.filter((ev) => {
              const mins = getEventMinutes(ev);
              return mins >= viewStart && mins <= viewEnd;
            })
            .map((ev) => {
              const pos = getPositionInView(getEventMinutes(ev));
              return (
                <div
                  key={ev.id}
                  className="absolute right-1 -translate-y-1/2 rounded-full w-3 h-3 border border-slate-900"
                  style={{
                    top: `${pos}%`,
                    backgroundColor: getEventColor(ev),
                  }}
                  title={`${new Date(ev.time).toLocaleTimeString()} — ${ev.named_object_name || ev.object_type || ev.event_type}`}
                />
              );
            })}

          {/* "Now" line */}
          {date === todayStr && (() => {
            const nowMins = new Date().getHours() * 60 + new Date().getMinutes();
            if (nowMins < viewStart || nowMins > viewEnd) return null;
            return (
              <div
                className="absolute left-0 right-0 h-0.5 bg-red-500/60 z-10"
                style={{ top: `${getPositionInView(nowMins)}%` }}
              />
            );
          })()}

          {/* Scrub indicator */}
          {!isLive && scrubMinutes >= viewStart && scrubMinutes <= viewEnd && (
            <div
              className="absolute left-0 right-0 h-0.5 bg-white z-20"
              style={{ top: `${getPositionInView(scrubMinutes)}%` }}
            >
              <div className="absolute top-1/2 right-0 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ═══════════ Presence Timeline View ═══════════ */

const CATEGORY_COLORS: Record<string, string> = {
  person: "#22c55e",   // green-500
  pet: "#f97316",      // orange-500
  vehicle: "#3b82f6",  // blue-500
  other: "#a855f7",    // purple-500
};

// Per-object color palette (rotate for multiple objects in same category)
const PERSON_PALETTE = ["#22c55e", "#10b981", "#14b8a6", "#06b6d4", "#0ea5e9", "#6366f1"];
const PET_PALETTE = ["#f97316", "#f59e0b", "#eab308", "#ef4444", "#ec4899"];

function PresenceTimelineView({ rows, date, navigate }: {
  rows: PresenceRow[];
  date: string;
  navigate: (path: string) => void;
}) {
  if (!rows.length) {
    return <p className="text-xs text-slate-600 text-center py-4">No presence data for this date</p>;
  }

  // Assign colors per object
  const personIdx = { current: 0 };
  const petIdx = { current: 0 };
  const colorMap = new Map<number, string>();
  for (const row of rows) {
    if (row.category === "person") {
      colorMap.set(row.named_object_id, PERSON_PALETTE[personIdx.current % PERSON_PALETTE.length]);
      personIdx.current++;
    } else if (row.category === "pet") {
      colorMap.set(row.named_object_id, PET_PALETTE[petIdx.current % PET_PALETTE.length]);
      petIdx.current++;
    } else {
      colorMap.set(row.named_object_id, CATEGORY_COLORS[row.category] || "#a855f7");
    }
  }

  // Get the day boundaries in local time
  const dayStart = new Date(`${date}T00:00:00`).getTime();
  const dayEnd = dayStart + 24 * 60 * 60 * 1000;
  const dayMs = dayEnd - dayStart;

  const toPercent = (iso: string) => {
    const t = new Date(iso).getTime();
    return Math.max(0, Math.min(100, ((t - dayStart) / dayMs) * 100));
  };

  // Group by category for section headers
  let lastCategory = "";

  return (
    <div className="space-y-1">
      {/* Hour ticks */}
      <div className="relative h-4 mb-1">
        {[0, 3, 6, 9, 12, 15, 18, 21].map((hr) => (
          <span
            key={hr}
            className="absolute text-[8px] text-slate-600 -translate-x-1/2"
            style={{ left: `${(hr / 24) * 100}%` }}
          >
            {String(hr).padStart(2, "0")}
          </span>
        ))}
      </div>

      {rows.map((row) => {
        const color = colorMap.get(row.named_object_id) || "#888";
        const showHeader = row.category !== lastCategory;
        lastCategory = row.category;

        return (
          <div key={row.named_object_id}>
            {showHeader && (
              <p className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mt-2 mb-0.5">
                {row.category === "person" ? "People" : row.category === "pet" ? "Pets" : row.category === "vehicle" ? "Vehicles" : "Other"}
              </p>
            )}
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-300 w-14 truncate shrink-0" title={row.name}>
                {row.name}
              </span>
              <div className="flex-1 relative h-5 bg-slate-800 rounded overflow-hidden">
                {/* Hour grid lines */}
                {[6, 12, 18].map((hr) => (
                  <div
                    key={hr}
                    className="absolute top-0 bottom-0 w-px bg-slate-700/50"
                    style={{ left: `${(hr / 24) * 100}%` }}
                  />
                ))}
                {/* Presence bars */}
                {row.bars.map((bar, i) => {
                  const left = toPercent(bar.start);
                  const right = toPercent(bar.end);
                  const width = Math.max(right - left, 0.3); // min visible width
                  const startTime = new Date(bar.start).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                  const endTime = new Date(bar.end).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                  const durationMs = new Date(bar.end).getTime() - new Date(bar.start).getTime();
                  const durationMin = Math.round(durationMs / 60000);
                  const durationLabel = durationMin >= 60
                    ? `${Math.floor(durationMin / 60)}h ${durationMin % 60}m`
                    : `${durationMin}m`;
                  return (
                    <button
                      key={i}
                      className="absolute top-0.5 bottom-0.5 rounded-sm cursor-pointer hover:brightness-125 transition-all"
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                        backgroundColor: color,
                        opacity: 0.85,
                      }}
                      title={`${row.name}: ${startTime} – ${endTime} (${durationLabel})`}
                      onClick={() => navigate(`/events?id=${bar.event_id}`)}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        );
      })}

      {/* Now indicator */}
      {(() => {
        const now = new Date();
        const nowStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
        if (date !== nowStr) return null;
        const pct = ((now.getTime() - dayStart) / dayMs) * 100;
        if (pct < 0 || pct > 100) return null;
        return (
          <div className="relative h-0 -mt-1" style={{ marginLeft: "4.5rem" }}>
            <div className="absolute top-0 w-px h-full bg-red-500/60" style={{ left: `${pct}%` }} />
          </div>
        );
      })()}
    </div>
  );
}

/* ═══════════ Detection Settings Panel (replaces event list when active) ═══════════ */

interface PreviewDetection {
  class_name: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

interface PreviewResult {
  detections: PreviewDetection[];
  snapshot_b64: string;
  image_width: number;
  image_height: number;
}

interface NamedObj {
  id: number;
  name: string;
  category: string;
  reference_image_count: number;
}

function DetectionSettingsPanel({ cameraId }: { cameraId: number }) {
  const qc = useQueryClient();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const token = getToken();

  const { data: detData } = useQuery({
    queryKey: ["detection-settings", cameraId],
    queryFn: () => api.get<DetectionData>(`/api/cameras/${cameraId}/detection-settings`),
  });

  // Named objects the AI is actively recognizing
  const { data: namedObjects } = useQuery({
    queryKey: ["named-objects"],
    queryFn: () => api.get<NamedObj[]>("/api/training/objects"),
  });

  const [localConf, setLocalConf] = useState(0.5);
  const [localSettings, setLocalSettings] = useState<Record<string, ObjectSettings>>({});
  const [dirty, setDirty] = useState(false);

  // Preview state
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Train from snapshot
  const [trainMode, setTrainMode] = useState(false);
  const [trainName, setTrainName] = useState("");
  const [trainCategory, setTrainCategory] = useState("pet");
  const [selectedDet, setSelectedDet] = useState<PreviewDetection | null>(null);
  const [customBbox, setCustomBbox] = useState<{x1:number;y1:number;x2:number;y2:number}|null>(null);
  const drawingRef = useRef<{startX:number;startY:number}|null>(null);
  const previewImgRef = useRef<HTMLImageElement|null>(null);
  const [linkExistingId, setLinkExistingId] = useState<number | null>(null);

  useEffect(() => {
    if (detData) {
      setLocalConf(detData.detection_confidence);
      setLocalSettings(detData.detection_settings || {});
      setDirty(false);
    }
  }, [detData]);

  const saveMut = useMutation({
    mutationFn: (payload: { detection_objects: string[]; detection_confidence: number; detection_settings: Record<string, ObjectSettings> }) =>
      api.put(`/api/cameras/${cameraId}/detection-settings`, payload),
    onSuccess: () => {
      setDirty(false);
      qc.invalidateQueries({ queryKey: ["detection-settings", cameraId] });
    },
  });

  const handleSave = () => {
    if (!detData) return;
    const filtered: Record<string, ObjectSettings> = {};
    for (const cls of detData.detection_objects) {
      if (localSettings[cls]) filtered[cls] = localSettings[cls];
    }
    saveMut.mutate({
      detection_objects: detData.detection_objects,
      detection_confidence: localConf,
      detection_settings: filtered,
    });
  };

  const updateSetting = (cls: string, key: keyof ObjectSettings, value: number) => {
    setDirty(true);
    setLocalSettings((s) => ({ ...s, [cls]: { ...s[cls], [key]: value } }));
  };

  // Capture snapshot + run detection with current settings
  const capturePreview = async () => {
    if (!detData) return;
    setPreviewLoading(true);
    try {
      const filtered: Record<string, ObjectSettings> = {};
      for (const cls of detData.detection_objects) {
        if (localSettings[cls]) filtered[cls] = localSettings[cls];
      }
      const result = await api.post<PreviewResult>(`/api/cameras/${cameraId}/detect-preview`, {
        detection_objects: detData.detection_objects,
        detection_confidence: localConf,
        detection_settings: filtered,
      });
      setPreview(result);
      setSelectedDet(null);
      setCustomBbox(null);
    } catch { /* ignore */ }
    setPreviewLoading(false);
  };

  // Re-run detection on existing snapshot when settings change
  const rerunPreview = async () => {
    if (!detData || !preview) return;
    setPreviewLoading(true);
    try {
      const filtered: Record<string, ObjectSettings> = {};
      for (const cls of detData.detection_objects) {
        if (localSettings[cls]) filtered[cls] = localSettings[cls];
      }
      const result = await api.post<PreviewResult>(`/api/cameras/${cameraId}/detect-preview`, {
        detection_objects: detData.detection_objects,
        detection_confidence: localConf,
        detection_settings: filtered,
      });
      // Keep same snapshot, update detections
      setPreview((prev) => prev ? { ...result, snapshot_b64: prev.snapshot_b64 } : result);
    } catch { /* ignore */ }
    setPreviewLoading(false);
  };

  // Auto-rerun when settings change and we have a preview
  const settingsKey = JSON.stringify({ localConf, localSettings });
  const prevSettingsKey = useRef(settingsKey);
  useEffect(() => {
    if (prevSettingsKey.current !== settingsKey && preview) {
      prevSettingsKey.current = settingsKey;
      const timer = setTimeout(rerunPreview, 500); // debounce
      return () => clearTimeout(timer);
    }
    prevSettingsKey.current = settingsKey;
  }, [settingsKey, preview]);

  // Draw helper — renders image + detections + custom/drag box
  const drawOnCanvas = (tempDragBox?: {x1:number;y1:number;x2:number;y2:number}) => {
    const canvas = canvasRef.current;
    const img = previewImgRef.current;
    if (!canvas || !img || !preview) return;

    const aspect = img.width / img.height;
    canvas.width = canvas.parentElement?.clientWidth || 300;
    canvas.height = canvas.width / aspect;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const sx = canvas.width / preview.image_width;
    const sy = canvas.height / preview.image_height;

    const COLORS: Record<string, string> = {
      person: "#22c55e", cat: "#f97316", dog: "#f97316", bird: "#f97316",
      car: "#3b82f6", truck: "#3b82f6", bus: "#3b82f6", motorcycle: "#3b82f6",
    };

    for (const det of preview.detections) {
      const { x1, y1, x2, y2 } = det.bbox;
      const dx = x1 * sx;
      const dy = y1 * sy;
      const dw = (x2 - x1) * sx;
      const dh = (y2 - y1) * sy;
      const color = COLORS[det.class_name] || "#6b7280";
      const isSelected = selectedDet && det.bbox.x1 === selectedDet.bbox.x1 && det.bbox.y1 === selectedDet.bbox.y1;

      ctx.strokeStyle = isSelected ? "#facc15" : color;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(dx, dy, dw, dh);

      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 11px sans-serif";
      const tm = ctx.measureText(label);
      ctx.fillStyle = isSelected ? "#facc15" : color;
      ctx.fillRect(dx, dy - 16, tm.width + 8, 16);
      ctx.fillStyle = isSelected ? "#000" : "#fff";
      ctx.fillText(label, dx + 4, dy - 4);
    }

    // Draw custom bbox or active drag box
    const box = tempDragBox || customBbox;
    if (box) {
      const dx = box.x1 * sx;
      const dy = box.y1 * sy;
      const dw = (box.x2 - box.x1) * sx;
      const dh = (box.y2 - box.y1) * sy;
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(dx, dy, dw, dh);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(250, 204, 21, 0.15)";
      ctx.fillRect(dx, dy, dw, dh);
    }
  };

  // Load preview image + draw
  useEffect(() => {
    if (!preview) { previewImgRef.current = null; return; }
    const img = new Image();
    img.onload = () => { previewImgRef.current = img; drawOnCanvas(); };
    img.src = `data:image/jpeg;base64,${preview.snapshot_b64}`;
  }, [preview, selectedDet, customBbox]);

  // Convert pointer event to image coordinates
  const pointerToImageCoords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !preview) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * preview.image_width,
      y: ((e.clientY - rect.top) / rect.height) * preview.image_height,
    };
  };

  // Tap existing detection → select it; drag on empty space → draw custom box
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const coords = pointerToImageCoords(e);
    if (!coords || !preview) return;

    // Check if clicking on an existing detection
    for (const det of preview.detections) {
      if (coords.x >= det.bbox.x1 && coords.x <= det.bbox.x2 && coords.y >= det.bbox.y1 && coords.y <= det.bbox.y2) {
        setSelectedDet(det);
        setCustomBbox(null);
        setTrainMode(true);
        const t = det.class_name;
        if (t === "person") setTrainCategory("person");
        else if (["cat", "dog", "bird"].includes(t)) setTrainCategory("pet");
        else if (["car", "truck", "bus", "motorcycle", "bicycle"].includes(t)) setTrainCategory("vehicle");
        else setTrainCategory("other");
        return;
      }
    }

    // Start drawing a custom box on empty space
    setSelectedDet(null);
    setCustomBbox(null);
    setTrainMode(false);
    drawingRef.current = { startX: coords.x, startY: coords.y };
    canvasRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current || !preview) return;
    const coords = pointerToImageCoords(e);
    if (!coords) return;
    const { startX, startY } = drawingRef.current;
    const box = {
      x1: Math.max(0, Math.min(startX, coords.x)),
      y1: Math.max(0, Math.min(startY, coords.y)),
      x2: Math.min(preview.image_width, Math.max(startX, coords.x)),
      y2: Math.min(preview.image_height, Math.max(startY, coords.y)),
    };
    drawOnCanvas(box);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const start = drawingRef.current;
    drawingRef.current = null;
    if (!start || !preview) return;
    const coords = pointerToImageCoords(e);
    if (!coords) { drawOnCanvas(); return; }

    const box = {
      x1: Math.max(0, Math.min(start.startX, coords.x)),
      y1: Math.max(0, Math.min(start.startY, coords.y)),
      x2: Math.min(preview.image_width, Math.max(start.startX, coords.x)),
      y2: Math.min(preview.image_height, Math.max(start.startY, coords.y)),
    };

    // Minimum 20x20px in image space
    if (box.x2 - box.x1 > 20 && box.y2 - box.y1 > 20) {
      setCustomBbox(box);
      setTrainMode(true);
      setTrainCategory("pet");
    } else {
      drawOnCanvas();
    }
  };

  // Train mutation — save snapshot crop as new named object
  const trainMut = useMutation({
    mutationFn: async (payload: { name: string; category: string; snapshot_b64: string; bbox: { x1: number; y1: number; x2: number; y2: number } }) => {
      // Crop the snapshot on client side by drawing to an offscreen canvas
      const img = new Image();
      await new Promise<void>((resolve) => { img.onload = () => resolve(); img.src = `data:image/jpeg;base64,${payload.snapshot_b64}`; });
      const offscreen = document.createElement("canvas");
      const { x1, y1, x2, y2 } = payload.bbox;
      offscreen.width = x2 - x1;
      offscreen.height = y2 - y1;
      const octx = offscreen.getContext("2d");
      if (octx) octx.drawImage(img, x1, y1, x2 - x1, y2 - y1, 0, 0, x2 - x1, y2 - y1);
      const cropB64 = offscreen.toDataURL("image/jpeg", 0.9).split(",")[1];

      return api.post("/api/training/create-and-train-image",
        linkExistingId
          ? { object_id: linkExistingId, image_b64: cropB64 }
          : { name: payload.name, category: payload.category, image_b64: cropB64 },
      );
    },
    onSuccess: () => {
      setTrainMode(false);
      setTrainName("");
      setSelectedDet(null);
      setCustomBbox(null);
      setLinkExistingId(null);
      qc.invalidateQueries({ queryKey: ["named-objects"] });
    },
  });

  const CATEGORY_OPTIONS = [
    { value: "person", label: "Person", icon: User },
    { value: "pet", label: "Pet", icon: Cat },
    { value: "vehicle", label: "Vehicle", icon: Car },
    { value: "other", label: "Other", icon: Box },
  ];

  if (!detData) return <p className="text-xs text-slate-500 text-center py-4">Loading...</p>;

  const trainedObjects = namedObjects?.filter((o) => o.reference_image_count > 0) || [];

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold flex items-center gap-1.5">
          <SlidersHorizontal size={13} className="text-amber-400" /> Detection Tuning
        </h3>
        <div className="flex items-center gap-1">
          <button
            onClick={capturePreview}
            disabled={previewLoading}
            className="flex items-center gap-1 text-[10px] px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 transition-colors font-medium"
          >
            <CameraIcon size={10} /> {previewLoading ? "..." : "Capture"}
          </button>
          {dirty && (
            <button
              onClick={handleSave}
              disabled={saveMut.isPending}
              className="flex items-center gap-1 text-[10px] px-2 py-1 rounded bg-blue-600 hover:bg-blue-500 transition-colors font-medium"
            >
              <Save size={10} /> Save
            </button>
          )}
        </div>
      </div>

      {/* Preview snapshot with detections */}
      {preview && (
        <div className="space-y-1">
          <div className="relative">
            <canvas
              ref={canvasRef}
              className="w-full rounded-lg cursor-crosshair touch-none"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
            />
            {previewLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-lg">
                <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              </div>
            )}
          </div>
          <p className="text-[9px] text-slate-500 text-center">
            {preview.detections.length} detection{preview.detections.length !== 1 ? "s" : ""} found
            {" · Tap detection or draw box to train"}
          </p>
        </div>
      )}

      {/* Train from selected detection */}
      {trainMode && (selectedDet || customBbox) && (
        <div className="bg-slate-800 rounded-lg p-2.5 space-y-2">
          <p className="text-[10px] font-medium flex items-center gap-1">
            <GraduationCap size={12} className="text-amber-400" /> Train: {selectedDet?.class_name || "custom selection"}
          </p>

          {/* Existing named objects to link to */}
          {namedObjects && namedObjects.length > 0 && (
            <div className="space-y-1">
              <p className="text-[9px] text-slate-500">Link to existing:</p>
              <div className="flex flex-wrap gap-1">
                {namedObjects.map((obj) => (
                  <button
                    key={obj.id}
                    onClick={() => { setLinkExistingId(obj.id); setTrainName(obj.name); setTrainCategory(obj.category); }}
                    className={`inline-flex items-center gap-0.5 text-[9px] px-1.5 py-0.5 rounded transition-colors ${
                      linkExistingId === obj.id
                        ? "bg-purple-600 text-white"
                        : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                    }`}
                  >
                    {obj.name}
                    <span className="text-[8px] opacity-60">({obj.reference_image_count})</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Or create new */}
          <div className="space-y-1.5">
            {linkExistingId ? (
              <p className="text-[9px] text-slate-400">Adding image to <span className="text-purple-300 font-medium">{trainName}</span>
                <button onClick={() => { setLinkExistingId(null); setTrainName(""); }} className="text-red-400 ml-1 underline">clear</button>
              </p>
            ) : (
              <>
                <p className="text-[9px] text-slate-500">Or create new:</p>
                <input
                  type="text"
                  placeholder="Name (e.g. Luna, My Car)"
                  value={trainName}
                  onChange={(e) => setTrainName(e.target.value)}
                  className="input text-xs w-full py-1.5"
                  autoFocus
                />
                <div className="flex gap-1">
                  {CATEGORY_OPTIONS.map((cat) => {
                    const Icon = cat.icon;
                    return (
                      <button
                        key={cat.value}
                        onClick={() => setTrainCategory(cat.value)}
                        className={`flex-1 flex items-center justify-center gap-0.5 py-1 rounded text-[9px] font-medium transition-colors ${
                          trainCategory === cat.value
                            ? "bg-blue-600 text-white"
                            : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                        }`}
                      >
                        <Icon size={10} /> {cat.label}
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>
          <div className="flex gap-1.5">
            <button
              onClick={() => {
                const bbox = selectedDet?.bbox || customBbox;
                if (trainName.trim() && preview && bbox) {
                  trainMut.mutate({
                    name: trainName.trim(),
                    category: trainCategory,
                    snapshot_b64: preview.snapshot_b64,
                    bbox,
                  });
                }
              }}
              disabled={(!trainName.trim() && !linkExistingId) || trainMut.isPending}
              className="btn-primary text-[10px] py-1 px-3 flex-1 disabled:opacity-40"
            >
              {trainMut.isPending ? "Training..." : linkExistingId ? "Add Image" : "Create & Train"}
            </button>
            <button onClick={() => { setTrainMode(false); setSelectedDet(null); setCustomBbox(null); setLinkExistingId(null); }} className="btn-secondary text-[10px] py-1 px-2">
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Settings sliders */}
      <div>
        <label className="text-[10px] text-slate-400">Global Confidence: {(localConf * 100).toFixed(0)}%</label>
        <input
          type="range" min="0.1" max="0.95" step="0.05" value={localConf}
          onChange={(e) => { setLocalConf(parseFloat(e.target.value)); setDirty(true); }}
          className="w-full"
        />
      </div>
      {detData.detection_objects.map((cls) => {
        const s = localSettings[cls] || {};
        const conf = s.confidence ?? localConf;
        const minArea = s.min_area ?? 0;
        return (
          <div key={cls} className="bg-slate-800 rounded-lg p-2 space-y-1.5">
            <span className="text-xs font-medium capitalize">{cls}</span>
            <div>
              <label className="text-[10px] text-slate-500">Confidence: {(conf * 100).toFixed(0)}%</label>
              <input
                type="range" min="0.1" max="0.95" step="0.05" value={conf}
                onChange={(e) => updateSetting(cls, "confidence", parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[9px] text-slate-600">
                <span>More detections</span>
                <span>Fewer false positives</span>
              </div>
            </div>
            <div>
              <label className="text-[10px] text-slate-500">Min Area: {minArea}px²</label>
              <input
                type="range" min="0" max="20000" step="500" value={minArea}
                onChange={(e) => updateSetting(cls, "min_area", parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[9px] text-slate-600">
                <span>All sizes</span>
                <span>Large only</span>
              </div>
            </div>
          </div>
        );
      })}

      {/* Named objects the AI recognizes */}
      {trainedObjects.length > 0 && (
        <div className="space-y-1.5">
          <h4 className="text-[10px] font-semibold text-slate-400 flex items-center gap-1">
            <Sparkles size={11} className="text-purple-400" /> AI Recognizes ({trainedObjects.length})
          </h4>
          <div className="flex flex-wrap gap-1">
            {trainedObjects.map((obj) => (
              <span
                key={obj.id}
                className="inline-flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded bg-purple-900/40 text-purple-300 border border-purple-800/30"
              >
                <Eye size={9} /> {obj.name}
                <span className="text-purple-500">({obj.reference_image_count})</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
