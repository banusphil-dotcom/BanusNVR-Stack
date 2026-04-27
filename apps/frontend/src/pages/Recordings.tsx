import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { api, getToken } from "../api";
import Hls from "hls.js";
import { Play, Pause, Download, Calendar } from "lucide-react";

interface CameraInfo {
  id: number;
  name: string;
}

interface TimelineHour {
  hour: string;
  has_recording: boolean;
  segments: number;
}

interface TimelineResponse {
  camera_id: number;
  date: string;
  hours: TimelineHour[];
}

export default function Recordings() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [searchParams] = useSearchParams();
  const paramCamera = searchParams.get("camera");
  const paramTime = searchParams.get("time");

  const [cameraId, setCameraId] = useState<number | null>(
    paramCamera ? Number(paramCamera) : null
  );
  const [date, setDate] = useState(() => {
    const d = paramTime ? new Date(paramTime) : new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  });
  const [startTime, setStartTime] = useState(() => {
    if (paramTime) {
      // Start 1 minute before the event
      const d = new Date(paramTime);
      d.setMinutes(d.getMinutes() - 1);
      return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    }
    return "00:00";
  });
  const [endTime, setEndTime] = useState(() => {
    if (paramTime) {
      // End 2 minutes after the event
      const d = new Date(paramTime);
      d.setMinutes(d.getMinutes() + 2);
      return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    }
    return "23:59";
  });
  const [playing, setPlaying] = useState(false);
  const [autoPlay, setAutoPlay] = useState(!!paramTime);

  const { data: cameras } = useQuery({
    queryKey: ["cameras"],
    queryFn: () => api.get<CameraInfo[]>("/api/cameras"),
  });

  const { data: timeline } = useQuery({
    queryKey: ["rec-timeline", cameraId, date],
    queryFn: () => api.get<TimelineResponse>(`/api/recordings/${cameraId}/timeline?date=${date}`),
    enabled: cameraId != null,
  });

  useEffect(() => {
    if (!cameras?.length || cameraId != null) return;
    setCameraId(cameras[0].id);
  }, [cameras, cameraId]);

  const loadPlayback = () => {
    const video = videoRef.current;
    if (!video || cameraId == null) return;

    const startDate = new Date(`${date}T${startTime}:00`);
    const endDate = new Date(`${date}T${endTime}:59`);
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
      hls.loadSource(url);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play();
        setPlaying(true);
      });
      hls.on(Hls.Events.ERROR, (_event, data) => {
        if (data.fatal) {
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            hls.startLoad();
          } else {
            hls.destroy();
          }
        }
      });
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = url;
      video.play();
      setPlaying(true);
    }
  };

  // Auto-play when navigating from Events page
  useEffect(() => {
    if (autoPlay && cameraId != null && timeline) {
      setAutoPlay(false);
      loadPlayback();
    }
  }, [autoPlay, cameraId, timeline]);

  const exportClip = async () => {
    if (cameraId == null) return;
    const startDate = new Date(`${date}T${startTime}:00`);
    const endDate = new Date(`${date}T${endTime}:59`);

    const res = await fetch(
      `/api/recordings/${cameraId}/export?start=${encodeURIComponent(startDate.toISOString())}&end=${encodeURIComponent(endDate.toISOString())}`,
      { method: "POST", headers: { Authorization: `Bearer ${localStorage.getItem("banusnas_token")}` } },
    );

    if (res.ok) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `clip_${cameraId}_${date}.mp4`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-lg font-bold">Playback</h2>

      {/* Controls */}
      <div className="card space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label">Camera</label>
            <select className="input" value={cameraId ?? ""} onChange={(e) => setCameraId(Number(e.target.value))}>
              {cameras?.map((cam) => (
                <option key={cam.id} value={cam.id}>
                  {cam.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="label">Date</label>
            <input type="date" className="input" value={date} onChange={(e) => setDate(e.target.value)} />
          </div>
          <div>
            <label className="label">Start</label>
            <input type="time" className="input" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
          </div>
          <div>
            <label className="label">End</label>
            <input type="time" className="input" value={endTime} onChange={(e) => setEndTime(e.target.value)} />
          </div>
        </div>
        <div className="flex gap-2">
          <button onClick={loadPlayback} className="btn-primary flex-1 flex items-center justify-center gap-1.5">
            <Play size={16} /> Play
          </button>
          <button onClick={exportClip} className="btn-secondary flex items-center gap-1.5">
            <Download size={16} /> Export
          </button>
        </div>
      </div>

      {/* Video player */}
      <div className="relative bg-black rounded-xl overflow-hidden aspect-video">
        <video ref={videoRef} controls playsInline className="w-full h-full" />
      </div>

      {/* Timeline visualization */}
      {timeline?.hours && (
        <div className="card">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
            <Calendar size={14} /> Recording Timeline
          </h3>
          <div className="flex gap-0.5">
            {timeline.hours.map((seg) => (
              <div
                key={seg.hour}
                className={`flex-1 h-6 rounded-sm ${
                  seg.has_recording ? "bg-blue-600" : "bg-slate-800"
                }`}
                title={`${seg.hour} — ${seg.has_recording ? `${seg.segments} segments` : "no recording"}`}
              />
            ))}
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>00:00</span>
            <span>06:00</span>
            <span>12:00</span>
            <span>18:00</span>
            <span>23:59</span>
          </div>
        </div>
      )}
    </div>
  );
}
