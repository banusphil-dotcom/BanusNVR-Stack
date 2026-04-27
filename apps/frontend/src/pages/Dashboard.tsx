import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import CameraStream from "../components/CameraStream";
import {
  Camera,
  ChevronLeft,
  ChevronRight,
  Video,
  Film,
  Maximize2,
  Minimize2,
} from "lucide-react";

interface CameraInfo {
  id: number;
  name: string;
  camera_type: string;
  enabled: boolean;
}

interface DashboardLayoutData {
  layout: string;
  camera_order: number[];
  hidden_cameras: number[];
}

function snapUrl(camId: number) {
  return `/frigate/api/camera_${camId}/latest.jpg`;
}

/* ── Snapshot tile ── */
function SnapshotTile({
  camera,
  active,
  onClick,
}: {
  camera: CameraInfo;
  active?: boolean;
  onClick: () => void;
}) {
  const [imgSrc, setImgSrc] = useState(snapUrl(camera.id));
  const [hasError, setHasError] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    setImgSrc(`${snapUrl(camera.id)}?_=${Date.now()}`);
    intervalRef.current = setInterval(() => {
      setImgSrc(`${snapUrl(camera.id)}?_=${Date.now()}`);
    }, 5000);
    return () => clearInterval(intervalRef.current);
  }, [camera.id]);

  return (
    <div
      className={`relative w-full h-full bg-slate-900 rounded overflow-hidden cursor-pointer
        transition-all duration-150 select-none
        ${active
          ? "ring-2 ring-blue-500 shadow-md shadow-blue-500/25"
          : "ring-1 ring-slate-700/40 active:scale-95"}`}
      onClick={onClick}
    >
      {!hasError ? (
        <img
          src={imgSrc}
          alt={camera.name}
          className="w-full h-full object-cover"
          draggable={false}
          onError={() => setHasError(true)}
          onLoad={() => setHasError(false)}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-slate-800/60">
          <Camera size={14} className="text-slate-600" />
        </div>
      )}

      {/* Name at bottom */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-1 py-0.5">
        <span className="text-[8px] sm:text-[10px] font-medium text-white drop-shadow truncate block">
          {camera.name}
        </span>
      </div>

      {/* Status dot */}
      <div className="absolute top-0.5 left-0.5">
        <div className={`w-1.5 h-1.5 rounded-full ${
          hasError ? "bg-red-500" : "bg-green-500 shadow shadow-green-500/50"
        }`} />
      </div>
    </div>
  );
}

/* ── Main Dashboard ── */
export default function Dashboard() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: cameras } = useQuery({
    queryKey: ["cameras"],
    queryFn: () => api.get<CameraInfo[]>("/api/cameras"),
  });

  const { data: savedLayout } = useQuery({
    queryKey: ["dashboard-layout"],
    queryFn: () => api.get<DashboardLayoutData>("/api/system/dashboard-layout"),
  });

  const saveLayout = useMutation({
    mutationFn: (data: DashboardLayoutData) =>
      api.put<DashboardLayoutData>("/api/system/dashboard-layout", data),
    onSuccess: (data) => queryClient.setQueryData(["dashboard-layout"], data),
  });

  const activeCameras = cameras?.filter((c) => c.enabled) ?? [];

  const [cameraOrder, setCameraOrder] = useState<number[]>([]);
  const [selectedCamId, setSelectedCamId] = useState<number | null>(null);
  const [initialized, setInitialized] = useState(false);
  const liveContainerRef = useRef<HTMLDivElement | null>(null);
  const [isLiveFullscreen, setIsLiveFullscreen] = useState(false);

  useEffect(() => {
    const onFsChange = () => setIsLiveFullscreen(document.fullscreenElement === liveContainerRef.current);
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  const toggleLiveFullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else if (liveContainerRef.current) {
      liveContainerRef.current.requestFullscreen().catch(() => {});
    }
  }, []);

  useEffect(() => {
    if (!savedLayout || initialized) return;
    if (savedLayout.camera_order?.length) setCameraOrder(savedLayout.camera_order);
    setInitialized(true);
  }, [savedLayout, initialized]);

  useEffect(() => {
    if (selectedCamId === null && activeCameras.length > 0) {
      const ordered = getOrderedCameras(activeCameras, cameraOrder);
      setSelectedCamId(ordered[0]?.id ?? null);
    }
  }, [activeCameras, cameraOrder, selectedCamId]);

  const getOrderedCameras = useCallback((cams: CameraInfo[], order: number[]): CameraInfo[] => {
    if (!order.length) return cams;
    const ordered: CameraInfo[] = [];
    const remaining = [...cams];
    for (const id of order) {
      const idx = remaining.findIndex((c) => c.id === id);
      if (idx >= 0) ordered.push(remaining.splice(idx, 1)[0]);
    }
    return [...ordered, ...remaining];
  }, []);

  const orderedCameras = useMemo(
    () => getOrderedCameras(activeCameras, cameraOrder),
    [activeCameras, cameraOrder, getOrderedCameras]
  );

  const selectedCam = orderedCameras.find((c) => c.id === selectedCamId) ?? null;
  const selectedIdx = selectedCam ? orderedCameras.indexOf(selectedCam) : -1;

  // Keyboard nav
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!selectedCam) return;
      const idx = orderedCameras.indexOf(selectedCam);
      if (e.key === "ArrowLeft" && idx > 0) setSelectedCamId(orderedCameras[idx - 1].id);
      else if (e.key === "ArrowRight" && idx < orderedCameras.length - 1) setSelectedCamId(orderedCameras[idx + 1].id);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedCam, orderedCameras]);

  const handleSelectCamera = useCallback((camId: number) => {
    setSelectedCamId(camId);
  }, []);

  if (orderedCameras.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400">
        <Camera size={48} className="mb-3 opacity-50" />
        <p className="text-lg font-medium">No cameras configured</p>
        <p className="text-sm mt-1">Add cameras in the Cameras tab to get started</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* ── Main live view — capped at ~33 % of available height ── */}
      {selectedCam && (
        <div ref={liveContainerRef} className="shrink-0 h-[33%] relative bg-black group/live">
          {/* Click-through overlay → navigate to dedicated camera page */}
          <button
            type="button"
            onClick={() => navigate(`/camera/${selectedCam.id}`)}
            className="absolute inset-0 z-[1] cursor-pointer"
            aria-label={`Open ${selectedCam.name} dedicated view`}
          />
          <CameraStream
            key={selectedCam.id}
            cameraId={selectedCam.id}
            cameraName={selectedCam.name}
            className="w-full h-full"
            hideLabel
          />

          {/* Prev / Next */}
          {selectedIdx > 0 && (
            <button
              onClick={(e) => { e.stopPropagation(); handleSelectCamera(orderedCameras[selectedIdx - 1].id); }}
              className="absolute left-1 top-1/2 -translate-y-1/2 p-1.5 bg-black/60 hover:bg-black/90 active:bg-blue-600 rounded-full text-white z-10 transition-colors"
            >
              <ChevronLeft size={20} />
            </button>
          )}
          {selectedIdx < orderedCameras.length - 1 && (
            <button
              onClick={(e) => { e.stopPropagation(); handleSelectCamera(orderedCameras[selectedIdx + 1].id); }}
              className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 bg-black/60 hover:bg-black/90 active:bg-blue-600 rounded-full text-white z-10 transition-colors"
            >
              <ChevronRight size={20} />
            </button>
          )}

          {/* Fullscreen toggle */}
          <button
            onClick={(e) => { e.stopPropagation(); toggleLiveFullscreen(); }}
            className="absolute top-1 right-1 p-1.5 bg-black/60 hover:bg-black/90 rounded text-white z-10 transition-colors"
            aria-label={isLiveFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            title={isLiveFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isLiveFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      )}

      {/* ── Info bar: camera name + Open Recordings ── */}
      {selectedCam && (
        <div className="shrink-0 flex items-center justify-between px-3 py-1.5 bg-slate-900/90 border-y border-slate-800/60">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <Video size={14} className="text-blue-400 shrink-0" />
            <span className="text-xs sm:text-sm font-semibold text-white truncate">{selectedCam.name}</span>
            <span className="text-[10px] text-slate-500 shrink-0 tabular-nums">{selectedIdx + 1}/{orderedCameras.length}</span>
          </div>
          <button
            onClick={() => navigate(`/camera/${selectedCam.id}`)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-xs font-medium transition-colors shrink-0"
          >
            <Film size={13} />
            Open Recordings
          </button>
        </div>
      )}

      {/* ── Camera grid — horizontal scroll, 16:9 tiles, no vertical scroll ── */}
      <div className="shrink-0 overflow-x-auto overflow-y-hidden scrollbar-none py-1 px-1.5"
           style={{ height: "42%", WebkitOverflowScrolling: "touch" }}>
        <div className="grid grid-rows-3 grid-flow-col gap-1.5 h-full w-max">
          {orderedCameras.map((cam) => (
            <div key={cam.id} className="h-full" style={{ aspectRatio: "16/9" }}>
              <SnapshotTile
                camera={cam}
                active={cam.id === selectedCamId}
                onClick={() => handleSelectCamera(cam.id)}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
