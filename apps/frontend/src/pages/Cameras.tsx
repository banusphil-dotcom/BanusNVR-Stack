import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Plus, Trash2, Video, TestTube, X, Wifi, Pencil, Search, Loader2, Router, SlidersHorizontal, Move, Eye, EyeOff, RefreshCw } from "lucide-react";

export interface PtzConfig {
  enabled: boolean;
  protocol: "onvif" | "tapo" | "none";
  onvif_host?: string;
  onvif_port?: number;
  onvif_user?: string;
  onvif_password?: string;
  autotrack_enabled?: boolean;
  autotrack_objects?: string[];
  autotrack_timeout?: number;
}

export interface CameraInfo {
  id: number;
  name: string;
  camera_type: string;
  connection_config: Record<string, string>;
  enabled: boolean;
  recording_mode: string;
  detection_enabled: boolean;
  detection_objects: string[];
  detection_confidence: number;
  detection_settings: Record<string, any> | null;
  ptz_mode: boolean;
  ptz_config: PtzConfig | null;
  is_recording: boolean;
  is_detecting: boolean;
}

interface ProbeResult {
  path: string;
  available: boolean;
  source_url: string;
  snapshot: string | null;
  width: number | null;
  height: number | null;
  status: "ok" | "no_frame" | "timeout" | "error";
  codec: string | null;
  quality: "hd" | "sd" | "low" | null;
  transport?: "tcp" | "udp";
}

interface ScanDevice {
  ip: string;
  ports: { port: number; protocol: string; service: string }[];
  mac: string | null;
  brand: string | null;
  inferred_type: string;
  has_rtsp: boolean;
  has_http: boolean;
}

/** Quality / status helpers for probed stream cards */
const STATUS_DOT: Record<string, string> = {
  ok: "bg-green-400",
  no_frame: "bg-yellow-400",
  timeout: "bg-red-400",
  error: "bg-slate-600",
};
const STATUS_LABEL: Record<string, string> = {
  ok: "Live",
  no_frame: "Connected",
  timeout: "Timeout",
  error: "Error",
};
const QUALITY_BADGE: Record<string, { bg: string; text: string; label: string }> = {
  hd: { bg: "bg-green-900/60", text: "text-green-300", label: "HD" },
  sd: { bg: "bg-yellow-900/60", text: "text-yellow-300", label: "SD" },
  low: { bg: "bg-red-900/60", text: "text-red-300", label: "Low" },
};

/** Reusable probe results grid with show-hidden toggle and rescan */
function ProbeResultsGrid({
  probeResults,
  setProbeResults,
  config,
  setConfig,
  setPreview,
  cameraType,
}: {
  probeResults: ProbeResult[];
  setProbeResults: React.Dispatch<React.SetStateAction<ProbeResult[] | null>>;
  config: Record<string, string>;
  setConfig: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  setPreview: (snap: string | null) => void;
  cameraType: string;
}) {
  const [showHidden, setShowHidden] = useState(false);
  const [rescanPaths, setRescanPaths] = useState<Set<string>>(new Set());
  const [rescanning, setRescanning] = useState(false);

  const available = probeResults.filter((r) => r.available);
  const hidden = probeResults.filter((r) => !r.available);
  const displayed = showHidden ? probeResults : available;

  const toggleRescan = (path: string) => {
    setRescanPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const handleRescan = async () => {
    if (rescanPaths.size === 0) return;
    setRescanning(true);
    try {
      const data = await api.post<{ streams: ProbeResult[] }>("/api/cameras/probe-streams/rescan", {
        camera_type: cameraType,
        connection_config: config,
        paths: [...rescanPaths],
      });
      // Merge rescanned results into existing probeResults
      setProbeResults((prev) => {
        if (!prev) return prev;
        const updated = [...prev];
        for (const fresh of data.streams) {
          const idx = updated.findIndex((r) => r.path === fresh.path);
          if (idx >= 0) updated[idx] = fresh;
        }
        return updated;
      });
      setRescanPaths(new Set());
    } catch {
      // ignore
    } finally {
      setRescanning(false);
    }
  };

  if (available.length === 0 && hidden.length === 0) {
    return <p className="text-xs text-red-400">No streams found. Check IP address and credentials.</p>;
  }

  return (
    <div className="space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-400 font-medium">
          Found {available.length} stream{available.length !== 1 ? "s" : ""}
          {hidden.length > 0 && !showHidden && ` (${hidden.length} hidden)`}
        </p>
        {hidden.length > 0 && (
          <button
            type="button"
            onClick={() => { setShowHidden((v) => !v); setRescanPaths(new Set()); }}
            className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-200 transition-colors"
          >
            {showHidden ? <EyeOff size={12} /> : <Eye size={12} />}
            {showHidden ? "Hide unavailable" : `Show all (${probeResults.length})`}
          </button>
        )}
      </div>

      {/* Rescan bar */}
      {showHidden && hidden.length > 0 && (
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={rescanPaths.size === 0 || rescanning}
            onClick={handleRescan}
            className="flex items-center gap-1.5 text-[10px] px-2.5 py-1 rounded font-medium bg-blue-600 text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-blue-500 transition-colors"
          >
            <RefreshCw size={11} className={rescanning ? "animate-spin" : ""} />
            {rescanning ? "Rescanning…" : `Rescan${rescanPaths.size > 0 ? ` (${rescanPaths.size})` : ""}`}
          </button>
          {rescanPaths.size === 0 && !rescanning && (
            <span className="text-[10px] text-slate-500">Select streams to rescan</span>
          )}
        </div>
      )}

      {/* Thumbnail grid */}
      <div className="grid grid-cols-2 gap-2">
        {displayed.map((r) => {
          const isRecord = config.stream_path === r.path;
          const isDetect = config.sub_stream_path === r.path;
          const qb = r.quality ? QUALITY_BADGE[r.quality] : null;
          const isHidden = !r.available;
          const isChecked = rescanPaths.has(r.path);
          return (
            <div
              key={r.path}
              className={`rounded-lg border overflow-hidden transition-colors ${
                isRecord ? "border-emerald-500 ring-1 ring-emerald-500/40"
                : isDetect ? "border-blue-500 ring-1 ring-blue-500/40"
                : isHidden ? "border-slate-700/50 opacity-60"
                : "border-slate-700"
              }`}
            >
              {/* Preview area */}
              <div className="relative">
                {r.snapshot ? (
                  <img
                    src={`data:image/jpeg;base64,${r.snapshot}`}
                    alt={r.path}
                    className="w-full aspect-video object-cover"
                  />
                ) : (
                  <div className={`w-full aspect-video bg-slate-800 flex flex-col items-center justify-center gap-1 ${isHidden ? "bg-slate-900" : ""}`}>
                    <Video size={20} className="text-slate-600" />
                    <span className="text-[9px] text-slate-500">
                      {r.status === "no_frame" ? "Stream active — no keyframe"
                        : r.status === "error" ? "Unreachable"
                        : "No preview"}
                    </span>
                  </div>
                )}
                {/* Status + quality overlay */}
                <div className="absolute top-1 left-1 flex gap-1">
                  <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium bg-black/60 text-slate-200`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${STATUS_DOT[r.status] || "bg-slate-600"}`} />
                    {STATUS_LABEL[r.status] || r.status}
                  </span>
                  {qb && (
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${qb.bg} ${qb.text}`}>
                      {qb.label}
                    </span>
                  )}
                </div>
                {/* Codec overlay */}
                {r.codec && (
                  <span className="absolute top-1 right-1 flex gap-1">
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-black/60 text-slate-300">
                      {r.codec}
                    </span>
                    {r.transport === "udp" && (
                      <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-purple-900/60 text-purple-300">
                        UDP
                      </span>
                    )}
                  </span>
                )}
                {/* Rescan checkbox for hidden streams */}
                {isHidden && showHidden && (
                  <button
                    type="button"
                    onClick={() => toggleRescan(r.path)}
                    className={`absolute bottom-1 right-1 w-5 h-5 rounded border flex items-center justify-center text-[10px] transition-colors ${
                      isChecked
                        ? "bg-blue-600 border-blue-500 text-white"
                        : "bg-black/50 border-slate-500 text-slate-400 hover:border-blue-400"
                    }`}
                  >
                    {isChecked && "✓"}
                  </button>
                )}
              </div>
              <div className="p-2 bg-slate-800/80 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-200">{r.path}</span>
                  {r.width && r.height && (
                    <span className="text-[10px] font-mono text-slate-400">{r.width}×{r.height}</span>
                  )}
                </div>
                {r.available && (
                  <div className="flex gap-1">
                    <button
                      type="button"
                      onClick={() => {
                        setConfig((c) => ({ ...c, stream_path: r.path }));
                        if (r.snapshot) setPreview(r.snapshot);
                      }}
                      className={`flex-1 text-[10px] py-0.5 rounded font-medium transition-colors ${
                        isRecord ? "bg-emerald-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                      }`}
                    >
                      Record
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfig((c) => ({ ...c, sub_stream_path: r.path }))}
                      className={`flex-1 text-[10px] py-0.5 rounded font-medium transition-colors ${
                        isDetect ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                      }`}
                    >
                      Detect
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected streams summary */}
      <div className="flex gap-2 text-xs flex-wrap">
        {config.stream_path && (
          <span className="px-2 py-1 rounded bg-emerald-900/40 text-emerald-300 border border-emerald-700/50">
            Record: {config.stream_path}
            {(() => { const r = probeResults.find((s) => s.path === config.stream_path); return r?.width ? ` (${r.width}×${r.height})` : ""; })()}
          </span>
        )}
        {config.sub_stream_path && (
          <span className="px-2 py-1 rounded bg-blue-900/40 text-blue-300 border border-blue-700/50">
            Detect: {config.sub_stream_path}
            {(() => { const r = probeResults.find((s) => s.path === config.sub_stream_path); return r?.width ? ` (${r.width}×${r.height})` : ""; })()}
          </span>
        )}
      </div>
    </div>
  );
}

const CAMERA_TYPES = [
  {
    value: "tapo", label: "TP-Link Tapo",
    fields: ["ip", "username", "password"],
    supportsProbe: true,
  },
  {
    value: "hikvision", label: "Hikvision",
    fields: ["ip", "username", "password", "channel"],
    supportsProbe: true,
  },
  {
    value: "onvif", label: "ONVIF",
    fields: ["ip", "username", "password", "port"],
    supportsProbe: true,
  },
  { value: "rtsp", label: "RTSP URL", fields: ["rtsp_url"], supportsProbe: false },
  { value: "ring", label: "Ring", fields: ["ring_device_name"], supportsProbe: false },
  { value: "other", label: "Other", fields: ["stream_url"], supportsProbe: false },
];

export default function Cameras() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [showAdd, setShowAdd] = useState(false);
  const [editingCamera, setEditingCamera] = useState<CameraInfo | null>(null);

  const { data: cameras, isLoading } = useQuery({
    queryKey: ["cameras"],
    queryFn: () => api.get<CameraInfo[]>("/api/cameras"),
  });

  const deleteMut = useMutation({
    mutationFn: (id: number) => api.delete(`/api/cameras/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cameras"] }),
  });

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Cameras</h2>
        <button onClick={() => setShowAdd(true)} className="btn-primary text-sm py-2 flex items-center gap-1.5">
          <Plus size={16} /> Add Camera
        </button>
      </div>

      {isLoading ? (
        <div className="text-center py-12 text-slate-400">Loading...</div>
      ) : !cameras?.length ? (
        <div className="card text-center py-12 text-slate-400">
          <Video size={48} className="mx-auto mb-3 opacity-50" />
          <p>No cameras added yet</p>
        </div>
      ) : (
        <div className="space-y-3">
          {cameras.map((cam) => (
            <div key={cam.id} className="card">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Wifi size={16} className={cam.enabled ? "text-green-400" : "text-red-400"} />
                  <h3 className="font-semibold">{cam.name}</h3>
                </div>
                <div className="flex items-center gap-1.5">
                  {cam.ptz_mode && (
                    <span className="badge bg-blue-900/30 text-blue-400 text-[10px] flex items-center gap-1">
                      <Move size={10} /> PTZ
                    </span>
                  )}
                  <span className="badge bg-slate-800 text-slate-300">{cam.camera_type}</span>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 text-xs text-slate-400 mb-3">
                <span>Recording: {cam.recording_mode}</span>
                <span>·</span>
                <span>Detection: {cam.detection_enabled ? "ON" : "OFF"}</span>
                {cam.detection_enabled && (
                  <>
                    <span>·</span>
                    <span>Objects: {cam.detection_objects.join(", ")}</span>
                  </>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setEditingCamera(cam)}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <Pencil size={14} /> Edit
                </button>
                <button
                  onClick={() => navigate(`/cameras/${cam.id}/detection`)}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <SlidersHorizontal size={14} /> Detection
                </button>
                <button
                  onClick={() => {
                    if (confirm(`Delete camera "${cam.name}"?`)) deleteMut.mutate(cam.id);
                  }}
                  className="btn-danger text-xs py-1.5 px-3 flex items-center gap-1"
                >
                  <Trash2 size={14} /> Remove
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showAdd && <AddCameraModal onClose={() => setShowAdd(false)} />}
      {editingCamera && <EditCameraModal camera={editingCamera} onClose={() => setEditingCamera(null)} />}
    </div>
  );
}

export function AddCameraModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [step, setStep] = useState(0);
  const [cameraType, setCameraType] = useState("");
  const [name, setName] = useState("");
  const [config, setConfig] = useState<Record<string, string>>({});
  const [recordingMode, setRecordingMode] = useState("continuous");
  const [detectionEnabled, setDetectionEnabled] = useState(true);
  const [detectionObjects, setDetectionObjects] = useState<string[]>(["person"]);
  const [detectionConfidence, setDetectionConfidence] = useState(0.5);
  const [detectFps, setDetectFps] = useState(5);
  const [ptzMode, setPtzMode] = useState(false);
  const [ptzConfig, setPtzConfig] = useState<PtzConfig>({ enabled: false, protocol: "none" });
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [testStatus, setTestStatus] = useState<"idle" | "testing" | "success" | "failed">("idle");
  const [testMessage, setTestMessage] = useState("");
  const [probeResults, setProbeResults] = useState<ProbeResult[] | null>(null);

  // LAN Scanner state
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<{ local_ip: string; subnet: string; devices: ScanDevice[]; scanned: number } | null>(null);
  const [scanError, setScanError] = useState("");
  const [subnetPrefix, setSubnetPrefix] = useState("192.168.68");

  const runScan = async () => {
    setScanning(true);
    setScanError("");
    setScanResult(null);
    try {
      const data = await api.post<{ local_ip: string; subnet: string; devices: ScanDevice[]; scanned: number }>("/api/cameras/scan-lan", { subnet_prefix: subnetPrefix });
      setScanResult(data);
    } catch (err: any) {
      setScanError(err.message || "Scan failed");
    } finally {
      setScanning(false);
    }
  };

  const brandToType: Record<string, string> = {
    tapo: "tapo", hikvision: "hikvision", dahua: "onvif", reolink: "onvif",
    amcrest: "onvif", axis: "onvif", uniview: "onvif", ip_camera: "onvif", rtsp: "rtsp",
  };

  const brandLabels: Record<string, string> = {
    tapo: "TP-Link Tapo", hikvision: "Hikvision", dahua: "Dahua", reolink: "Reolink",
    amcrest: "Amcrest", axis: "Axis", uniview: "Uniview", ip_camera: "IP Camera",
    rtsp: "RTSP Device", xmeye: "XMEye", unknown: "Unknown",
  };

  const selectDevice = (dev: ScanDevice) => {
    const type = brandToType[dev.brand || dev.inferred_type] || "onvif";
    setCameraType(type);
    setConfig({ ip: dev.ip, username: "", password: "" });
    setStep(1);
  };

  const addMut = useMutation({
    mutationFn: (data: any) => api.post("/api/cameras", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["cameras"] });
      onClose();
    },
    onError: (err: any) => setError(err.message || "Failed to add camera"),
  });

  const testMut = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; snapshot?: string; message?: string; source_url?: string; go2rtc_status?: any }>("/api/cameras/test-connection", {
        camera_type: cameraType,
        connection_config: config,
      }),
    onSuccess: (data) => {
      if (data.success) {
        setTestStatus("success");
        setTestMessage("");
        if (data.snapshot) setPreview(data.snapshot);
      } else {
        setTestStatus("failed");
        const msg = data.message || "Connection failed";
        const urlInfo = data.source_url ? `\nSource URL: ${data.source_url}` : "";
        setTestMessage(msg + urlInfo);
        setPreview(null);
      }
    },
    onError: (err: any) => {
      setTestStatus("failed");
      setTestMessage(err.message || "Connection failed");
      setPreview(null);
    },
  });

  const probeMut = useMutation({
    mutationFn: () =>
      api.post<{ streams: ProbeResult[] }>("/api/cameras/probe-streams", {
        camera_type: cameraType,
        connection_config: config,
      }),
    onSuccess: (data) => {
      setProbeResults(data.streams);
      // Auto-select: highest resolution for record, lowest for detect
      const available = data.streams
        .filter((s) => s.available)
        .sort((a, b) => ((b.width || 0) * (b.height || 0)) - ((a.width || 0) * (a.height || 0)));
      if (available.length >= 2) {
        setConfig((c) => ({
          ...c,
          stream_path: available[0].path,
          sub_stream_path: available[available.length - 1].path,
        }));
        if (available[0].snapshot) setPreview(available[0].snapshot);
      } else if (available.length === 1) {
        setConfig((c) => ({
          ...c,
          stream_path: available[0].path,
          sub_stream_path: available[0].path,
        }));
        if (available[0].snapshot) setPreview(available[0].snapshot);
      }
    },
  });

  const typeConfig = CAMERA_TYPES.find((t) => t.value === cameraType);

  const handleTestConnection = () => {
    setTestStatus("testing");
    setPreview(null);
    setTestMessage("");
    setError("");
    testMut.mutate();
  };

  const handleProbeStreams = () => {
    setProbeResults(null);
    setPreview(null);
    probeMut.mutate();
  };

  const handleSubmit = () => {
    setError("");
    addMut.mutate({
      name,
      camera_type: cameraType,
      connection_config: config,
      recording_mode: recordingMode,
      detection_enabled: detectionEnabled,
      detection_objects: detectionObjects,
      detection_confidence: detectionConfidence,
      detection_settings: { fps: detectFps },
      ptz_mode: ptzConfig.enabled,
      ptz_config: ptzConfig.enabled ? ptzConfig : null,
    });
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-end sm:items-center justify-center z-50 p-0 sm:p-4">
      <div className="bg-slate-900 rounded-t-2xl sm:rounded-2xl w-full max-w-md max-h-[90vh] overflow-y-auto p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold">Add Camera</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        {step === 0 && (
          <div className="space-y-4">
            {/* LAN Scanner */}
            <div className="card space-y-3">
              <h4 className="font-semibold text-sm flex items-center gap-2">
                <Router size={16} className="text-blue-400" /> Scan Network
              </h4>
              <p className="text-xs text-slate-400">
                Find cameras on your LAN automatically.
              </p>
              <div className="flex gap-2">
                <input
                  className="input flex-1 text-sm"
                  value={subnetPrefix}
                  onChange={(e) => setSubnetPrefix(e.target.value)}
                  placeholder="192.168.1"
                />
                <button
                  onClick={runScan}
                  disabled={scanning}
                  className="btn-primary text-sm px-3 flex items-center gap-1.5 whitespace-nowrap"
                >
                  {scanning ? (
                    <><Loader2 size={14} className="animate-spin" /> Scanning...</>
                  ) : (
                    <><Search size={14} /> Scan</>
                  )}
                </button>
              </div>
              {scanError && <p className="text-xs text-red-400">{scanError}</p>}
              {scanResult && (
                <div className="space-y-1.5">
                  <p className="text-xs text-slate-500">
                    Found {scanResult.devices.length} device{scanResult.devices.length !== 1 ? "s" : ""} on {scanResult.subnet}
                  </p>
                  {scanResult.devices.length === 0 && (
                    <p className="text-xs text-slate-400 py-2 text-center">No cameras found. Check subnet and try again.</p>
                  )}
                  {scanResult.devices.map((dev) => (
                    <button
                      key={dev.ip}
                      onClick={() => selectDevice(dev)}
                      className="w-full flex items-center gap-2 p-2.5 bg-slate-800 rounded-lg text-left hover:bg-slate-700 transition-colors"
                    >
                      <div className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold shrink-0 ${
                        dev.has_rtsp ? "bg-blue-600/30 text-blue-400" : "bg-slate-700 text-slate-400"
                      }`}>
                        {dev.has_rtsp ? "CAM" : "DEV"}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium">
                          {dev.ip}
                          {dev.brand && (
                            <span className="ml-1.5 text-xs px-1.5 py-0.5 rounded bg-emerald-900/40 text-emerald-400">
                              {brandLabels[dev.brand] || dev.brand}
                            </span>
                          )}
                        </p>
                        <p className="text-xs text-slate-500 truncate">
                          {dev.ports.map((p) => `${p.service} (:${p.port})`).join(", ")}
                        </p>
                      </div>
                      {dev.has_rtsp && (
                        <span className="text-xs bg-blue-900/30 text-blue-400 px-1.5 py-0.5 rounded shrink-0">RTSP</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Manual camera type selection */}
            <div className="relative">
              <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-slate-700" /></div>
              <div className="relative flex justify-center"><span className="bg-slate-900 px-3 text-xs text-slate-500">or select manually</span></div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {CAMERA_TYPES.map((t) => (
                <button
                  key={t.value}
                  onClick={() => {
                    setCameraType(t.value);
                    setStep(1);
                  }}
                  className={`card text-center py-4 hover:border-blue-500 transition-colors cursor-pointer ${
                    cameraType === t.value ? "border-blue-500" : ""
                  }`}
                >
                  <Video size={24} className="mx-auto mb-2 text-blue-400" />
                  <span className="text-sm font-medium">{t.label}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {step === 1 && typeConfig && (
          <div className="space-y-3">
            <div>
              <label className="label">Camera Name</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Front Door" />
            </div>
            {typeConfig.fields.map((field) => (
              <div key={field}>
                <label className="label">{field.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</label>
                <input
                  className="input"
                  type={field === "password" ? "password" : "text"}
                  value={config[field] || ""}
                  onChange={(e) => setConfig({ ...config, [field]: e.target.value })}
                  placeholder={field === "port" ? "554" : field === "channel" ? "101" : ""}
                />
              </div>
            ))}

            {/* Stream Discovery */}
            {typeConfig.supportsProbe && (
              <div className="space-y-3">
                <button
                  onClick={handleProbeStreams}
                  disabled={probeMut.isPending || !config.ip}
                  className="btn-secondary w-full flex items-center justify-center gap-2"
                  title="Scan all RTSP stream paths and capture snapshots"
                >
                  <Search size={16} />
                  {probeMut.isPending ? "Scanning streams…" : "Discover Streams"}
                </button>

                {/* Probe Results Grid */}
                {probeResults && (
                  <ProbeResultsGrid
                    probeResults={probeResults}
                    setProbeResults={setProbeResults}
                    config={config}
                    setConfig={setConfig}
                    setPreview={setPreview}
                    cameraType={cameraType}
                  />
                )}
              </div>
            )}

            {/* Test Connection */}
            <button
              onClick={handleTestConnection}
              disabled={testMut.isPending}
              className="btn-secondary w-full flex items-center justify-center gap-2"
            >
              <TestTube size={16} />
              {testMut.isPending ? "Testing..." : "Test Connection"}
            </button>

            {testStatus === "success" && (
              <div className="space-y-2">
                <p className="text-green-400 text-sm flex items-center gap-1">
                  <Wifi size={14} /> Connection successful
                </p>
                {preview && (
                  <img
                    src={`data:image/jpeg;base64,${preview}`}
                    alt="Camera preview"
                    className="w-full rounded-lg border border-slate-700"
                  />
                )}
              </div>
            )}

            {testStatus === "failed" && (
              <div className="space-y-1">
                <p className="text-red-400 text-sm">Connection failed</p>
                {testMessage && testMessage.split("\n").map((line, i) => (
                  <p key={i} className="text-xs text-slate-500 break-all">{line}</p>
                ))}
              </div>
            )}

            <div className="flex gap-2">
              <button onClick={() => { setStep(0); setTestStatus("idle"); setPreview(null); setProbeResults(null); }} className="btn-secondary flex-1">
                Back
              </button>
              <button onClick={() => setStep(2)} className="btn-primary flex-1" disabled={!name}>
                Next
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-3">
            <div>
              <label className="label">Recording Mode</label>
              <select className="input" value={recordingMode} onChange={(e) => setRecordingMode(e.target.value)}>
                <option value="continuous">Continuous</option>
                <option value="events">Events Only</option>
                <option value="motion">Motion Only</option>
                <option value="disabled">Disabled</option>
              </select>
            </div>

            <DetectionSettingsSection
              detectionEnabled={detectionEnabled}
              setDetectionEnabled={setDetectionEnabled}
              detectionObjects={detectionObjects}
              setDetectionObjects={setDetectionObjects}
              detectionConfidence={detectionConfidence}
              setDetectionConfidence={setDetectionConfidence}
              detectFps={detectFps}
              setDetectFps={setDetectFps}
            />

            <PtzSettings
              cameraType={cameraType}
              ptzConfig={ptzConfig}
              onChange={setPtzConfig}
              connectionConfig={config}
            />

            {error && <p className="text-red-400 text-sm">{error}</p>}

            <div className="flex gap-2">
              <button onClick={() => setStep(1)} className="btn-secondary flex-1">
                Back
              </button>
              <button onClick={handleSubmit} className="btn-primary flex-1" disabled={addMut.isPending}>
                {addMut.isPending ? "Adding..." : "Add Camera"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════ Detection Settings Section ═══════════════════════ */

const DETECTION_OBJECTS = ["person", "cat", "dog", "car", "bird", "bicycle", "motorcycle", "bus", "truck", "boat"];

function DetectionSettingsSection({
  detectionEnabled,
  setDetectionEnabled,
  detectionObjects,
  setDetectionObjects,
  detectionConfidence,
  setDetectionConfidence,
  detectFps,
  setDetectFps,
}: {
  detectionEnabled: boolean;
  setDetectionEnabled: (v: boolean) => void;
  detectionObjects: string[];
  setDetectionObjects: (v: string[]) => void;
  detectionConfidence: number;
  setDetectionConfidence: (v: number) => void;
  detectFps: number;
  setDetectFps: (v: number) => void;
}) {
  return (
    <div className="card bg-slate-800/40 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm font-medium">
          <Eye size={14} className={detectionEnabled ? "text-emerald-400" : "text-slate-500"} />
          Object Detection
        </label>
        <button
          type="button"
          onClick={() => setDetectionEnabled(!detectionEnabled)}
          className={`w-10 h-5 rounded-full transition-colors ${detectionEnabled ? "bg-emerald-600" : "bg-slate-700"}`}
        >
          <div className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${detectionEnabled ? "translate-x-5" : ""}`} />
        </button>
      </div>

      {detectionEnabled && (
        <>
          <div>
            <label className="label text-xs">Detect Objects</label>
            <div className="flex flex-wrap gap-1.5">
              {DETECTION_OBJECTS.map((obj) => {
                const selected = detectionObjects.includes(obj);
                return (
                  <button
                    key={obj}
                    type="button"
                    onClick={() => {
                      if (selected) {
                        if (detectionObjects.length > 1) setDetectionObjects(detectionObjects.filter((o) => o !== obj));
                      } else {
                        setDetectionObjects([...detectionObjects, obj]);
                      }
                    }}
                    className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                      selected
                        ? "bg-emerald-600/30 text-emerald-300 border border-emerald-500/40"
                        : "bg-slate-700 text-slate-400 border border-slate-600 hover:border-slate-500"
                    }`}
                  >
                    {obj}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="label text-xs mb-0">Confidence Threshold</label>
              <span className="text-xs text-slate-400 font-mono">{Math.round(detectionConfidence * 100)}%</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={0.95}
              step={0.05}
              value={detectionConfidence}
              onChange={(e) => setDetectionConfidence(parseFloat(e.target.value))}
              className="w-full mt-1 accent-emerald-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>Sensitive</span>
              <span>Strict</span>
            </div>
          </div>

          <div>
            <label className="label text-xs">Detect FPS</label>
            <select
              className="input text-sm"
              value={detectFps}
              onChange={(e) => setDetectFps(parseInt(e.target.value))}
            >
              <option value={3}>3 fps (Low power)</option>
              <option value={5}>5 fps (Recommended)</option>
              <option value={10}>10 fps (High)</option>
              <option value={15}>15 fps (Max)</option>
            </select>
            <p className="text-[10px] text-slate-500 mt-0.5">Higher FPS uses more CPU. 5 is recommended for most cameras.</p>
          </div>
        </>
      )}
    </div>
  );
}

/* ═══════════════════════ PTZ Settings Component ═══════════════════════ */

const PTZ_PROTOCOLS: Record<string, { label: string; types: string[]; desc: string }> = {
  onvif: { label: "ONVIF", types: ["hikvision", "onvif", "other"], desc: "Standard IP camera PTZ via ONVIF — Frigate autotracking" },
  tapo: { label: "Tapo", types: ["tapo"], desc: "TP-Link Tapo pan/tilt — preset positions only (no autotrack)" },
  none: { label: "None", types: [], desc: "Fixed camera — no PTZ" },
};

function PtzSettings({
  cameraType,
  ptzConfig,
  onChange,
  connectionConfig,
}: {
  cameraType: string;
  ptzConfig: PtzConfig;
  onChange: (c: PtzConfig) => void;
  connectionConfig: Record<string, string>;
}) {
  // Ring cameras have no PTZ
  const isRing = cameraType === "ring";
  // Determine available protocols for this camera type
  const availableProtocols = Object.entries(PTZ_PROTOCOLS).filter(
    ([key, p]) => key === "none" || p.types.includes(cameraType)
  );

  // Auto-populate ONVIF fields from connection config
  const autoFillOnvif = () => {
    onChange({
      ...ptzConfig,
      onvif_host: connectionConfig.ip || connectionConfig.host || ptzConfig.onvif_host || "",
      onvif_user: connectionConfig.username || ptzConfig.onvif_user || "admin",
      onvif_password: connectionConfig.password || ptzConfig.onvif_password || "",
      onvif_port: ptzConfig.onvif_port || (cameraType === "tapo" ? 2020 : cameraType === "hikvision" ? 8000 : 80),
    });
  };

  if (isRing) {
    return (
      <div className="card bg-slate-800/40 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Move size={14} />
          <span>PTZ</span>
          <span className="text-xs px-1.5 py-0.5 rounded bg-slate-700 text-slate-500">Not available</span>
        </div>
        <p className="text-xs text-slate-500 mt-1">Ring cameras are fixed — PTZ is not supported.</p>
      </div>
    );
  }

  return (
    <div className="card bg-slate-800/40 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm font-medium">
          <Move size={14} className={ptzConfig.enabled ? "text-blue-400" : "text-slate-500"} />
          PTZ Control
        </label>
        <button
          type="button"
          onClick={() => onChange({ ...ptzConfig, enabled: !ptzConfig.enabled, protocol: !ptzConfig.enabled ? (cameraType === "tapo" ? "tapo" : "onvif") : "none" })}
          className={`w-10 h-5 rounded-full transition-colors ${ptzConfig.enabled ? "bg-blue-600" : "bg-slate-700"}`}
        >
          <div className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${ptzConfig.enabled ? "translate-x-5" : ""}`} />
        </button>
      </div>

      {ptzConfig.enabled && (
        <>
          {/* Protocol selector */}
          {availableProtocols.length > 1 && (
            <div>
              <label className="label text-xs">Protocol</label>
              <div className="flex gap-2">
                {availableProtocols.filter(([k]) => k !== "none").map(([key, p]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => {
                      onChange({ ...ptzConfig, protocol: key as PtzConfig["protocol"] });
                      if (key === "onvif") autoFillOnvif();
                    }}
                    className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                      ptzConfig.protocol === key
                        ? "bg-blue-600 text-white"
                        : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-slate-500 mt-1">
                {PTZ_PROTOCOLS[ptzConfig.protocol]?.desc}
              </p>
            </div>
          )}

          {/* ONVIF Settings */}
          {ptzConfig.protocol === "onvif" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400">ONVIF Connection</span>
                <button
                  type="button"
                  onClick={autoFillOnvif}
                  className="text-[10px] text-blue-400 hover:text-blue-300"
                >
                  Auto-fill from camera
                </button>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className="col-span-2">
                  <input
                    className="input text-sm"
                    placeholder="ONVIF Host / IP"
                    value={ptzConfig.onvif_host || ""}
                    onChange={(e) => onChange({ ...ptzConfig, onvif_host: e.target.value })}
                  />
                </div>
                <div>
                  <input
                    className="input text-sm"
                    placeholder="Port"
                    type="number"
                    value={ptzConfig.onvif_port || ""}
                    onChange={(e) => onChange({ ...ptzConfig, onvif_port: parseInt(e.target.value) || 80 })}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <input
                  className="input text-sm"
                  placeholder="Username"
                  value={ptzConfig.onvif_user || ""}
                  onChange={(e) => onChange({ ...ptzConfig, onvif_user: e.target.value })}
                />
                <input
                  className="input text-sm"
                  type="password"
                  placeholder="Password"
                  value={ptzConfig.onvif_password || ""}
                  onChange={(e) => onChange({ ...ptzConfig, onvif_password: e.target.value })}
                />
              </div>

              {/* Autotracking */}
              <div className="border-t border-slate-700 pt-2 mt-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-slate-400">Frigate Autotracking</label>
                  <button
                    type="button"
                    onClick={() => onChange({ ...ptzConfig, autotrack_enabled: !ptzConfig.autotrack_enabled })}
                    className={`w-9 h-4.5 rounded-full transition-colors ${ptzConfig.autotrack_enabled ? "bg-blue-600" : "bg-slate-700"}`}
                  >
                    <div className={`w-3.5 h-3.5 bg-white rounded-full transition-transform mx-0.5 ${ptzConfig.autotrack_enabled ? "translate-x-4" : ""}`} />
                  </button>
                </div>
                <p className="text-[10px] text-slate-500 mt-0.5">
                  Camera automatically follows detected objects using ONVIF PTZ commands
                </p>

                {ptzConfig.autotrack_enabled && (
                  <div className="mt-2 space-y-2">
                    <div>
                      <label className="label text-xs">Track Objects</label>
                      <div className="flex flex-wrap gap-1.5">
                        {["person", "cat", "dog", "car"].map((obj) => {
                          const selected = (ptzConfig.autotrack_objects || ["person"]).includes(obj);
                          return (
                            <button
                              key={obj}
                              type="button"
                              onClick={() => {
                                const current = ptzConfig.autotrack_objects || ["person"];
                                onChange({
                                  ...ptzConfig,
                                  autotrack_objects: selected
                                    ? current.filter((o) => o !== obj)
                                    : [...current, obj],
                                });
                              }}
                              className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                                selected
                                  ? "bg-blue-600/30 text-blue-300 border border-blue-500/40"
                                  : "bg-slate-700 text-slate-400 border border-slate-600 hover:border-slate-500"
                              }`}
                            >
                              {obj}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <div>
                      <label className="label text-xs">Return timeout (seconds)</label>
                      <input
                        className="input text-sm w-24"
                        type="number"
                        min={5}
                        max={300}
                        value={ptzConfig.autotrack_timeout || 30}
                        onChange={(e) => onChange({ ...ptzConfig, autotrack_timeout: parseInt(e.target.value) || 30 })}
                      />
                      <p className="text-[10px] text-slate-500 mt-0.5">
                        Seconds before camera returns to home position after losing track
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tapo Settings */}
          {ptzConfig.protocol === "tapo" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400">ONVIF Connection (port 2020)</span>
                <button
                  type="button"
                  onClick={autoFillOnvif}
                  className="text-[10px] text-blue-400 hover:text-blue-300"
                >
                  Auto-fill from camera
                </button>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className="col-span-2">
                  <input
                    className="input text-sm"
                    placeholder="Host / IP"
                    value={ptzConfig.onvif_host || ""}
                    onChange={(e) => onChange({ ...ptzConfig, onvif_host: e.target.value })}
                  />
                </div>
                <div>
                  <input
                    className="input text-sm"
                    placeholder="Port"
                    type="number"
                    value={ptzConfig.onvif_port || ""}
                    onChange={(e) => onChange({ ...ptzConfig, onvif_port: parseInt(e.target.value) || 2020 })}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <input
                  className="input text-sm"
                  placeholder="Username"
                  value={ptzConfig.onvif_user || ""}
                  onChange={(e) => onChange({ ...ptzConfig, onvif_user: e.target.value })}
                />
                <input
                  className="input text-sm"
                  type="password"
                  placeholder="Password"
                  value={ptzConfig.onvif_password || ""}
                  onChange={(e) => onChange({ ...ptzConfig, onvif_password: e.target.value })}
                />
              </div>
              <div className="bg-amber-900/20 border border-amber-700/30 rounded-lg p-2.5">
                <p className="text-xs text-amber-300">
                  Tapo cameras use ONVIF on port 2020 for PTZ control.
                </p>
                <p className="text-[10px] text-amber-400/70 mt-1">
                  Frigate autotracking is not available — preset positions can be set in the Tapo app.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function EditCameraModal({ camera, onClose }: { camera: CameraInfo; onClose: () => void }) {
  const qc = useQueryClient();
  const [name, setName] = useState(camera.name);
  const [config, setConfig] = useState<Record<string, string>>(camera.connection_config || {});
  const [recordingMode, setRecordingMode] = useState(camera.recording_mode);
  const [detectionEnabled, setDetectionEnabled] = useState(camera.detection_enabled);
  const [detectionObjects, setDetectionObjects] = useState<string[]>(camera.detection_objects || ["person"]);
  const [detectionConfidence, setDetectionConfidence] = useState(camera.detection_confidence || 0.5);
  const [detectFps, setDetectFps] = useState((camera.detection_settings as any)?.fps || 5);
  const [ptzConfig, setPtzConfig] = useState<PtzConfig>(
    camera.ptz_config || { enabled: camera.ptz_mode, protocol: camera.ptz_mode ? "onvif" : "none" }
  );
  const [enabled, setEnabled] = useState(camera.enabled);
  const [error, setError] = useState("");
  const [probeResults, setProbeResults] = useState<ProbeResult[] | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const typeConfig = CAMERA_TYPES.find((t) => t.value === camera.camera_type);

  const updateMut = useMutation({
    mutationFn: (data: any) => api.put(`/api/cameras/${camera.id}`, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["cameras"] });
      onClose();
    },
    onError: (err: any) => setError(err.message || "Failed to update camera"),
  });

  const testMut = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; snapshot?: string; message?: string; source_url?: string; go2rtc_status?: any }>("/api/cameras/test-connection", {
        camera_type: camera.camera_type,
        connection_config: config,
      }),
    onSuccess: (data) => {
      if (data.success && data.snapshot) setPreview(data.snapshot);
    },
  });

  const probeMut = useMutation({
    mutationFn: () =>
      api.post<{ streams: ProbeResult[] }>("/api/cameras/probe-streams", {
        camera_type: camera.camera_type,
        connection_config: config,
      }),
    onSuccess: (data) => {
      setProbeResults(data.streams);
      const available = data.streams
        .filter((s) => s.available)
        .sort((a, b) => ((b.width || 0) * (b.height || 0)) - ((a.width || 0) * (a.height || 0)));
      if (available.length >= 2) {
        setConfig((c) => ({
          ...c,
          stream_path: c.stream_path || available[0].path,
          sub_stream_path: c.sub_stream_path || available[available.length - 1].path,
        }));
      } else if (available.length === 1) {
        setConfig((c) => ({
          ...c,
          stream_path: c.stream_path || available[0].path,
          sub_stream_path: c.sub_stream_path || available[0].path,
        }));
      }
    },
  });

  const handleSave = () => {
    setError("");
    updateMut.mutate({
      name,
      connection_config: config,
      recording_mode: recordingMode,
      detection_enabled: detectionEnabled,
      detection_objects: detectionObjects,
      detection_confidence: detectionConfidence,
      detection_settings: { ...(camera.detection_settings || {}), fps: detectFps },
      ptz_mode: ptzConfig.enabled,
      ptz_config: ptzConfig.enabled ? ptzConfig : null,
      enabled,
    });
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-end sm:items-center justify-center z-50 p-0 sm:p-4">
      <div className="bg-slate-900 rounded-t-2xl sm:rounded-2xl w-full max-w-md max-h-[90vh] overflow-y-auto p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold">Edit Camera</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        <div>
          <label className="label">Camera Name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
        </div>

        <div className="text-xs text-slate-500">Type: {camera.camera_type}</div>

        {typeConfig?.fields.map((field) => (
          <div key={field}>
            <label className="label">
              {field.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
            </label>
            <input
              className="input"
              type={field === "password" ? "password" : "text"}
              value={config[field] || ""}
              onChange={(e) => setConfig({ ...config, [field]: e.target.value })}
            />
          </div>
        ))}

        {/* Stream Discovery */}
        {typeConfig?.supportsProbe && (
          <div className="space-y-3">
            <button
              onClick={() => probeMut.mutate()}
              disabled={probeMut.isPending || !config.ip}
              className="btn-secondary w-full flex items-center justify-center gap-2"
            >
              <Search size={16} />
              {probeMut.isPending ? "Scanning streams…" : "Discover Streams"}
            </button>

            {probeResults && (
              <ProbeResultsGrid
                probeResults={probeResults}
                setProbeResults={setProbeResults}
                config={config}
                setConfig={setConfig}
                setPreview={setPreview}
                cameraType={camera.camera_type}
              />
            )}
          </div>
        )}

        {/* Current stream info (when no probe) */}
        {!probeResults && config.stream_path && (
          <div className="flex gap-2 text-xs">
            <span className="px-2 py-1 rounded bg-emerald-900/40 text-emerald-300 border border-emerald-700/50">
              Record: {config.stream_path}
            </span>
            {config.sub_stream_path && (
              <span className="px-2 py-1 rounded bg-blue-900/40 text-blue-300 border border-blue-700/50">
                Detect: {config.sub_stream_path}
              </span>
            )}
          </div>
        )}

        <button
          onClick={() => testMut.mutate()}
          disabled={testMut.isPending}
          className="btn-secondary w-full flex items-center justify-center gap-2"
        >
          <TestTube size={16} />
          {testMut.isPending ? "Testing..." : "Test Connection"}
        </button>

        {testMut.isSuccess && (
          <div className="space-y-1">
            <p className={`text-sm ${testMut.data.success ? "text-green-400" : "text-red-400"}`}>
              {testMut.data.success ? "Connection successful" : testMut.data.message || "Connection failed"}
            </p>
            {!testMut.data.success && testMut.data.source_url && (
              <p className="text-xs text-slate-500 break-all">Source URL: {testMut.data.source_url}</p>
            )}
          </div>
        )}

        {preview && (
          <img
            src={`data:image/jpeg;base64,${preview}`}
            alt="Preview"
            className="w-full rounded-lg border border-slate-700"
          />
        )}

        <div className="flex items-center justify-between">
          <label className="label mb-0">Enabled</label>
          <button
            onClick={() => setEnabled(!enabled)}
            className={`w-12 h-6 rounded-full transition-colors ${enabled ? "bg-blue-600" : "bg-slate-700"}`}
          >
            <div
              className={`w-5 h-5 bg-white rounded-full transition-transform mx-0.5 ${
                enabled ? "translate-x-6" : ""
              }`}
            />
          </button>
        </div>

        <div>
          <label className="label">Recording Mode</label>
          <select className="input" value={recordingMode} onChange={(e) => setRecordingMode(e.target.value)}>
            <option value="continuous">Continuous</option>
            <option value="events">Events Only</option>
            <option value="motion">Motion Only</option>
            <option value="disabled">Disabled</option>
          </select>
        </div>

        <DetectionSettingsSection
          detectionEnabled={detectionEnabled}
          setDetectionEnabled={setDetectionEnabled}
          detectionObjects={detectionObjects}
          setDetectionObjects={setDetectionObjects}
          detectionConfidence={detectionConfidence}
          setDetectionConfidence={setDetectionConfidence}
          detectFps={detectFps}
          setDetectFps={setDetectFps}
        />

        <PtzSettings
          cameraType={camera.camera_type}
          ptzConfig={ptzConfig}
          onChange={setPtzConfig}
          connectionConfig={config}
        />

        {error && <p className="text-red-400 text-sm">{error}</p>}

        <button onClick={handleSave} className="btn-primary w-full" disabled={updateMut.isPending}>
          {updateMut.isPending ? "Saving..." : "Save Changes"}
        </button>
      </div>
    </div>
  );
}
