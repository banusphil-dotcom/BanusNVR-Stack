import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api, getToken } from "../api";
import { useAuth } from "../hooks/useAuth";
import {
  Bell, Mail, Shield, HardDrive, Save, Key, Cpu, MemoryStick,
  Monitor, RefreshCw, Smartphone, LogOut,
  Activity, Clock, Camera, ChevronDown, ChevronRight, Zap, Leaf, Scale,
  SlidersHorizontal, RotateCcw, Server, Wifi, WifiOff,
  Usb, ArrowLeftRight, AlertCircle, Timer, Hash,
  Radio, Sun, Moon, Palette, ExternalLink,
} from "lucide-react";

interface GlobalSettings {
  retention_events_days: number;
  retention_continuous_days: number;
  retention_snapshots_days: number;
  smtp_host: string;
  smtp_port: number;
  smtp_user: string;
  smtp_from: string;
}

interface HardwareResources {
  cpu_name: string;
  cpu_cores: number;
  cpu_percent: number;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_percent: number;
  gpu_available: boolean;
  gpu_name?: string;
  gpu_percent?: number;
  gpu_inference_device?: string;
  coral_available: boolean;
  coral_status?: { available: boolean; active_model?: string; swap_count: number; last_swap_ms: number; yolo_input_size: number; cnn_embed_dim?: number } | null;
  detector_type?: string | null;
  detector_devices?: string[];
  storage_used_gb: number;
  storage_total_gb: number;
  storage_percent: number;
  cameras_active: number;
  cameras_relay: number;
  cameras_transcode: number;
  estimated_max_cameras_relay: number;
  estimated_max_cameras_transcode: number;
  uptime_seconds: number;
  warnings: { level: string; category: string; message: string; value: number; limit?: number }[];
}

interface PerformanceSettings {
  motion_frame_skip: number;
  motion_blur_kernel: number;
  motion_cooldown: number;
  track_interval: number;
  enhanced_scan_interval: number;
  yolo_concurrency: number;
  max_detection_pipelines: number;
  jpeg_quality: number;
  preset: string | null;
  ml_offload_enabled: boolean;
  ml_offload_url: string;
  coral_enabled: boolean;
}

interface MLHealthInfo {
  enabled: boolean;
  online?: boolean;
  gpu?: string;
  models?: Record<string, boolean>;
  error?: string;
}

const PRESETS: { key: string; label: string; icon: React.ReactNode; desc: string; color: string; values: Omit<PerformanceSettings, "preset"> }[] = [
  {
    key: "performance", label: "Performance", icon: <Zap size={14} />, desc: "More responsive detection, higher CPU",
    color: "text-amber-400 border-amber-500/40 bg-amber-500/10",
    values: { motion_frame_skip: 6, motion_blur_kernel: 7, motion_cooldown: 3, track_interval: 2, enhanced_scan_interval: 10, yolo_concurrency: 2, max_detection_pipelines: 3, jpeg_quality: 95, ml_offload_enabled: false, ml_offload_url: "https://ml.banusphotos.com", coral_enabled: false },
  },
  {
    key: "balanced", label: "Balanced", icon: <Scale size={14} />, desc: "Recommended — good detection, low load",
    color: "text-blue-400 border-blue-500/40 bg-blue-500/10",
    values: { motion_frame_skip: 10, motion_blur_kernel: 7, motion_cooldown: 5, track_interval: 3, enhanced_scan_interval: 30, yolo_concurrency: 1, max_detection_pipelines: 2, jpeg_quality: 80, ml_offload_enabled: false, ml_offload_url: "https://ml.banusphotos.com", coral_enabled: false },
  },
  {
    key: "eco", label: "Eco", icon: <Leaf size={14} />, desc: "Minimal resources, reduced detection",
    color: "text-emerald-400 border-emerald-500/40 bg-emerald-500/10",
    values: { motion_frame_skip: 15, motion_blur_kernel: 9, motion_cooldown: 8, track_interval: 5, enhanced_scan_interval: 60, yolo_concurrency: 1, max_detection_pipelines: 1, jpeg_quality: 65, ml_offload_enabled: false, ml_offload_url: "https://ml.banusphotos.com", coral_enabled: false },
  },
];

export default function Settings() {
  const [showAdvanced, setShowAdvanced] = useState(false);
  return (
    <div className="p-4 space-y-6 pb-24 max-w-2xl mx-auto">
      <h2 className="text-lg font-bold">Settings</h2>
      <ResourceMonitor />
      <MLServerSettings />
      <RingSettings />
      <NotificationSettings />
      <StorageSettings />
      <AccountSettings />
      <ThemeSettings />
      <AppSettings />

      <button
        onClick={() => setShowAdvanced((v) => !v)}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-slate-800/50 hover:bg-slate-800 border border-slate-700 text-sm text-slate-300"
      >
        <SlidersHorizontal size={14} />
        {showAdvanced ? "Hide Advanced" : "Show Advanced"}
        {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>

      {showAdvanced && (
        <div className="space-y-6">
          <div className="text-xs text-slate-500 px-2">
            Power-user controls. Defaults are tuned for the current hardware — change only if you know what you're doing.
          </div>
          <AdvancedSettings />
          <TrainingReinforcementSettings />
          <CoralStatus />
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ Coral Edge TPU Status ═══════════════════════ */

interface CoralStatusData {
  enabled: boolean;
  available: boolean;
  active_model?: string;
  swap_count?: number;
  last_swap_ms?: number;
  yolo_input_size?: number;
  cnn_embed_dim?: number | null;
  detect_count?: number;
  detect_avg_ms?: number;
  detect_last_ms?: number;
  cnn_count?: number;
  cnn_avg_ms?: number;
  cnn_last_ms?: number;
  uptime_seconds?: number;
  last_error?: string | null;
  error?: string;
}

function CoralStatus() {
  const [expanded, setExpanded] = useState(false);

  const { data: coral } = useQuery({
    queryKey: ["coral-status"],
    queryFn: () => api.get<CoralStatusData>("/api/system/coral"),
    refetchInterval: expanded ? 5000 : 30000,
  });

  if (!coral || !coral.enabled) return null;

  const formatMs = (ms: number) => ms < 1 ? "<1ms" : `${ms.toFixed(1)}ms`;
  const formatUptime = (s: number) => {
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (d > 0) return `${d}d ${h}h`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  };

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <Usb size={16} className="text-teal-400" /> AI Detector
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-bold ${
            coral.available
              ? "bg-teal-500/20 text-teal-400"
              : "bg-red-500/20 text-red-400"
          }`}>
            {coral.available ? "ONLINE" : "OFFLINE"}
          </span>
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>

      {expanded && (
        <div className="card p-4 space-y-4">
          {!coral.available ? (
            <div className="text-center py-4">
              <AlertCircle size={24} className="mx-auto mb-2 text-red-400 opacity-60" />
              <p className="text-sm text-red-300">Detector not available</p>
              {coral.error && <p className="text-xs text-slate-500 mt-1">{coral.error}</p>}
              {coral.last_error && <p className="text-xs text-slate-500 mt-1">{coral.last_error}</p>}
            </div>
          ) : (
            <>
              {/* Status overview */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-slate-800/60 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Active Model</div>
                  <div className="text-sm font-bold text-teal-300 capitalize">{coral.active_model || "—"}</div>
                </div>
                <div className="bg-slate-800/60 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Uptime</div>
                  <div className="text-sm font-bold text-white">{coral.uptime_seconds ? formatUptime(coral.uptime_seconds) : "—"}</div>
                </div>
                <div className="bg-slate-800/60 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Model Swaps</div>
                  <div className="text-sm font-bold text-white">{coral.swap_count?.toLocaleString() ?? "—"}</div>
                </div>
              </div>

              {/* Detection stats */}
              <div className="space-y-2">
                <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Camera size={11} /> Object Detection (SSD MobileNet V2)
                </h4>
                <div className="bg-slate-800/40 rounded-lg p-3">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-[10px] text-slate-500">Inferences</div>
                      <div className="text-lg font-bold text-white tabular-nums">{coral.detect_count?.toLocaleString() ?? 0}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-slate-500">Avg Latency</div>
                      <div className={`text-lg font-bold tabular-nums ${
                        (coral.detect_avg_ms ?? 0) > 100 ? "text-amber-400" : "text-emerald-400"
                      }`}>
                        {coral.detect_avg_ms ? formatMs(coral.detect_avg_ms) : "—"}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-slate-500">Last</div>
                      <div className="text-lg font-bold text-slate-300 tabular-nums">
                        {coral.detect_last_ms ? formatMs(coral.detect_last_ms) : "—"}
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 text-[10px] text-slate-600 text-center">
                    Input: {coral.yolo_input_size ?? 300}×{coral.yolo_input_size ?? 300} INT8
                  </div>
                </div>
              </div>

              {/* CNN stats */}
              <div className="space-y-2">
                <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Cpu size={11} /> CNN Features (MobileNet V2)
                </h4>
                <div className="bg-slate-800/40 rounded-lg p-3">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-[10px] text-slate-500">Inferences</div>
                      <div className="text-lg font-bold text-white tabular-nums">{coral.cnn_count?.toLocaleString() ?? 0}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-slate-500">Avg Latency</div>
                      <div className={`text-lg font-bold tabular-nums ${
                        (coral.cnn_avg_ms ?? 0) > 50 ? "text-amber-400" : "text-emerald-400"
                      }`}>
                        {coral.cnn_avg_ms ? formatMs(coral.cnn_avg_ms) : "—"}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-slate-500">Last</div>
                      <div className="text-lg font-bold text-slate-300 tabular-nums">
                        {coral.cnn_last_ms ? formatMs(coral.cnn_last_ms) : "—"}
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 text-[10px] text-slate-600 text-center">
                    Output: {coral.cnn_embed_dim ?? 1001}-dim &middot; 224×224 INT8
                  </div>
                </div>
              </div>

              {/* Model swap info */}
              <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                <span className="flex items-center gap-1">
                  <ArrowLeftRight size={11} /> Last swap: {coral.last_swap_ms ? formatMs(coral.last_swap_ms) : "—"}
                </span>
                <span className="flex items-center gap-1">
                  <Timer size={11} /> {(coral.detect_avg_ms ?? 0) > 0 ? `~${Math.round(1000 / (coral.detect_avg_ms!))} det/s` : "—"}
                </span>
              </div>

              {/* Error display */}
              {coral.last_error && (
                <div className="text-xs px-3 py-2 rounded-lg bg-red-900/20 text-red-300 border border-red-800/40">
                  <AlertCircle size={12} className="inline mr-1.5 -mt-0.5" />
                  {coral.last_error}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ Resource Monitor (Task Manager Style) ═══════════════════════ */

function DetectorIndicator({ hw }: { hw: HardwareResources }) {
  const devices = hw.detector_devices && hw.detector_devices.length
    ? hw.detector_devices
    : [hw.gpu_available ? (hw.gpu_name || "Intel iGPU") + " (OpenVINO)" : "CPU"];
  const isCoral = hw.detector_type === "edgetpu" || devices.some((d) => d.toLowerCase().includes("coral"));
  const accentRing = isCoral ? "bg-teal-500/15 text-teal-300 border-teal-700/40" : "bg-blue-500/15 text-blue-300 border-blue-800/40";
  const Icon = isCoral ? Usb : Zap;
  const [showHelp, setShowHelp] = useState(false);
  return (
    <div className="card py-3 px-4 space-y-2">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <span className="text-xs font-medium flex items-center gap-1.5">
          <Icon size={13} className={isCoral ? "text-teal-400" : "text-blue-400"} />
          AI Detector
        </span>
        <div className="flex items-center gap-1.5 flex-wrap justify-end">
          {devices.map((d) => (
            <span key={d} className={`text-[10px] px-2 py-0.5 rounded-full border ${accentRing}`}>{d}</span>
          ))}
          {isCoral && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-700/40 font-bold">
              ACCELERATED
            </span>
          )}
        </div>
      </div>
      {!isCoral && (
        <>
          <button
            onClick={() => setShowHelp((v) => !v)}
            className="text-[10px] text-slate-500 hover:text-slate-300 flex items-center gap-1"
          >
            {showHelp ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
            No Coral TPU detected — how to add one
          </button>
          {showHelp && (
            <div className="text-[11px] text-slate-400 bg-slate-900/60 border border-slate-800 rounded p-2.5 space-y-1.5">
              <p>To enable hardware-accelerated detection, plug in a Google Coral USB Accelerator <em>or</em> install a PCIe / M.2 Coral, then:</p>
              <ol className="list-decimal list-inside space-y-0.5 text-slate-500">
                <li>Install the EdgeTPU runtime on the host: <code className="text-slate-300">sudo apt install libedgetpu1-std</code> (USB) or <code className="text-slate-300">gasket-dkms</code> (M.2)</li>
                <li>For USB Coral, ensure compose maps it: <code className="text-slate-300">devices: ["/dev/bus/usb:/dev/bus/usb"]</code> (already set by default)</li>
                <li>For M.2 Coral, add <code className="text-slate-300">/dev/apex_0:/dev/apex_0</code> to the <code className="text-slate-300">frigate</code> service (it runs <code className="text-slate-300">privileged</code> so usually no edit needed)</li>
                <li>Restart the stack: <code className="text-slate-300">docker compose restart api frigate</code></li>
              </ol>
              <p className="text-slate-500">Detection is automatic — the API regenerates Frigate config and picks the TPU on next start.</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ════════════════ ML Offload Server (top-level) ══════════════ */

function MLServerSettings() {
  const qc = useQueryClient();
  const { data: perf } = useQuery({
    queryKey: ["performance-settings"],
    queryFn: () => api.get<PerformanceSettings>("/api/system/performance"),
  });
  const mut = useMutation({
    mutationFn: (data: PerformanceSettings) => api.put("/api/system/performance", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["performance-settings"] });
      qc.invalidateQueries({ queryKey: ["ml-health"] });
    },
  });
  const [form, setForm] = useState<{ enabled: boolean; url: string } | null>(null);
  const enabled = form?.enabled ?? perf?.ml_offload_enabled ?? false;
  const url = form?.url ?? perf?.ml_offload_url ?? "https://ml.banusphotos.com";
  const dirty = form !== null && perf !== undefined && (form.enabled !== perf.ml_offload_enabled || form.url !== perf.ml_offload_url);

  const save = () => {
    if (!perf) return;
    mut.mutate({ ...perf, ml_offload_enabled: enabled, ml_offload_url: url });
    setForm(null);
  };

  return (
    <div className="space-y-3">
      <h3 className="font-semibold flex items-center gap-2">
        <Server size={16} className="text-indigo-400" /> ML Offload Server
      </h3>
      <div className="card p-4 space-y-4">
        <p className="text-xs text-slate-400">
          Optional remote GPU server for heavy ML workloads (face recognition, person re-identification, deep search). Local detection still runs in Frigate — this only offloads enrichment.
        </p>
        <MLOffloadSection
          enabled={enabled}
          url={url}
          onToggle={(v) => setForm({ enabled: v, url })}
          onUrlChange={(v) => setForm({ enabled, url: v })}
        />
        {dirty && (
          <div className="flex items-center gap-2 pt-2 border-t border-slate-800">
            <button
              onClick={save}
              disabled={mut.isPending}
              className="px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-xs font-medium text-white flex items-center gap-1.5"
            >
              <Save size={12} /> Save
            </button>
            <button
              onClick={() => setForm(null)}
              className="px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-xs text-slate-300"
            >
              Cancel
            </button>
            {mut.isError && <span className="text-[11px] text-red-400">Failed to save</span>}
            {mut.isSuccess && <span className="text-[11px] text-emerald-400">Saved</span>}
          </div>
        )}
      </div>
    </div>
  );
}

function ResourceMonitor() {
  const [history, setHistory] = useState<HardwareResources[]>([]);

  const { data: hw } = useQuery({
    queryKey: ["hardware-resources"],
    queryFn: () => api.get<HardwareResources>("/api/system/resources"),
    refetchInterval: 10000,
  });

  useEffect(() => {
    if (hw) {
      setHistory((prev) => [...prev.slice(-59), hw]);
    }
  }, [hw]);

  if (!hw) return <div className="card text-slate-400 text-center py-6">Loading system resources...</div>;

  const formatUptime = (s: number) => {
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold flex items-center gap-2">
          <Activity size={16} className="text-blue-400" /> System Resources
        </h3>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="flex items-center gap-1"><Clock size={12} /> {formatUptime(hw.uptime_seconds)}</span>
          <span className="flex items-center gap-1"><Camera size={12} /> {hw.cameras_active} cameras</span>
        </div>
      </div>

      {/* Resource gauges grid */}
      <div className="grid grid-cols-3 gap-3">
        <ResourceGauge
          label="CPU"
          value={hw.cpu_percent}
          icon={<Cpu size={14} />}
          detail={`${hw.cpu_cores} cores`}
          history={history.map((h) => h.cpu_percent)}
          color="blue"
        />
        <ResourceGauge
          label="GPU"
          value={hw.gpu_percent ?? 0}
          icon={<Monitor size={14} />}
          detail={hw.gpu_inference_device === "GPU" ? "OpenVINO" : hw.gpu_available ? "Idle" : "N/A"}
          history={history.map((h) => h.gpu_percent ?? 0)}
          color="emerald"
          badge={hw.gpu_inference_device === "GPU" ? "ACTIVE" : undefined}
        />
        <ResourceGauge
          label="RAM"
          value={hw.ram_percent}
          icon={<MemoryStick size={14} />}
          detail={`${hw.ram_used_gb.toFixed(1)}/${hw.ram_total_gb.toFixed(1)} GB`}
          history={history.map((h) => h.ram_percent)}
          color="purple"
        />
      </div>

      {/* Storage bar */}
      <div className="card py-3 px-4">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-medium flex items-center gap-1.5">
            <HardDrive size={13} className="text-slate-400" /> Storage
          </span>
          <span className="text-xs text-slate-400">
            {hw.storage_used_gb.toFixed(1)} / {hw.storage_total_gb.toFixed(1)} GB
          </span>
        </div>
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              hw.storage_percent > 90 ? "bg-red-500" : hw.storage_percent > 75 ? "bg-amber-500" : "bg-blue-500"
            }`}
            style={{ width: `${Math.min(100, hw.storage_percent)}%` }}
          />
        </div>
      </div>

      {/* AI Detector / accelerator indicator */}
      <DetectorIndicator hw={hw} />

      {/* System info line */}
      <div className="text-[10px] text-slate-600 px-1">
        {hw.cpu_name} &middot; {hw.gpu_name || "No GPU"}{hw.coral_available ? " · Coral TPU" : ""} &middot; Max {hw.estimated_max_cameras_relay} cameras
      </div>

      {/* Warnings */}
      {hw.warnings.filter((w) => w.level !== "info").length > 0 && (
        <div className="space-y-1">
          {hw.warnings.filter((w) => w.level !== "info").map((w, i) => (
            <div
              key={i}
              className={`text-xs px-3 py-2 rounded-lg ${
                w.level === "critical"
                  ? "bg-red-900/30 text-red-300 border border-red-800/50"
                  : "bg-amber-900/20 text-amber-300 border border-amber-800/50"
              }`}
            >
              {w.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ResourceGauge({
  label, value, icon, detail, history, color, badge,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
  detail: string;
  history: number[];
  color: "blue" | "emerald" | "purple";
  badge?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const colorMap = {
    blue: { ring: "#3b82f6", bg: "rgba(59,130,246,0.15)", text: "text-blue-400" },
    emerald: { ring: "#10b981", bg: "rgba(16,185,129,0.15)", text: "text-emerald-400" },
    purple: { ring: "#a855f7", bg: "rgba(168,85,247,0.15)", text: "text-purple-400" },
  };
  const c = colorMap[color];

  // Draw sparkline on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const points = history.slice(-30);
    const step = w / (points.length - 1);

    // Fill area
    ctx.beginPath();
    ctx.moveTo(0, h);
    points.forEach((p, i) => ctx.lineTo(i * step, h - (p / 100) * h));
    ctx.lineTo((points.length - 1) * step, h);
    ctx.closePath();
    ctx.fillStyle = c.bg;
    ctx.fill();

    // Stroke line
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = i * step, y = h - (p / 100) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = c.ring;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [history, c]);

  const dangerLevel = value > 90 ? "text-red-400" : value > 70 ? "text-amber-400" : c.text;

  return (
    <div className="card p-3 relative overflow-hidden">
      {/* Sparkline background */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-40" />

      <div className="relative z-10">
        <div className="flex items-center justify-between mb-1">
          <span className={`${c.text} flex items-center gap-1 text-xs font-medium`}>
            {icon} {label}
          </span>
          {badge && (
            <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-bold">
              {badge}
            </span>
          )}
        </div>
        <div className={`text-2xl font-bold tabular-nums ${dangerLevel}`}>
          {Math.round(value)}%
        </div>
        <div className="text-[10px] text-slate-500 mt-0.5">{detail}</div>
      </div>
    </div>
  );
}

/* ═══════════════════════ Training / Auto-Enroll Settings ═══════════════════════ */

interface TrainingSettings {
  auto_enroll_enabled: boolean;
  auto_enroll_threshold: number;
  training_retention_days: number;
  auto_reinforce_cap: number;
}

function TrainingReinforcementSettings() {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const { data } = useQuery({
    queryKey: ["training-settings"],
    queryFn: () => api.get<TrainingSettings>("/api/system/training-settings"),
  });
  const [form, setForm] = useState<TrainingSettings | null>(null);
  const saveMut = useMutation({
    mutationFn: (d: TrainingSettings) => api.put("/api/system/training-settings", d),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["training-settings"] }); },
  });

  useEffect(() => { if (data && !form) setForm(data); }, [data]);

  if (!form) return null;

  const setVal = (key: keyof TrainingSettings, value: number | boolean) =>
    setForm((prev) => prev ? { ...prev, [key]: value } : prev);

  const dirty = data ? JSON.stringify(form) !== JSON.stringify(data) : false;

  return (
    <div className="card p-4 space-y-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full text-left"
      >
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-emerald-400" />
          <span className="font-semibold text-sm">Training & Auto-Enroll</span>
        </div>
        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
      </button>

      {expanded && (
        <div className="space-y-4 pt-2">
          <p className="text-xs text-slate-400">
            High-confidence recognitions can automatically reinforce profile embeddings.
            Manually assigned and auto-enrolled detections are pinned and protected from removal.
          </p>

          {/* Auto-enroll toggle */}
          <label className="flex items-center justify-between">
            <span className="text-sm">Auto-enroll enabled</span>
            <input
              type="checkbox"
              checked={form.auto_enroll_enabled}
              onChange={(e) => setVal("auto_enroll_enabled", e.target.checked)}
              className="toggle"
            />
          </label>

          {/* Threshold slider */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-sm">Confidence threshold</span>
              <span className="text-xs font-mono text-blue-400">{Math.round(form.auto_enroll_threshold * 100)}%</span>
            </div>
            <input
              type="range"
              min={50}
              max={100}
              step={5}
              value={Math.round(form.auto_enroll_threshold * 100)}
              onChange={(e) => setVal("auto_enroll_threshold", parseInt(e.target.value) / 100)}
              className="w-full accent-blue-500"
            />
            <p className="text-[10px] text-slate-500">
              Only detections above this confidence will be auto-enrolled as training data
            </p>
          </div>

          {/* Retention days */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-sm">Retention period</span>
              <span className="text-xs font-mono text-blue-400">{form.training_retention_days} days</span>
            </div>
            <input
              type="range"
              min={7}
              max={365}
              step={7}
              value={form.training_retention_days}
              onChange={(e) => setVal("training_retention_days", parseInt(e.target.value))}
              className="w-full accent-blue-500"
            />
            <p className="text-[10px] text-slate-500">
              Pinned detections (manual + auto-enrolled) are protected from removal for this period
            </p>
          </div>

          {/* Reference cap */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-sm">Max references per profile</span>
              <span className="text-xs font-mono text-blue-400">{form.auto_reinforce_cap}</span>
            </div>
            <input
              type="range"
              min={10}
              max={200}
              step={10}
              value={form.auto_reinforce_cap}
              onChange={(e) => setVal("auto_reinforce_cap", parseInt(e.target.value))}
              className="w-full accent-blue-500"
            />
            <p className="text-[10px] text-slate-500">
              Stop auto-reinforcing embeddings after this many training images
            </p>
          </div>

          {/* Save button */}
          {dirty && (
            <button
              onClick={() => saveMut.mutate(form)}
              disabled={saveMut.isPending}
              className="btn-primary w-full text-sm py-2 flex items-center justify-center gap-1.5"
            >
              <Save size={14} /> Save Training Settings
            </button>
          )}
          {saveMut.isSuccess && <p className="text-emerald-400 text-xs text-center">Saved</p>}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ Advanced / Performance Settings ═══════════════════════ */

function AdvancedSettings() {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(false);

  const { data: perf } = useQuery({
    queryKey: ["performance-settings"],
    queryFn: () => api.get<PerformanceSettings>("/api/system/performance"),
  });

  const [form, setForm] = useState<PerformanceSettings | null>(null);
  const current = form ?? perf;

  const saveMut = useMutation({
    mutationFn: (data: PerformanceSettings) => api.put("/api/system/performance", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["performance-settings"] });
      setForm(null);
    },
  });

  const set = (key: keyof PerformanceSettings, value: number) => {
    if (!current) return;
    // Blur kernel must be odd
    if (key === "motion_blur_kernel" && value % 2 === 0) value = value + 1;
    setForm({ ...current, [key]: value, preset: null });
  };

  const applyPreset = (p: typeof PRESETS[number]) => {
    setForm({ ...p.values, preset: p.key });
  };

  const isDirty = form !== null;

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <SlidersHorizontal size={16} className="text-blue-400" /> Advanced Settings
          {current?.preset && (
            <span className="text-[10px] text-slate-500 font-normal capitalize">({current.preset})</span>
          )}
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>

      {expanded && current && (
        <div className="card space-y-4 p-4">
          {/* Preset buttons */}
          <div>
            <label className="label mb-2">Presets</label>
            <div className="grid grid-cols-3 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.key}
                  onClick={() => applyPreset(p)}
                  className={`flex flex-col items-center gap-1 p-2.5 rounded-lg border text-xs transition-all ${
                    current.preset === p.key
                      ? p.color + " border-opacity-100"
                      : "border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-300"
                  }`}
                >
                  {p.icon}
                  <span className="font-medium">{p.label}</span>
                  <span className="text-[9px] text-slate-500 leading-tight text-center">{p.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Motion Detection */}
          <div className="space-y-3 border-t border-slate-800 pt-3">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Motion Detection</h4>
            <SliderRow
              label="Frame Skip"
              description="Process every Nth frame (higher = less CPU)"
              value={current.motion_frame_skip}
              min={4} max={30} step={1}
              onChange={(v) => set("motion_frame_skip", v)}
            />
            <SliderRow
              label="Blur Kernel"
              description="Gaussian blur size for noise reduction (odd only)"
              value={current.motion_blur_kernel}
              min={3} max={21} step={2}
              onChange={(v) => set("motion_blur_kernel", v)}
            />
            <SliderRow
              label="Cooldown"
              description="Seconds between motion triggers"
              value={current.motion_cooldown}
              min={1} max={30} step={0.5}
              unit="s"
              onChange={(v) => set("motion_cooldown", v)}
            />
          </div>

          {/* Object Tracking */}
          <div className="space-y-3 border-t border-slate-800 pt-3">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Object Tracking (YOLO)</h4>
            <SliderRow
              label="Track Interval"
              description="Seconds between YOLO inferences per camera"
              value={current.track_interval}
              min={1} max={10} step={0.5}
              unit="s"
              onChange={(v) => set("track_interval", v)}
            />
            <SliderRow
              label="Enhanced Scan"
              description="Seconds between small-object scans"
              value={current.enhanced_scan_interval}
              min={10} max={120} step={5}
              unit="s"
              onChange={(v) => set("enhanced_scan_interval", v)}
            />
            <SliderRow
              label="YOLO Concurrency"
              description="Max simultaneous YOLO inferences"
              value={current.yolo_concurrency}
              min={1} max={4} step={1}
              onChange={(v) => set("yolo_concurrency", v)}
            />
          </div>

          {/* Recognition & Output */}
          <div className="space-y-3 border-t border-slate-800 pt-3">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Recognition & Output</h4>
            <SliderRow
              label="Detection Pipelines"
              description="Max concurrent recognition pipelines"
              value={current.max_detection_pipelines}
              min={1} max={6} step={1}
              onChange={(v) => set("max_detection_pipelines", v)}
            />
            <SliderRow
              label="JPEG Quality"
              description="Snapshot image quality (lower saves storage)"
              value={current.jpeg_quality}
              min={40} max={100} step={5}
              unit="%"
              onChange={(v) => set("jpeg_quality", v)}
            />
          </div>

          {/* ML Offload */}
          <div className="space-y-3 border-t border-slate-800 pt-3">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Server size={12} /> ML Offload (GPU Server)
            </h4>
            <MLOffloadSection
              enabled={current.ml_offload_enabled}
              url={current.ml_offload_url}
              onToggle={(v) => setForm({ ...current, ml_offload_enabled: v, preset: null })}
              onUrlChange={(v) => setForm({ ...current, ml_offload_url: v, preset: null })}
            />
          </div>

          {/* Coral Edge TPU */}
          <div className="space-y-3 border-t border-slate-800 pt-3">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu size={12} /> Coral Edge TPU
            </h4>
            <div className="flex items-center justify-between bg-slate-800/50 rounded-lg px-3 py-2">
              <div>
                <p className="text-sm text-white">USB Accelerator</p>
                <p className="text-xs text-slate-400">Use Coral TPU for object detection &amp; CNN features</p>
              </div>
              <button
                onClick={() => setForm({ ...current, coral_enabled: !current.coral_enabled, preset: null })}
                className={`w-10 h-5 rounded-full transition-colors relative ${current.coral_enabled ? "bg-teal-500" : "bg-slate-600"}`}
              >
                <span className={`block w-4 h-4 rounded-full bg-white absolute top-0.5 transition-all ${current.coral_enabled ? "left-5" : "left-0.5"}`} />
              </button>
            </div>
            {current.coral_enabled && (
              <p className="text-xs text-teal-400/80 px-1">
                Requires INT8 Edge TPU-compiled models in /models/. Detection will fall back to iGPU if Coral is unavailable.
              </p>
            )}
          </div>

          {/* Save / Reset */}
          {isDirty && (
            <div className="flex gap-2 pt-2">
              <button
                onClick={() => saveMut.mutate(form!)}
                disabled={saveMut.isPending}
                className="btn-primary flex-1 flex items-center justify-center gap-2 text-sm"
              >
                <Save size={14} /> {saveMut.isPending ? "Applying..." : "Apply Settings"}
              </button>
              <button
                onClick={() => setForm(null)}
                className="btn-secondary text-sm flex items-center gap-1.5"
              >
                <RotateCcw size={14} /> Reset
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MLOffloadSection({
  enabled, url, onToggle, onUrlChange,
}: {
  enabled: boolean;
  url: string;
  onToggle: (v: boolean) => void;
  onUrlChange: (v: string) => void;
}) {
  const { data: health, isLoading } = useQuery<MLHealthInfo>({
    queryKey: ["ml-health"],
    queryFn: () => api.get<MLHealthInfo>("/api/system/ml-health"),
    refetchInterval: enabled ? 15000 : false,
  });

  return (
    <div className="space-y-3">
      {/* Toggle */}
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs font-medium text-slate-300">Enable GPU Offload</div>
          <div className="text-[10px] text-slate-500">Send ML inference to remote GPU server</div>
        </div>
        <button
          onClick={() => onToggle(!enabled)}
          className={`relative w-10 h-5 rounded-full transition-all ${
            enabled ? "bg-blue-500" : "bg-slate-700"
          }`}
        >
          <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${
            enabled ? "left-5" : "left-0.5"
          }`} />
        </button>
      </div>

      {/* URL */}
      {enabled && (
        <div>
          <label className="text-[10px] text-slate-500 mb-1 block">Server URL</label>
          <input
            type="text"
            value={url}
            onChange={(e) => onUrlChange(e.target.value)}
            className="w-full bg-slate-800/50 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-blue-500"
            placeholder="https://ml.banusphotos.com"
          />
        </div>
      )}

      {/* Health status */}
      {enabled && (
        <div className="flex items-center gap-2 text-xs">
          {isLoading ? (
            <span className="text-slate-500">Checking...</span>
          ) : health?.online ? (
            <>
              <Wifi size={12} className="text-emerald-400" />
              <span className="text-emerald-400">Connected</span>
              {health.gpu && (
                <span className="text-slate-500 ml-1">— {health.gpu}</span>
              )}
            </>
          ) : (
            <>
              <WifiOff size={12} className="text-red-400" />
              <span className="text-red-400">Offline</span>
              {health?.error && (
                <span className="text-slate-600 ml-1 truncate max-w-48">{health.error}</span>
              )}
            </>
          )}
        </div>
      )}

      {/* Troubleshooting hint when offline */}
      {enabled && !isLoading && health && !health.online && (
        <div className="text-[11px] text-slate-400 bg-slate-900/60 border border-slate-800 rounded p-2.5 space-y-1">
          <div className="font-medium text-amber-300">ML server unreachable. Check:</div>
          <ul className="list-disc list-inside space-y-0.5 text-slate-500">
            <li>The URL above is correct (include <code className="text-slate-300">https://</code> and any port)</li>
            <li>The ml-server container is running: <code className="text-slate-300">docker compose -f docker-compose.ml.yml up -d</code></li>
            <li>The host is reachable from this network (firewall, DNS)</li>
            <li>Look at server logs: <code className="text-slate-300">docker logs banusnvr-ml-server</code></li>
          </ul>
        </div>
      )}

      {/* Model status */}
      {enabled && health?.online && health.models && (
        <div className="grid grid-cols-2 gap-1">
          {Object.entries(health.models).map(([model, ready]) => (
            <div key={model} className="flex items-center gap-1.5 text-[10px]">
              <div className={`w-1.5 h-1.5 rounded-full ${ready ? "bg-emerald-400" : "bg-red-400"}`} />
              <span className="text-slate-500 capitalize">{model.replace("_", " ")}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SliderRow({
  label, description, value, min, max, step, unit, onChange,
}: {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-300">{label}</span>
        <span className="text-xs font-mono text-blue-400 tabular-nums">
          {Number.isInteger(step) ? value : value.toFixed(1)}{unit || ""}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-slate-700 rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-slate-900"
      />
      <p className="text-[10px] text-slate-600">{description}</p>
    </div>
  );
}

/* ═══════════════════════ Notification Settings ═══════════════════════ */

function NotificationSettings() {
  const [expanded, setExpanded] = useState(false);
  const [pushEnabled, setPushEnabled] = useState(false);
  const [status, setStatus] = useState("");

  const subscribePush = async () => {
    try {
      const reg = await navigator.serviceWorker.ready;
      const { vapid_public_key } = await api.get<{ vapid_public_key: string }>("/api/notifications/vapid-key");
      const keyArray = urlBase64ToUint8Array(vapid_public_key);
      const sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: keyArray.buffer as ArrayBuffer,
      });
      await api.post("/api/notifications/subscribe", sub.toJSON());
      setPushEnabled(true);
      setStatus("Push notifications enabled");
    } catch (err: any) {
      setStatus(`Failed: ${err.message}`);
    }
  };

  const testNotification = async () => {
    try {
      await api.post("/api/notifications/test", { type: "push" });
      setStatus("Test notification sent");
    } catch (err: any) {
      setStatus(`Failed: ${err.message}`);
    }
  };

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <Bell size={16} className="text-blue-400" /> Notifications
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>

      {expanded && (
        <div className="card space-y-3">
          <div className="flex gap-2">
            <button onClick={subscribePush} className="btn-primary text-sm">Enable Push</button>
            <button onClick={testNotification} className="btn-secondary text-sm">Test</button>
          </div>
          {status && <p className="text-xs text-slate-400">{status}</p>}
          <div className="border-t border-slate-800 pt-3">
            <p className="text-xs text-slate-500 flex items-center gap-1.5">
              <Mail size={12} /> Email: Configure SMTP in Storage settings
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ Storage Settings ═══════════════════════ */

function StorageSettings() {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(false);

  const { data: settings } = useQuery({
    queryKey: ["settings"],
    queryFn: () => api.get<GlobalSettings>("/api/system/settings"),
  });

  const [form, setForm] = useState<GlobalSettings | null>(null);

  const currentSettings = form ?? settings;

  const saveMut = useMutation({
    mutationFn: (data: GlobalSettings) => api.put("/api/system/settings", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["settings"] });
      setForm(null);
    },
  });

  const updateField = (key: keyof GlobalSettings, value: any) => {
    if (!currentSettings) return;
    setForm({ ...currentSettings, [key]: value });
  };

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <HardDrive size={16} className="text-blue-400" /> Storage & Retention
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>

      {expanded && currentSettings && (
        <div className="card space-y-3">
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="label">Events (days)</label>
              <input type="number" className="input" value={currentSettings.retention_events_days}
                onChange={(e) => updateField("retention_events_days", parseInt(e.target.value))} />
            </div>
            <div>
              <label className="label">Continuous (days)</label>
              <input type="number" className="input" value={currentSettings.retention_continuous_days}
                onChange={(e) => updateField("retention_continuous_days", parseInt(e.target.value))} />
            </div>
            <div>
              <label className="label">Snapshots (days)</label>
              <input type="number" className="input" value={currentSettings.retention_snapshots_days}
                onChange={(e) => updateField("retention_snapshots_days", parseInt(e.target.value))} />
            </div>
          </div>

          <details className="text-sm">
            <summary className="text-slate-400 cursor-pointer hover:text-white text-xs">SMTP Settings</summary>
            <div className="mt-2 space-y-2">
              <input className="input" placeholder="SMTP Host" value={currentSettings.smtp_host || ""}
                onChange={(e) => updateField("smtp_host", e.target.value)} />
              <div className="grid grid-cols-2 gap-2">
                <input type="number" className="input" placeholder="Port" value={currentSettings.smtp_port || ""}
                  onChange={(e) => updateField("smtp_port", parseInt(e.target.value))} />
                <input className="input" placeholder="Username" value={currentSettings.smtp_user || ""}
                  onChange={(e) => updateField("smtp_user", e.target.value)} />
              </div>
              <input className="input" placeholder="From address" value={currentSettings.smtp_from || ""}
                onChange={(e) => updateField("smtp_from", e.target.value)} />
            </div>
          </details>

          {form && (
            <button
              onClick={() => saveMut.mutate(form)}
              className="btn-primary w-full flex items-center justify-center gap-2 text-sm"
              disabled={saveMut.isPending}
            >
              <Save size={14} /> {saveMut.isPending ? "Saving..." : "Save Settings"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ Account Settings ═══════════════════════ */

function AccountSettings() {
  const { user, logout } = useAuth();
  const [expanded, setExpanded] = useState(false);
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [status, setStatus] = useState("");

  const changeMut = useMutation({
    mutationFn: (data: { old_password: string; new_password: string }) =>
      api.put("/api/auth/password", data),
    onSuccess: () => {
      setStatus("Password updated");
      setOldPassword("");
      setNewPassword("");
    },
    onError: (err: any) => setStatus(err.message),
  });

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <Shield size={16} className="text-blue-400" /> Account
          <span className="text-xs text-slate-500 font-normal">{user?.username}</span>
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>

      {expanded && (
        <div className="card space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <input type="password" className="input" placeholder="Current password" value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)} />
            <input type="password" className="input" placeholder="New password" value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)} />
          </div>
          {status && <p className="text-xs text-slate-400">{status}</p>}
          <div className="flex gap-2">
            <button
              onClick={() => changeMut.mutate({ old_password: oldPassword, new_password: newPassword })}
              className="btn-primary text-sm flex-1"
              disabled={!oldPassword || !newPassword || changeMut.isPending}
            >
              <Key size={14} className="inline mr-1" /> Update Password
            </button>
            <button onClick={logout} className="btn-danger text-sm flex items-center gap-1">
              <LogOut size={14} /> Logout
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ App Settings ═══════════════════════ */

function AppSettings() {
  const [expanded, setExpanded] = useState(false);
  const [checking, setChecking] = useState(false);
  const [status, setStatus] = useState("");

  const checkForUpdate = async () => {
    if (!("serviceWorker" in navigator)) { setStatus("Service workers not supported"); return; }
    setChecking(true);
    setStatus("Checking...");
    try {
      const reg = await navigator.serviceWorker.getRegistration();
      if (!reg) { setStatus("No service worker registered"); setChecking(false); return; }
      await reg.update();
      await new Promise((r) => setTimeout(r, 2000));
      if (reg.waiting) {
        setStatus("Update found! Applying...");
        reg.waiting.postMessage({ type: "SKIP_WAITING" });
        window.location.reload();
      } else {
        setStatus("You're on the latest version");
      }
    } catch (e: any) {
      setStatus(`Update check failed: ${e.message}`);
    }
    setChecking(false);
  };

  const forceReload = () => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.getRegistration().then((reg) => {
        if (reg) {
          reg.unregister().then(() => {
            if ("caches" in window) caches.keys().then((names) => names.forEach((n) => caches.delete(n)));
            window.location.reload();
          });
        } else window.location.reload();
      });
    } else window.location.reload();
  };

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <Smartphone size={16} className="text-blue-400" /> App
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>

      {expanded && (
        <div className="card space-y-3">
          <div className="flex gap-2">
            <button onClick={checkForUpdate} disabled={checking}
              className="btn-primary text-sm flex-1 flex items-center justify-center gap-1.5">
              <RefreshCw size={14} className={checking ? "animate-spin" : ""} />
              {checking ? "Checking..." : "Check for Update"}
            </button>
            <button onClick={forceReload} className="btn-secondary text-sm">Clear Cache</button>
          </div>
          {status && <p className="text-xs text-slate-400">{status}</p>}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ Utilities ═══════════════════════ */

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const rawData = window.atob(base64);
  return Uint8Array.from(rawData, (char) => char.charCodeAt(0));
}

/* ═══════════════════════ Ring Settings ═══════════════════════ */

interface RingStatus { online: boolean; message: string; hub_waiting?: boolean }
interface RingAuthState { connected?: boolean; displayName?: string; error?: string }

function RingSettings() {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();

  const { data: status } = useQuery({
    queryKey: ["ring-status"],
    queryFn: () => api.get<RingStatus>("/api/ring/status"),
    enabled: expanded,
    refetchInterval: expanded ? 10_000 : false,
  });
  const { data: authState } = useQuery({
    queryKey: ["ring-auth-state"],
    queryFn: () => api.get<RingAuthState>("/api/ring/auth/state"),
    enabled: expanded,
  });

  const connected = status?.online === true || authState?.connected === true;

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <Radio size={16} className="text-blue-400" /> Ring Integration
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>

      {expanded && (
        <div className="card space-y-3">
          <div className="flex items-center gap-2">
            {connected ? (
              <>
                <Wifi size={14} className="text-green-400" />
                <span className="text-sm text-green-300 font-medium">Connected</span>
                {authState?.displayName && (
                  <span className="text-xs text-slate-400">· {authState.displayName}</span>
                )}
              </>
            ) : (
              <>
                <WifiOff size={14} className="text-red-400" />
                <span className="text-sm text-red-300 font-medium">Not connected</span>
              </>
            )}
          </div>
          {status?.message && (
            <p className="text-xs text-slate-400">{status.message}</p>
          )}
          <p className="text-xs text-slate-500">
            Manage your Ring account credentials, refresh tokens, and add Ring devices as cameras.
          </p>
          <button
            onClick={() => navigate("/ring")}
            className="btn-primary text-sm w-full flex items-center justify-center gap-2"
          >
            <ExternalLink size={14} />
            {connected ? "Manage Ring Connection" : "Set Up Ring"}
          </button>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ Theme Settings ═══════════════════════ */

type ThemeChoice = "light" | "dark" | "system";

function applyTheme(theme: ThemeChoice) {
  const root = document.documentElement;
  let resolved: "light" | "dark" = "dark";
  if (theme === "system") {
    resolved = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } else {
    resolved = theme;
  }
  root.classList.remove("light", "dark");
  root.classList.add(resolved);
  root.dataset.theme = resolved;
  localStorage.setItem("banusnas_theme", theme);
}

function ThemeSettings() {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const { data: me } = useQuery({
    queryKey: ["auth-me"],
    queryFn: () => api.get<{ theme: string }>("/api/auth/me"),
  });
  const current = (me?.theme as ThemeChoice) || ((localStorage.getItem("banusnas_theme") as ThemeChoice) || "system");

  const updateMut = useMutation({
    mutationFn: (theme: ThemeChoice) => api.patch("/api/auth/me", { theme }),
    onSuccess: (_d, theme) => {
      applyTheme(theme);
      qc.invalidateQueries({ queryKey: ["auth-me"] });
    },
  });

  const choose = (t: ThemeChoice) => {
    applyTheme(t);
    updateMut.mutate(t);
  };

  const opts: { key: ThemeChoice; label: string; icon: React.ReactNode }[] = [
    { key: "system", label: "System", icon: <Monitor size={14} /> },
    { key: "light", label: "Light", icon: <Sun size={14} /> },
    { key: "dark", label: "Dark", icon: <Moon size={14} /> },
  ];

  return (
    <div className="space-y-3">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full">
        <h3 className="font-semibold flex items-center gap-2">
          <Palette size={16} className="text-blue-400" /> Appearance
        </h3>
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
      </button>
      {expanded && (
        <div className="card space-y-3">
          <p className="text-xs text-slate-400">Choose how BanusNVR looks. System matches your device setting.</p>
          <div className="grid grid-cols-3 gap-2">
            {opts.map((o) => (
              <button
                key={o.key}
                onClick={() => choose(o.key)}
                className={`flex flex-col items-center justify-center gap-1.5 py-3 rounded-lg border transition-colors ${
                  current === o.key
                    ? "bg-blue-600/20 border-blue-500 text-blue-200"
                    : "bg-slate-800/40 border-slate-700 text-slate-300 hover:border-slate-500"
                }`}
              >
                {o.icon}
                <span className="text-xs font-medium">{o.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
