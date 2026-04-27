import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import {
  Radio, Plus, CheckCircle, AlertTriangle, Loader2, Camera,
  Wifi, WifiOff, RefreshCw, Mail, KeyRound, Eye, EyeOff, LogIn, RotateCcw,
} from "lucide-react";

interface RingDevice {
  device_id: string;
  name: string;
  model: string;
  manufacturer: string;
  unique_id: string;
  status: string;
  already_added: boolean;
  firmware?: string;
  location?: string;
}

interface DiscoveryResult {
  devices: RingDevice[];
  mqtt_host: string;
  mqtt_port: number;
  ring_rtsp_user: string;
}

interface RingStatus { online: boolean; message: string; hub_waiting?: boolean }
interface AuthState { connected?: boolean; displayName?: string; error?: string }

export default function RingSetup() {
  const qc = useQueryClient();
  const [addingId, setAddingId] = useState<string | null>(null);

  /* Auth form state */
  const [authStep, setAuthStep] = useState<"idle" | "login" | "2fa" | "success">("idle");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [code, setCode] = useState("");
  const [authError, setAuthError] = useState("");

  // Check ring-mqtt service status
  const { data: status, isLoading: statusLoading, refetch: refetchStatus } = useQuery({
    queryKey: ["ring-status"],
    queryFn: () => api.get<RingStatus>("/api/ring/status"),
  });

  const { data: authState, refetch: refetchAuth } = useQuery({
    queryKey: ["ring-auth-state"],
    queryFn: () => api.get<AuthState>("/api/ring/auth/state"),
  });

  // Discover Ring devices
  const { data: discovery, isLoading: discovering, refetch: refetchDevices } = useQuery({
    queryKey: ["ring-devices"],
    queryFn: () => api.get<DiscoveryResult>("/api/ring/devices"),
    enabled: status?.online === true || authState?.connected === true,
  });

  /* Auth mutations */
  const submitAccountMut = useMutation({
    mutationFn: (d: { email: string; password: string }) =>
      api.post<{ requires2fa?: boolean; success?: boolean }>("/api/ring/auth/submit-account", d),
    onSuccess: (data) => {
      setAuthError("");
      if (data.requires2fa) {
        setAuthStep("2fa");
      } else if (data.success) {
        setAuthStep("success");
        setTimeout(() => { refetchStatus(); refetchAuth(); refetchDevices(); }, 3000);
      }
    },
    onError: (e: Error) => setAuthError(e.message),
  });

  const submitCodeMut = useMutation({
    mutationFn: (d: { code: string }) =>
      api.post<{ success?: boolean }>("/api/ring/auth/submit-code", d),
    onSuccess: (data) => {
      setAuthError("");
      if (data.success) {
        setAuthStep("success");
        setTimeout(() => { refetchStatus(); refetchAuth(); refetchDevices(); }, 3000);
      }
    },
    onError: (e: Error) => setAuthError(e.message),
  });

  const addMut = useMutation({
    mutationFn: (device: RingDevice) => {
      const ringDeviceName = device.name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
      return api.post<{ camera_id: number; name: string }>("/api/ring/add-camera", {
        device_id: device.device_id,
        name: device.name,
        ring_device_name: ringDeviceName,
      });
    },
    onSuccess: () => {
      setAddingId(null);
      qc.invalidateQueries({ queryKey: ["ring-devices"] });
      qc.invalidateQueries({ queryKey: ["cameras"] });
    },
    onError: () => setAddingId(null),
  });

  const handleAdd = (device: RingDevice) => {
    setAddingId(device.device_id);
    addMut.mutate(device);
  };

  const handleAccountSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError("");
    submitAccountMut.mutate({ email, password });
  };

  const handleCodeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError("");
    submitCodeMut.mutate({ code });
  };

  const startAuth = () => {
    setAuthStep("login");
    setAuthError("");
    setEmail("");
    setPassword("");
    setCode("");
  };

  const isConnected = status?.online === true;
  const isAuthed = authState?.connected === true;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Radio className="text-blue-400" size={24} />
          <h2 className="text-lg font-bold">Ring Cameras</h2>
        </div>
        <button
          onClick={() => { refetchStatus(); refetchAuth(); refetchDevices(); }}
          className="btn btn-sm text-xs px-3 py-1.5 rounded flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Connection Status + Auth */}
      <div className="card p-4">
        <h3 className="text-sm font-semibold mb-3 text-slate-300">Connection Status</h3>
        {statusLoading ? (
          <div className="flex items-center gap-2 text-slate-400">
            <Loader2 size={16} className="animate-spin" />
            <span className="text-sm">Checking service status...</span>
          </div>
        ) : isConnected ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-emerald-400">
              <Wifi size={16} />
              <span className="text-sm font-medium">Connected — ring-mqtt is online</span>
            </div>
            {authState?.displayName && (
              <p className="text-xs text-slate-500">Device: {authState.displayName}</p>
            )}
          </div>
        ) : authStep === "success" ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle size={16} />
              <span className="text-sm font-medium">Authentication successful!</span>
            </div>
            <p className="text-xs text-slate-400">ring-mqtt is connecting to Ring servers. Devices will appear shortly...</p>
            <Loader2 size={14} className="animate-spin text-slate-500" />
          </div>
        ) : isAuthed ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-amber-400">
              <WifiOff size={16} />
              <span className="text-sm font-medium">Authenticated but not yet connected</span>
            </div>
            <p className="text-xs text-slate-400">ring-mqtt has a saved token but isn't online yet. It may be restarting.</p>
            <button onClick={startAuth} className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
              <RotateCcw size={12} /> Re-authenticate
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-amber-400">
              <WifiOff size={16} />
              <span className="text-sm font-medium">Not connected</span>
            </div>
            {authStep === "idle" && (
              <>
                <p className="text-xs text-slate-400">
                  Connect your Ring account to discover and add Ring cameras to the NVR.
                </p>
                <button
                  onClick={startAuth}
                  className="btn bg-blue-600 hover:bg-blue-500 text-white text-sm px-4 py-2 rounded flex items-center gap-2"
                >
                  <LogIn size={16} />
                  Connect Ring Account
                </button>
              </>
            )}
          </div>
        )}

        {/* Login form */}
        {authStep === "login" && !isConnected && (
          <form onSubmit={handleAccountSubmit} className="mt-4 space-y-4">
            <div className="bg-slate-800/60 rounded-lg p-4 space-y-3">
              <p className="text-sm font-medium text-slate-300 flex items-center gap-2">
                <Mail size={16} className="text-blue-400" />
                Sign in to your Ring account
              </p>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  autoFocus
                  className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
                  placeholder="your@email.com"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Password</label>
                <div className="relative">
                  <input
                    type={showPw ? "text" : "password"}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 pr-10 text-sm focus:border-blue-500 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw(!showPw)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-300"
                  >
                    {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setAuthStep("idle")}
                  className="btn bg-slate-600 hover:bg-slate-500 text-white text-sm px-4 py-2 rounded"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitAccountMut.isPending}
                  className="btn bg-blue-600 hover:bg-blue-500 text-white text-sm px-4 py-2 rounded flex-1 flex items-center justify-center gap-2"
                >
                  {submitAccountMut.isPending ? (
                    <><Loader2 size={14} className="animate-spin" /> Signing in...</>
                  ) : (
                    <><LogIn size={14} /> Sign In</>
                  )}
                </button>
              </div>
            </div>
          </form>
        )}

        {/* 2FA form */}
        {authStep === "2fa" && !isConnected && (
          <form onSubmit={handleCodeSubmit} className="mt-4 space-y-4">
            <div className="bg-slate-800/60 rounded-lg p-4 space-y-3">
              <p className="text-sm font-medium text-slate-300 flex items-center gap-2">
                <KeyRound size={16} className="text-amber-400" />
                Two-Factor Authentication
              </p>
              <p className="text-xs text-slate-400">
                A verification code has been sent to your phone or email. Enter it below.
              </p>
              <div>
                <label className="text-xs text-slate-400 block mb-1">2FA Code</label>
                <input
                  type="text"
                  value={code}
                  onChange={e => setCode(e.target.value)}
                  required
                  autoFocus
                  className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm text-center tracking-widest text-lg focus:border-blue-500 focus:outline-none"
                  placeholder="000000"
                  maxLength={6}
                  inputMode="numeric"
                />
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => { setAuthStep("login"); setAuthError(""); }}
                  className="btn bg-slate-600 hover:bg-slate-500 text-white text-sm px-4 py-2 rounded"
                >
                  Back
                </button>
                <button
                  type="submit"
                  disabled={submitCodeMut.isPending}
                  className="btn bg-blue-600 hover:bg-blue-500 text-white text-sm px-4 py-2 rounded flex-1 flex items-center justify-center gap-2"
                >
                  {submitCodeMut.isPending ? (
                    <><Loader2 size={14} className="animate-spin" /> Verifying...</>
                  ) : (
                    <><KeyRound size={14} /> Verify Code</>
                  )}
                </button>
              </div>
            </div>
          </form>
        )}

        {/* Auth error */}
        {authError && (
          <div className="mt-3 bg-red-900/40 border border-red-700 rounded p-3 text-sm flex items-center gap-2">
            <AlertTriangle size={16} className="text-red-400 flex-shrink-0" />
            <span>{authError}</span>
          </div>
        )}
      </div>

      {/* Discovered Devices */}
      {(isConnected || isAuthed) && (
        <div className="card p-4">
          <h3 className="text-sm font-semibold mb-3 text-slate-300">Discovered Cameras</h3>
          {discovering ? (
            <div className="flex items-center gap-2 text-slate-400 py-8 justify-center">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">Scanning MQTT for Ring devices...</span>
            </div>
          ) : !discovery?.devices?.length ? (
            <div className="text-center py-8 text-slate-500 space-y-2">
              <Camera size={32} className="mx-auto opacity-30" />
              <p className="text-sm">No Ring cameras discovered yet</p>
              {status?.hub_waiting ? (
                <div className="text-xs space-y-1">
                  <p className="text-amber-500">A Ring alarm hub at your location is offline.</p>
                  <p>ring-mqtt waits for all hubs to be online before discovering cameras.</p>
                  <p>If the hub is permanently offline, remove it from the Ring app to unblock discovery.</p>
                </div>
              ) : (
                <p className="text-xs">ring-mqtt may still be discovering devices. Click Refresh to check again.</p>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {discovery.devices.map((device) => (
                <div
                  key={device.device_id}
                  className={`flex items-center gap-4 p-3 rounded-lg border ${
                    device.already_added
                      ? "border-emerald-800/50 bg-emerald-900/20"
                      : "border-slate-700 bg-slate-800/40"
                  }`}
                >
                  <Camera size={28} className={device.already_added ? "text-emerald-400" : "text-slate-400"} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium truncate">{device.name}</p>
                      {device.already_added && (
                        <span className="flex items-center gap-1 text-xs text-emerald-400">
                          <CheckCircle size={12} /> Added
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-slate-500 mt-0.5">
                      {device.model && <span>{device.model}</span>}
                      {device.firmware && <span>FW: {device.firmware}</span>}
                    </div>
                  </div>
                  <div className="flex-shrink-0">
                    {device.already_added ? (
                      <span className="text-xs text-slate-500">In NVR</span>
                    ) : (
                      <button
                        onClick={() => handleAdd(device)}
                        disabled={addingId === device.device_id}
                        className="btn bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded flex items-center gap-1.5"
                      >
                        {addingId === device.device_id ? (
                          <><Loader2 size={14} className="animate-spin" /> Adding...</>
                        ) : (
                          <><Plus size={14} /> Add to NVR</>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error banner */}
      {addMut.isError && (
        <div className="card bg-red-900/40 border-red-700 p-3 text-sm flex items-center gap-2">
          <AlertTriangle size={16} className="text-red-400" />
          <span>{(addMut.error as Error)?.message || "Failed to add camera"}</span>
        </div>
      )}

      {/* Success banner */}
      {addMut.isSuccess && (
        <div className="card bg-emerald-900/40 border-emerald-700 p-3 text-sm flex items-center gap-2">
          <CheckCircle size={16} className="text-emerald-400" />
          <span>Camera added successfully! It will appear on the Cameras page.</span>
        </div>
      )}

      {/* Info section */}
      <div className="card p-4 bg-slate-800/30">
        <h3 className="text-sm font-semibold mb-2 text-slate-300">How it works</h3>
        <div className="text-xs text-slate-500 space-y-1.5">
          <p>Ring cameras connect through <strong>ring-mqtt</strong> which bridges your Ring account to a local RTSP stream.</p>
          <p>Each camera's live stream is relayed through go2rtc for low-latency WebRTC/HLS viewing.</p>
          <p>Ring cameras support on-demand streaming — the stream activates when you view the camera or during motion events.</p>
          <p className="text-amber-500/80">Note: Ring camera streams are on-demand. Continuous recording is not available.</p>
        </div>
      </div>
    </div>
  );
}
