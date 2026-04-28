import { useState } from "react";
import { api } from "../api";
import { useAuth } from "../hooks/useAuth";
import { Loader2, QrCode, ShieldCheck, X } from "lucide-react";

export default function TOTPSetupModal({ onClose }: { onClose: () => void }) {
  const { user, refreshProfile } = useAuth();
  const [step, setStep] = useState<"setup" | "verify" | "done">("setup");
  const [secret, setSecret] = useState("");
  const [uri, setUri] = useState("");
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const startSetup = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<{ secret: string; uri: string }>("/api/auth/totp/setup");
      setSecret(res.secret);
      setUri(res.uri);
      setStep("verify");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start TOTP setup");
    } finally {
      setBusy(false);
    }
  };

  const verify = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.post("/api/auth/totp/verify", { token });
      setStep("done");
      await refreshProfile();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid code");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-800" onClick={e => e.stopPropagation()}>
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <ShieldCheck size={20} className="text-blue-400" />
            Set up 2FA (TOTP)
          </h2>
          <button onClick={onClose} className="p-1 hover:bg-slate-800 rounded"><X size={18} /></button>
        </div>
        <div className="p-5">
          {step === "setup" && (
            <div className="flex flex-col items-center gap-4">
              <button
                onClick={startSetup}
                className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 font-medium flex items-center gap-2"
                disabled={busy}
              >
                {busy && <Loader2 size={16} className="animate-spin" />} Start setup
              </button>
              {error && <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">{error}</div>}
            </div>
          )}
          {step === "verify" && (
            <form onSubmit={verify} className="space-y-4">
              <div className="flex flex-col items-center gap-2">
                <div className="bg-white p-2 rounded-lg">
                  <img src={`https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(uri)}`} alt="QR" className="w-36 h-36" />
                </div>
                <div className="text-xs text-slate-400 break-all">Secret: <span className="font-mono">{secret}</span></div>
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Enter code from your app</label>
                <input
                  type="text"
                  value={token}
                  onChange={e => setToken(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700"
                  required
                  pattern="\\d{6}"
                  inputMode="numeric"
                  maxLength={6}
                />
              </div>
              {error && <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">{error}</div>}
              <button
                type="submit"
                disabled={busy}
                className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 font-medium flex items-center justify-center gap-2"
              >
                {busy && <Loader2 size={16} className="animate-spin" />} Enable 2FA
              </button>
            </form>
          )}
          {step === "done" && (
            <div className="flex flex-col items-center gap-4">
              <div className="text-emerald-400 font-semibold flex items-center gap-2"><ShieldCheck /> 2FA enabled!</div>
              <button onClick={onClose} className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 font-medium">Close</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
