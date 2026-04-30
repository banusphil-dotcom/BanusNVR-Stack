import { useEffect, useState } from "react";
import { useAuth } from "../hooks/useAuth";
import { isWebAuthnSupported } from "../webauthn";
import { api } from "../api";
import { Camera, Fingerprint, ShieldCheck } from "lucide-react";

interface AuthMethods {
  totp_enabled: boolean;
  webauthn_enabled: boolean;
  oidc_enabled: boolean;
  api_tokens_enabled: boolean;
  magic_links_enabled: boolean;
}

type Stage = "credentials" | "totp";

export default function Login() {
  const { login, loginTotp, loginPasskey, register } = useAuth();
  const [methods, setMethods] = useState<AuthMethods | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [tempToken, setTempToken] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>("credentials");
  const [isRegister, setIsRegister] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [passkeyLoading, setPasskeyLoading] = useState(false);

  useEffect(() => {
    api.get<AuthMethods>("/api/auth/settings").then(setMethods).catch(() => setMethods(null));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (stage === "totp" && tempToken) {
        await loginTotp(tempToken, code);
        return;
      }
      if (isRegister) {
        await register(username, password);
        return;
      }
      const result = await login(username, password);
      if (result.kind === "totp") {
        setTempToken(result.tempToken);
        setStage("totp");
        setCode("");
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  const handlePasskey = async () => {
    setError("");
    setPasskeyLoading(true);
    try {
      await loginPasskey(username || undefined);
    } catch (err: any) {
      setError(err.message || "Passkey sign-in failed");
    } finally {
      setPasskeyLoading(false);
    }
  };

  const showPasskey = methods?.webauthn_enabled !== false && isWebAuthnSupported();
  const showTotpStage = stage === "totp";

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-slate-950">
      <div className="card w-full max-w-sm">
        <div className="flex flex-col items-center mb-6">
          <div className="w-14 h-14 rounded-2xl bg-blue-600 flex items-center justify-center mb-3">
            <Camera size={28} />
          </div>
          <h1 className="text-xl font-bold">BanusNas NVR</h1>
          <p className="text-sm text-slate-400 mt-1">
            {showTotpStage ? "Enter your 2FA code" : "Smart CCTV Monitoring"}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {!showTotpStage && (
            <>
              <div>
                <label className="label">Username</label>
                <input
                  className="input"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
              <div>
                <label className="label">Password</label>
                <input
                  className="input"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  required
                  minLength={6}
                />
              </div>
            </>
          )}

          {showTotpStage && (
            <div>
              <label className="label">Authenticator code</label>
              <input
                className="input tracking-widest text-center font-mono text-lg"
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
                inputMode="numeric"
                pattern="\d{6}"
                maxLength={6}
                autoFocus
                required
              />
              <p className="text-xs text-slate-500 mt-1">
                Open your authenticator app and enter the 6-digit code.
              </p>
            </div>
          )}

          {error && <p className="text-red-400 text-sm">{error}</p>}

          <button type="submit" className="btn-primary w-full" disabled={loading}>
            {loading
              ? "..."
              : showTotpStage
              ? "Verify"
              : isRegister
              ? "Create Account"
              : (
                <span className="flex items-center justify-center gap-2">
                  <ShieldCheck size={16} /> Sign In
                </span>
              )}
          </button>
        </form>

        {!showTotpStage && showPasskey && !isRegister && (
          <button
            onClick={handlePasskey}
            disabled={passkeyLoading}
            className="w-full mt-3 btn-secondary flex items-center justify-center gap-2"
            type="button"
          >
            <Fingerprint size={16} />
            {passkeyLoading ? "Touch your authenticator…" : "Sign in with passkey / biometrics"}
          </button>
        )}

        {!showTotpStage && (
          <button
            onClick={() => {
              setIsRegister(!isRegister);
              setError("");
            }}
            className="w-full text-center text-sm text-slate-400 hover:text-white mt-4 transition-colors"
            type="button"
          >
            {isRegister ? "Already have an account? Sign in" : "First time? Create account"}
          </button>
        )}

        {showTotpStage && (
          <button
            onClick={() => {
              setStage("credentials");
              setTempToken(null);
              setCode("");
              setError("");
            }}
            className="w-full text-center text-sm text-slate-400 hover:text-white mt-4 transition-colors"
            type="button"
          >
            Back
          </button>
        )}
      </div>
    </div>
  );
}
