import { useState } from "react";
import { api } from "../api";
import { useAuth } from "../hooks/useAuth";
import { Lock, Loader2 } from "lucide-react";

/**
 * Modal forced on the user when their account flag `must_change_password`
 * is true (set by an admin invite or password reset). Cannot be dismissed.
 */
export default function ForceChangePasswordModal() {
  const { user, mustChangePassword, setMustChangePassword } = useAuth();
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (!user || !mustChangePassword) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (newPassword.length < 8) return setError("New password must be at least 8 characters");
    if (newPassword !== confirm) return setError("Passwords do not match");
    setBusy(true);
    try {
      await api.put("/api/auth/password", { old_password: oldPassword, new_password: newPassword });
      setMustChangePassword(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to change password");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl">
        <div className="p-6 border-b border-slate-800 flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-blue-600/20 flex items-center justify-center">
            <Lock size={20} className="text-blue-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Set a new password</h2>
            <p className="text-sm text-slate-400">An admin requires you to change it before continuing.</p>
          </div>
        </div>
        <form onSubmit={submit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Current password</label>
            <input
              type="password"
              value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white"
              required
              autoFocus
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">New password</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white"
              required
              minLength={8}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Confirm new password</label>
            <input
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white"
              required
              minLength={8}
            />
          </div>
          {error && <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">{error}</div>}
          <button
            type="submit"
            disabled={busy}
            className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 font-medium flex items-center justify-center gap-2"
          >
            {busy && <Loader2 size={16} className="animate-spin" />}
            Update password
          </button>
        </form>
      </div>
    </div>
  );
}
