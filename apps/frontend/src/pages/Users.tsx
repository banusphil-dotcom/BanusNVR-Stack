import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useAuth, type UserRole } from "../hooks/useAuth";
import {
  Loader2,
  Plus,
  Shield,
  ShieldOff,
  KeyRound,
  Trash2,
  Unlock,
  X,
  UserCog,
} from "lucide-react";

interface UserRow {
  id: number;
  username: string;
  email: string;
  is_admin: boolean;
  role: UserRole;
  must_change_password: boolean;
  disabled: boolean;
  last_login_at: string | null;
  created_at: string;
}

const ROLES: UserRole[] = ["admin", "operator", "viewer", "guest"];

const roleStyle: Record<UserRole, string> = {
  admin: "bg-red-500/15 text-red-300 border-red-500/30",
  operator: "bg-blue-500/15 text-blue-300 border-blue-500/30",
  viewer: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  guest: "bg-amber-500/15 text-amber-300 border-amber-500/30",
};

function formatDate(s: string | null) {
  if (!s) return "Never";
  return new Date(s).toLocaleString();
}

export default function Users() {
  const { user: me, hasPermission } = useAuth();
  const qc = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [resetTarget, setResetTarget] = useState<UserRow | null>(null);

  const canManage = hasPermission("manage_users");

  const usersQuery = useQuery({
    queryKey: ["admin-users"],
    queryFn: () => api.get<UserRow[]>("/api/users"),
    enabled: canManage,
  });

  const updateMut = useMutation({
    mutationFn: (vars: { id: number; patch: Partial<{ role: UserRole; disabled: boolean; must_change_password: boolean; email: string }> }) =>
      api.patch(`/api/users/${vars.id}`, vars.patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-users"] }),
  });

  const unlockMut = useMutation({
    mutationFn: (id: number) => api.post(`/api/users/${id}/unlock`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-users"] }),
  });

  const deleteMut = useMutation({
    mutationFn: (id: number) => api.delete(`/api/users/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-users"] }),
  });

  if (!canManage) {
    return <div className="p-6 text-slate-400">You don't have permission to manage users.</div>;
  }

  return (
    <div className="p-4 sm:p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold flex items-center gap-2">
            <UserCog size={24} className="text-blue-400" />
            Users
          </h1>
          <p className="text-slate-400 text-sm mt-1">Manage accounts, roles, and access</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 font-medium"
        >
          <Plus size={16} /> New user
        </button>
      </div>

      {usersQuery.isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-500">
          <Loader2 size={28} className="animate-spin" />
        </div>
      )}

      {usersQuery.data && (
        <div className="rounded-xl border border-slate-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-900 text-slate-400 text-left">
              <tr>
                <th className="px-4 py-3 font-medium">User</th>
                <th className="px-4 py-3 font-medium">Role</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Last login</th>
                <th className="px-4 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {usersQuery.data.map((u) => {
                const isMe = me?.id === String(u.id);
                return (
                  <tr key={u.id} className="border-t border-slate-800 hover:bg-slate-900/40">
                    <td className="px-4 py-3">
                      <div className="font-medium">{u.username} {isMe && <span className="text-xs text-slate-500">(you)</span>}</div>
                      <div className="text-xs text-slate-500">{u.email}</div>
                    </td>
                    <td className="px-4 py-3">
                      <select
                        value={u.role}
                        disabled={isMe}
                        onChange={(e) => updateMut.mutate({ id: u.id, patch: { role: e.target.value as UserRole } })}
                        className={`px-2 py-1 rounded border text-xs font-medium ${roleStyle[u.role]} disabled:opacity-50`}
                      >
                        {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                      </select>
                    </td>
                    <td className="px-4 py-3">
                      {u.disabled ? (
                        <span className="text-xs px-2 py-1 rounded-full bg-red-500/15 text-red-300 border border-red-500/30">Disabled</span>
                      ) : u.must_change_password ? (
                        <span className="text-xs px-2 py-1 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30">Must change pw</span>
                      ) : (
                        <span className="text-xs px-2 py-1 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">Active</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-400">{formatDate(u.last_login_at)}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => unlockMut.mutate(u.id)}
                          className="p-2 rounded hover:bg-slate-800 text-slate-400"
                          title="Reset failed attempts / unlock"
                        >
                          <Unlock size={16} />
                        </button>
                        <button
                          onClick={() => setResetTarget(u)}
                          className="p-2 rounded hover:bg-slate-800 text-slate-400"
                          title="Reset password"
                        >
                          <KeyRound size={16} />
                        </button>
                        <button
                          onClick={() => updateMut.mutate({ id: u.id, patch: { disabled: !u.disabled } })}
                          disabled={isMe}
                          className="p-2 rounded hover:bg-slate-800 text-slate-400 disabled:opacity-30"
                          title={u.disabled ? "Re-enable" : "Disable"}
                        >
                          {u.disabled ? <Shield size={16} /> : <ShieldOff size={16} />}
                        </button>
                        <button
                          onClick={() => {
                            if (confirm(`Permanently delete user "${u.username}"?`)) deleteMut.mutate(u.id);
                          }}
                          disabled={isMe}
                          className="p-2 rounded hover:bg-red-500/20 text-red-400 disabled:opacity-30"
                          title="Delete user"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {showCreate && <CreateUserModal onClose={() => setShowCreate(false)} />}
      {resetTarget && <ResetPasswordModal user={resetTarget} onClose={() => setResetTarget(null)} />}
    </div>
  );
}

function CreateUserModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("viewer");
  const [forceChange, setForceChange] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: () => api.post("/api/users", { username, email: email || undefined, password, role, must_change_password: forceChange }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["admin-users"] });
      onClose();
    },
    onError: (err) => setError(err instanceof Error ? err.message : "Failed to create user"),
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-800" onClick={(e) => e.stopPropagation()}>
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Invite user</h2>
          <button onClick={onClose} className="p-1 hover:bg-slate-800 rounded"><X size={18} /></button>
        </div>
        <form onSubmit={(e) => { e.preventDefault(); setError(null); mut.mutate(); }} className="p-5 space-y-3">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Username</label>
            <input value={username} onChange={(e) => setUsername(e.target.value)} required minLength={3}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700" />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Email (optional)</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700" />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Temporary password</label>
            <input type="text" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 font-mono" />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Role</label>
            <select value={role} onChange={(e) => setRole(e.target.value as UserRole)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700">
              {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={forceChange} onChange={(e) => setForceChange(e.target.checked)} />
            Require password change on first login
          </label>
          {error && <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">{error}</div>}
          <button type="submit" disabled={mut.isPending}
            className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 font-medium disabled:opacity-50 flex items-center justify-center gap-2">
            {mut.isPending && <Loader2 size={16} className="animate-spin" />}
            Create user
          </button>
        </form>
      </div>
    </div>
  );
}

function ResetPasswordModal({ user, onClose }: { user: UserRow; onClose: () => void }) {
  const [password, setPassword] = useState("");
  const [forceChange, setForceChange] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: () => api.post(`/api/users/${user.id}/reset-password`, { new_password: password, must_change_password: forceChange }),
    onSuccess: onClose,
    onError: (err) => setError(err instanceof Error ? err.message : "Failed to reset password"),
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-800" onClick={(e) => e.stopPropagation()}>
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Reset password</h2>
            <p className="text-sm text-slate-400">{user.username}</p>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-slate-800 rounded"><X size={18} /></button>
        </div>
        <form onSubmit={(e) => { e.preventDefault(); setError(null); mut.mutate(); }} className="p-5 space-y-3">
          <p className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded p-2">
            All of {user.username}'s active sessions will be revoked.
          </p>
          <div>
            <label className="block text-sm text-slate-400 mb-1">New temporary password</label>
            <input type="text" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 font-mono" />
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={forceChange} onChange={(e) => setForceChange(e.target.checked)} />
            Require user to change it on next login
          </label>
          {error && <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">{error}</div>}
          <button type="submit" disabled={mut.isPending}
            className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 font-medium disabled:opacity-50 flex items-center justify-center gap-2">
            {mut.isPending && <Loader2 size={16} className="animate-spin" />}
            Reset password
          </button>
        </form>
      </div>
    </div>
  );
}
