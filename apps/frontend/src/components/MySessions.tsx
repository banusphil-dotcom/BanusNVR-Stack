import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { Loader2, Monitor, X } from "lucide-react";

interface SessionRow {
  id: number;
  user_agent: string | null;
  ip_address: string | null;
  created_at: string;
  last_seen_at: string;
  revoked_at: string | null;
}

function deviceLabel(ua: string | null): string {
  if (!ua) return "Unknown device";
  const lower = ua.toLowerCase();
  let os = "Unknown OS";
  if (lower.includes("windows")) os = "Windows";
  else if (lower.includes("mac os") || lower.includes("macintosh")) os = "macOS";
  else if (lower.includes("iphone")) os = "iPhone";
  else if (lower.includes("ipad")) os = "iPad";
  else if (lower.includes("android")) os = "Android";
  else if (lower.includes("linux")) os = "Linux";
  let browser = "Browser";
  if (lower.includes("edg/")) browser = "Edge";
  else if (lower.includes("chrome/") && !lower.includes("edg/")) browser = "Chrome";
  else if (lower.includes("firefox/")) browser = "Firefox";
  else if (lower.includes("safari/") && !lower.includes("chrome/")) browser = "Safari";
  return `${browser} on ${os}`;
}

export default function MySessions() {
  const qc = useQueryClient();
  const sessions = useQuery({
    queryKey: ["my-sessions"],
    queryFn: () => api.get<SessionRow[]>("/api/auth/sessions"),
  });

  const revoke = useMutation({
    mutationFn: (id: number) => api.delete(`/api/auth/sessions/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["my-sessions"] }),
  });

  if (sessions.isLoading) {
    return (
      <div className="flex items-center justify-center py-8 text-slate-500">
        <Loader2 size={24} className="animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sessions.data?.map((s) => (
        <div
          key={s.id}
          className={`flex items-center gap-4 p-3 rounded-lg border ${
            s.revoked_at ? "bg-slate-900/40 border-slate-800 opacity-60" : "bg-slate-900 border-slate-800"
          }`}
        >
          <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center shrink-0">
            <Monitor size={18} className="text-slate-400" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium">{deviceLabel(s.user_agent)}</div>
            <div className="text-xs text-slate-500 truncate">
              {s.ip_address || "no IP"} • last seen {new Date(s.last_seen_at).toLocaleString()}
            </div>
          </div>
          {s.revoked_at ? (
            <span className="text-xs text-slate-500">Revoked</span>
          ) : (
            <button
              onClick={() => revoke.mutate(s.id)}
              className="p-2 rounded hover:bg-red-500/20 text-red-400"
              title="Sign out this device"
            >
              <X size={18} />
            </button>
          )}
        </div>
      ))}
      {sessions.data?.length === 0 && (
        <div className="text-center text-slate-500 py-8">No active sessions</div>
      )}
    </div>
  );
}
