import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useAuth } from "../hooks/useAuth";
import { ScrollText, Loader2, Search } from "lucide-react";

interface AuditEntry {
  id: number;
  user_id: number | null;
  actor_username: string | null;
  action: string;
  target_type: string | null;
  target_id: number | null;
  detail: Record<string, unknown> | null;
  ip_address: string | null;
  user_agent: string | null;
  created_at: string;
}

interface Page {
  items: AuditEntry[];
  total: number;
  page: number;
  page_size: number;
}

const actionStyle = (action: string) => {
  if (action.includes("failed") || action.includes("blocked")) return "text-red-300 bg-red-500/10 border-red-500/20";
  if (action.includes("deleted") || action.includes("revoked") || action.includes("disabled")) return "text-amber-300 bg-amber-500/10 border-amber-500/20";
  if (action.includes("created") || action.includes("login_success")) return "text-emerald-300 bg-emerald-500/10 border-emerald-500/20";
  return "text-slate-300 bg-slate-700/30 border-slate-600/40";
};

export default function AuditLog() {
  const { hasPermission } = useAuth();
  const [page, setPage] = useState(1);
  const [actionFilter, setActionFilter] = useState("");
  const [pendingFilter, setPendingFilter] = useState("");

  const canView = hasPermission("view_audit_log");

  const query = useQuery({
    queryKey: ["audit-logs", page, actionFilter],
    queryFn: () => {
      const params = new URLSearchParams({ page: String(page), page_size: "50" });
      if (actionFilter) params.set("action", actionFilter);
      return api.get<Page>(`/api/audit-logs?${params.toString()}`);
    },
    enabled: canView,
  });

  if (!canView) return <div className="p-6 text-slate-400">You don't have permission to view audit logs.</div>;

  const totalPages = query.data ? Math.max(1, Math.ceil(query.data.total / query.data.page_size)) : 1;

  return (
    <div className="p-4 sm:p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold flex items-center gap-2">
            <ScrollText size={24} className="text-blue-400" />
            Audit log
          </h1>
          <p className="text-slate-400 text-sm mt-1">Security-relevant actions across the system</p>
        </div>
      </div>

      <form
        onSubmit={(e) => { e.preventDefault(); setPage(1); setActionFilter(pendingFilter); }}
        className="flex items-center gap-2 mb-4"
      >
        <div className="relative flex-1 max-w-sm">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={pendingFilter}
            onChange={(e) => setPendingFilter(e.target.value)}
            placeholder="Filter by action (e.g. login_failed)"
            className="w-full pl-9 pr-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-sm"
          />
        </div>
        <button type="submit" className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-sm font-medium">
          Filter
        </button>
        {actionFilter && (
          <button
            type="button"
            onClick={() => { setActionFilter(""); setPendingFilter(""); setPage(1); }}
            className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm"
          >
            Clear
          </button>
        )}
      </form>

      {query.isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-500">
          <Loader2 size={28} className="animate-spin" />
        </div>
      )}

      {query.data && (
        <>
          <div className="rounded-xl border border-slate-800 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-slate-400 text-left">
                <tr>
                  <th className="px-4 py-3 font-medium">When</th>
                  <th className="px-4 py-3 font-medium">Actor</th>
                  <th className="px-4 py-3 font-medium">Action</th>
                  <th className="px-4 py-3 font-medium">Target</th>
                  <th className="px-4 py-3 font-medium">IP</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((row) => (
                  <tr key={row.id} className="border-t border-slate-800 align-top">
                    <td className="px-4 py-3 text-slate-400 whitespace-nowrap">
                      {new Date(row.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3">{row.actor_username || <span className="text-slate-600">—</span>}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-block px-2 py-0.5 rounded border text-xs font-medium ${actionStyle(row.action)}`}>
                        {row.action}
                      </span>
                      {row.detail && Object.keys(row.detail).length > 0 && (
                        <div className="text-xs text-slate-500 mt-1 font-mono break-all">
                          {JSON.stringify(row.detail)}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-400">
                      {row.target_type ? `${row.target_type}#${row.target_id ?? "?"}` : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="px-4 py-3 text-slate-400 font-mono text-xs">{row.ip_address || "—"}</td>
                  </tr>
                ))}
                {query.data.items.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-12 text-center text-slate-500">No entries</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between mt-4 text-sm text-slate-400">
            <div>{query.data.total} entries</div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30"
              >
                Prev
              </button>
              <span>Page {page} / {totalPages}</span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
