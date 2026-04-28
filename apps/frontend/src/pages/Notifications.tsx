import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, getToken } from "../api";
import { Bell, Trash2, CheckCheck, Mail, Smartphone } from "lucide-react";
import { prettyObjectType } from "../utils/objectType";

interface SentNotification {
  id: number;
  event_id: number | null;
  channel: string;
  title: string;
  body: string;
  camera_name: string | null;
  object_type: string | null;
  read: boolean;
  created_at: string;
}

interface NotificationPage {
  items: SentNotification[];
  total: number;
  page: number;
  page_size: number;
}

export default function Notifications() {
  const qc = useQueryClient();
  const [page, setPage] = useState(1);
  const token = getToken();

  const { data, isLoading } = useQuery({
    queryKey: ["notification-history", page],
    queryFn: () =>
      api.get<NotificationPage>(`/api/notifications/history?page=${page}&page_size=30`),
  });

  const markReadMut = useMutation({
    mutationFn: () => api.post("/api/notifications/mark-read"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notification-history"] });
      qc.invalidateQueries({ queryKey: ["unread-count"] });
    },
  });

  const clearMut = useMutation({
    mutationFn: () => api.delete("/api/notifications/history"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notification-history"] });
      qc.invalidateQueries({ queryKey: ["unread-count"] });
      setPage(1);
    },
  });

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 0;

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold flex items-center gap-2">
          <Bell size={20} /> Notifications
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => markReadMut.mutate()}
            disabled={markReadMut.isPending}
            className="flex items-center gap-1 text-xs px-2 py-1.5 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
            title="Mark all read"
          >
            <CheckCheck size={14} />
            <span className="hidden sm:inline">Mark read</span>
          </button>
          <button
            onClick={() => {
              if (confirm("Clear all notification history?")) clearMut.mutate();
            }}
            disabled={clearMut.isPending}
            className="flex items-center gap-1 text-xs px-2 py-1.5 bg-slate-800 hover:bg-red-900 rounded transition-colors text-red-400"
            title="Clear all"
          >
            <Trash2 size={14} />
            <span className="hidden sm:inline">Clear</span>
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="text-slate-400 text-center py-12">Loading...</div>
      ) : !data?.items.length ? (
        <div className="text-slate-500 text-center py-12 space-y-2">
          <Bell size={40} className="mx-auto opacity-30" />
          <p>No notifications yet</p>
          <p className="text-xs">Notifications will appear here when motion or objects are detected</p>
        </div>
      ) : (
        <div className="space-y-2">
          {data.items.map((n) => (
            <a
              key={n.id}
              href={n.event_id ? `/events/${n.event_id}` : undefined}
              className={`card flex items-start gap-3 block ${!n.read ? "border-l-2 border-l-blue-500" : ""}`}
            >
              {n.event_id ? (
                <img
                  src={`/api/events/${n.event_id}/snapshot?token=${encodeURIComponent(token || "")}`}
                  alt=""
                  className="w-16 h-12 rounded object-contain shrink-0 bg-slate-800"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                />
              ) : (
                <div className="shrink-0 mt-0.5">
                  {n.channel === "push" ? (
                    <Smartphone size={16} className="text-blue-400" />
                  ) : (
                    <Mail size={16} className="text-amber-400" />
                  )}
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{n.title}</p>
                <p className="text-xs text-slate-400 truncate">{n.body}</p>
                <div className="flex items-center gap-2 mt-1 text-xs text-slate-500">
                  {n.camera_name && <span>{n.camera_name}</span>}
                  {n.object_type && (
                    <>
                      <span>·</span>
                      <span className="capitalize">{prettyObjectType(n.object_type)}</span>
                    </>
                  )}
                  <span>·</span>
                  <span>{new Date(n.created_at).toLocaleString()}</span>
                </div>
              </div>
            </a>
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="btn-primary text-xs px-3 py-1 disabled:opacity-30"
          >
            Previous
          </button>
          <span className="text-xs text-slate-400">
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="btn-primary text-xs px-3 py-1 disabled:opacity-30"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
