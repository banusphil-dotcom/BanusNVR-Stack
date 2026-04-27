import { useEffect, useRef, useState, useCallback } from "react";
import { getToken } from "../api";

export interface NvrEvent {
  id: string;
  camera_id: string;
  camera_name: string;
  event_type: string;
  objects: { label: string; confidence: number; box: number[] }[];
  faces: { name: string; similarity: number }[];
  started_at: string;
  thumbnail_url?: string;
}

export function useWebSocket() {
  const [events, setEvents] = useState<NvrEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const retryCount = useRef(0);
  const visible = useRef(true);

  const connect = useCallback(() => {
    // Don't connect when tab is hidden
    if (!visible.current) return;

    // Prevent duplicate sockets
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/events`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      retryCount.current = 0;
    };
    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      // Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
      const delay = Math.min(1000 * Math.pow(2, retryCount.current), 30000);
      retryCount.current++;
      if (visible.current && getToken()) {
        reconnectTimer.current = setTimeout(connect, delay);
      }
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.type === "ping") return; // Ignore server keepalive pings
        setEvents((prev) => [data as NvrEvent, ...prev].slice(0, 100));
      } catch {}
    };
  }, []);

  useEffect(() => {
    // Reconnect when tab becomes visible, disconnect when hidden
    const onVisibility = () => {
      visible.current = !document.hidden;
      if (document.hidden) {
        clearTimeout(reconnectTimer.current);
        wsRef.current?.close();
        wsRef.current = null;
      } else if (getToken()) {
        retryCount.current = 0;
        connect();
      }
    };

    const onFocus = () => {
      visible.current = true;
      if (getToken()) {
        retryCount.current = 0;
        connect();
      }
    };

    const onOnline = () => {
      if (getToken()) {
        retryCount.current = 0;
        connect();
      }
    };

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("focus", onFocus);
    window.addEventListener("online", onOnline);
    if (getToken()) connect();
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("online", onOnline);
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const clearEvents = useCallback(() => setEvents([]), []);

  return { events, connected, clearEvents };
}
