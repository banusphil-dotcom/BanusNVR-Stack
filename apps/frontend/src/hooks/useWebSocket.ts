import { useEffect, useState, useCallback } from "react";
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
  // Generic publisher payload fields used by the backend (frigate_bridge etc.)
  type?: string;
  object_type?: string;
}

// ---------------------------------------------------------------------------
// Module-level singleton WebSocket broker.
//
// Every component that calls `useWebSocket()` previously opened its OWN
// socket connection. With multiple <CameraStream/> tiles + Layout + Events
// page that's a fan-out of parallel sockets. This broker collapses it to a
// single shared connection that all subscribers read from.
// ---------------------------------------------------------------------------

type Listener = (events: NvrEvent[], connected: boolean) => void;

const listeners = new Set<Listener>();
let sharedEvents: NvrEvent[] = [];
let sharedConnected = false;
let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let retryCount = 0;
let visible = typeof document === "undefined" ? true : !document.hidden;
let listenersAttached = false;

function notify() {
  for (const l of listeners) l(sharedEvents, sharedConnected);
}

function setConnected(v: boolean) {
  if (sharedConnected !== v) {
    sharedConnected = v;
    notify();
  }
}

function connect() {
  if (typeof window === "undefined") return;
  if (!visible) return;
  if (!getToken()) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${window.location.host}/ws/events`);

  ws.onopen = () => {
    retryCount = 0;
    setConnected(true);
  };
  ws.onclose = () => {
    setConnected(false);
    ws = null;
    const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
    retryCount++;
    if (visible && getToken()) {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, delay);
    }
  };
  ws.onerror = () => ws?.close();
  ws.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      if (data?.type === "ping") return;
      sharedEvents = [data as NvrEvent, ...sharedEvents].slice(0, 100);
      notify();
    } catch {}
  };
}

function attachGlobalListeners() {
  if (listenersAttached) return;
  if (typeof window === "undefined") return;
  listenersAttached = true;

  document.addEventListener("visibilitychange", () => {
    visible = !document.hidden;
    if (document.hidden) {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
      ws = null;
    } else if (getToken()) {
      retryCount = 0;
      connect();
    }
  });
  window.addEventListener("focus", () => {
    visible = true;
    if (getToken()) {
      retryCount = 0;
      connect();
    }
  });
  window.addEventListener("online", () => {
    if (getToken()) {
      retryCount = 0;
      connect();
    }
  });
}

export function useWebSocket() {
  const [events, setEvents] = useState<NvrEvent[]>(sharedEvents);
  const [connected, setConnected] = useState<boolean>(sharedConnected);

  useEffect(() => {
    attachGlobalListeners();

    const listener: Listener = (e, c) => {
      setEvents(e);
      setConnected(c);
    };
    listeners.add(listener);

    // Sync to current state in case events arrived before subscription.
    setEvents(sharedEvents);
    setConnected(sharedConnected);

    if (getToken()) connect();

    return () => {
      listeners.delete(listener);
      // Tear down the socket only when no more subscribers remain.
      if (listeners.size === 0) {
        if (reconnectTimer) clearTimeout(reconnectTimer);
        ws?.close();
        ws = null;
      }
    };
  }, []);

  const clearEvents = useCallback(() => {
    sharedEvents = [];
    notify();
  }, []);

  return { events, connected, clearEvents };
}
