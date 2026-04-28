import { useEffect, useRef, useState } from "react";
import { RefreshCw, Crosshair, SlidersHorizontal } from "lucide-react";
import { api } from "../api";
import { useWebSocket } from "../hooks/useWebSocket";

interface Props {
  cameraId: number | string;
  cameraName: string;
  className?: string;
  hideLabel?: boolean;
  onSettingsToggle?: (active: boolean) => void;
}

interface TrackedObject {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  named_object_name?: string;
  received_at?: number;  // client-side timestamp (ms) — used to suppress stale boxes
}

type StreamStatus = "connecting" | "live" | "snapshot" | "error";

// Supported codecs for MSE negotiation with go2rtc
const CODECS = [
  "avc1.640029", "avc1.64002A", "avc1.640033",
  "hvc1.1.6.L153.B0",
  "mp4a.40.2", "mp4a.40.5", "flac", "opus",
];

function supportedCodecs(): string {
  const MS = window.MediaSource || (window as any).ManagedMediaSource;
  if (!MS) return "";
  return CODECS.filter((c) => MS.isTypeSupported(`video/mp4; codecs="${c}"`)).join();
}

export default function CameraStream({ cameraId, cameraName, className = "", hideLabel, onSettingsToggle }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [liveReady, setLiveReady] = useState(false);
  const [snapUrl, setSnapUrl] = useState("");
  const [retryKey, setRetryKey] = useState(0);
  const [showOverlay, setShowOverlay] = useState(false);
  const [tracks, setTracks] = useState<Record<string, TrackedObject>>({});
  const stalledSinceRef = useRef<number | null>(null);
  const go2rtcErrorRef = useRef(false);

  const stream = `camera_${cameraId}`;

  // go2rtc WebSocket protocol: WebRTC (instant) + MSE fallback
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    setStatus("connecting");
    setLiveReady(false);
    go2rtcErrorRef.current = false;
    let fell = false;
    let pc: RTCPeerConnection | null = null;
    let ondata: ((data: ArrayBuffer) => void) | null = null;

    const markLiveReady = () => {
      // Don't mark as live if go2rtc reported an error (it sends black error frames)
      if (go2rtcErrorRef.current) return;
      if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 2) {
        setLiveReady(true);
      }
    };

    const fallback = () => {
      if (!fell) { fell = true; setStatus("snapshot"); }
    };

    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProto}//${location.host}/go2rtc/api/ws?src=${stream}`);
    ws.binaryType = "arraybuffer";

    const send = (msg: object) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
    };

    // Message handlers keyed by mode
    const handlers: Record<string, (msg: any) => void> = {};

    // ── MSE setup (runs immediately, acts as fallback if WebRTC fails) ──
    const setupMSE = () => {
      const MS = window.MediaSource || (window as any).ManagedMediaSource;
      if (!MS) return;

      const ms = new MS();
      if ("ManagedMediaSource" in window) {
        video.disableRemotePlayback = true;
        (video as any).srcObject = ms;
      } else {
        video.src = URL.createObjectURL(ms);
      }
      video.play().catch(() => { video.muted = true; video.play().catch(() => {}); });

      ms.addEventListener("sourceopen", () => {
        send({ type: "mse", value: supportedCodecs() });
      }, { once: true });

      handlers["mse"] = (msg: any) => {
        if (msg.type !== "mse") return;

        const sb: SourceBuffer = ms.addSourceBuffer(msg.value);
        sb.mode = "segments";

        const buf = new Uint8Array(2 * 1024 * 1024);
        let bufLen = 0;

        sb.addEventListener("updateend", () => {
          // Flush pending data
          if (!sb.updating && bufLen > 0) {
            try {
              sb.appendBuffer(buf.slice(0, bufLen));
              bufLen = 0;
            } catch { /* ignore */ }
          }

          // Live edge tracking: trim buffer + adjust playback rate
          if (!sb.updating && sb.buffered && sb.buffered.length) {
            const end = sb.buffered.end(sb.buffered.length - 1);
            const start0 = sb.buffered.start(0);
            // Keep max 8s of buffer (5s was too aggressive — caused stuttering)
            if (end - start0 > 8) {
              try {
                sb.remove(start0, end - 8);
                ms.setLiveSeekableRange?.(end - 8, end);
              } catch { /* ignore */ }
            }
            // Jump forward if too far behind
            if (video.currentTime < end - 4) {
              video.currentTime = end - 0.5;
            }
            // Smoother playback rate adjustment
            const gap = end - video.currentTime;
            video.playbackRate = gap > 2.0 ? 1.5 : gap > 0.5 ? 1.05 : 1.0;
          }
        });

        ondata = (data) => {
          if (sb.updating || bufLen > 0) {
            const b = new Uint8Array(data);
            buf.set(b, bufLen);
            bufLen += b.byteLength;
          } else {
            try { sb.appendBuffer(data); } catch { /* ignore */ }
          }
        };
      };
    };

    // ── WebRTC setup (lower latency, runs in parallel with MSE) ──
    const setupWebRTC = () => {
      if (!("RTCPeerConnection" in window)) return;

      pc = new RTCPeerConnection({
        bundlePolicy: "max-bundle",
        iceServers: [{ urls: ["stun:stun.cloudflare.com:3478", "stun:stun.l.google.com:19302"] }],
      });

      pc.addTransceiver("video", { direction: "recvonly" });
      pc.addTransceiver("audio", { direction: "recvonly" });

      pc.addEventListener("icecandidate", (ev) => {
        const candidate = ev.candidate ? ev.candidate.toJSON().candidate : "";
        send({ type: "webrtc/candidate", value: candidate });
      });

      pc.addEventListener("connectionstatechange", () => {
        if (pc?.connectionState === "connected") {
          go2rtcErrorRef.current = false; // Real stream connected, clear error
          const tracks = pc.getTransceivers()
            .filter((tr) => tr.currentDirection === "recvonly")
            .map((tr) => tr.receiver.track);
          if (tracks.length) {
            // WebRTC connected — switch video to WebRTC stream, close MSE WS
            video.srcObject = new MediaStream(tracks);
            video.play().catch(() => { video.muted = true; video.play().catch(() => {}); });
            setStatus("live");
          }
        } else if (pc?.connectionState === "failed" || pc?.connectionState === "disconnected") {
          pc.close();
          pc = null;
          // MSE will continue as fallback
        }
      });

      handlers["webrtc"] = (msg: any) => {
        if (!pc) return;
        if (msg.type === "webrtc/candidate") {
          pc.addIceCandidate({ candidate: msg.value, sdpMid: "0" }).catch(() => {});
        } else if (msg.type === "webrtc/answer") {
          pc.setRemoteDescription({ type: "answer", sdp: msg.value }).catch(() => {});
        } else if (msg.type === "error" && msg.value?.includes("webrtc")) {
          pc.close();
          pc = null;
        }
      };

      pc.createOffer().then((offer) => {
        pc!.setLocalDescription(offer);
        send({ type: "webrtc/offer", value: offer.sdp });
      });
    };

    ws.addEventListener("open", () => {
      setupMSE();
      setupWebRTC();
    });

    ws.addEventListener("message", (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        // Handle generic go2rtc errors (e.g. "no frames received")
        if (msg.type === "error" && !msg.value?.includes?.("webrtc")) {
          console.warn("go2rtc:", msg.value);
          go2rtcErrorRef.current = true;
          // Stop MSE from receiving more data
          ondata = null;
          // Kill the video source so the already-buffered error frame disappears
          video.pause();
          if (video.srcObject) video.srcObject = null;
          if (video.src) { try { URL.revokeObjectURL(video.src); } catch {} video.removeAttribute("src"); }
          video.load(); // reset the video element
          // Close WebRTC if active (it also delivers error frames)
          if (pc) { pc.close(); pc = null; }
          setLiveReady(false);
          fallback();
        }
        for (const key in handlers) handlers[key](msg);
      } else if (ondata) {
        ondata(ev.data);
      }
    });

    const onPlaying = () => {
      // Don't override snapshot/error status if go2rtc reported an error
      if (go2rtcErrorRef.current) return;
      if (status !== "live") setStatus("live");
      markLiveReady();
    };
    const onLoadedData = () => markLiveReady();
    const onTimeUpdate = () => markLiveReady();
    const onWaiting = () => setLiveReady(false);

    video.addEventListener("playing", onPlaying);
    video.addEventListener("loadeddata", onLoadedData);
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("waiting", onWaiting);

    const timer = setTimeout(() => {
      if (video.readyState < 2 && !video.srcObject) fallback();
    }, 10000);

    ws.addEventListener("error", fallback);
    ws.addEventListener("close", (e) => { if (e.code !== 1000) fallback(); });

    return () => {
      clearTimeout(timer);
      ws.close();
      if (pc) { pc.close(); pc = null; }
      if (video.src && !video.srcObject) URL.revokeObjectURL(video.src);
      video.srcObject = null;
      video.removeEventListener("playing", onPlaying);
      video.removeEventListener("loadeddata", onLoadedData);
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("waiting", onWaiting);
    };
  }, [stream, retryKey]);

  // Watchdog: if stream appears stalled/black for several seconds, retry automatically.
  useEffect(() => {
    if (status !== "live") return;
    const video = videoRef.current;
    if (!video) return;

    const iv = setInterval(() => {
      if (document.hidden) return;
      const noFrames = video.videoWidth === 0 || video.videoHeight === 0;
      const stalled = video.readyState < 2 || video.paused;
      if (noFrames || stalled) {
        setLiveReady(false);
        if (stalledSinceRef.current == null) {
          stalledSinceRef.current = Date.now();
        }
        // 8s continuous stall => force reconnect
        if (Date.now() - stalledSinceRef.current > 8000) {
          setStatus("connecting");
          stalledSinceRef.current = null;
          setRetryKey((k) => k + 1);
        }
      } else {
        stalledSinceRef.current = null;
      }
    }, 2000);

    return () => {
      clearInterval(iv);
      stalledSinceRef.current = null;
    };
  }, [status]);

  // Mobile/PWA foreground recovery: when returning to app, reconnect if stream isn't healthy.
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) return;
      const video = videoRef.current;
      const unhealthy =
        status === "error" ||
        status === "snapshot" ||
        !video ||
        video.videoWidth === 0 ||
        video.readyState < 2;
      if (unhealthy) {
        setStatus("connecting");
        setRetryKey((k) => k + 1);
      }
    };

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("focus", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("focus", onVisibility);
    };
  }, [status]);

  // Fallback: snapshot via Frigate's latest.jpg, with go2rtc frame.jpeg fallback.
  // Once Frigate has 404'd for a stream we remember it and skip straight to
  // go2rtc on subsequent polls — this matters for Ring (and other on-demand)
  // cameras that aren't in Frigate's `cameras:` section, where the Frigate
  // request always 404s and was wasting a full HTTP round-trip every 4s.
  const skipFrigateRef = useRef(false);
  useEffect(() => {
    const needsSnapshot =
      status === "snapshot" ||
      status === "connecting" ||
      (status === "live" && !liveReady);
    if (!needsSnapshot) return;

    let alive = true;
    const poll = async () => {
      if (!skipFrigateRef.current) {
        try {
          const r = await fetch(`/frigate/api/${stream}/latest.jpg?_=${Date.now()}`);
          if (r.ok && alive) {
            const blob = await r.blob();
            if (blob.size > 100 && (blob.type.startsWith("image/") || blob.type === "application/octet-stream")) {
              const url = URL.createObjectURL(blob);
              setSnapUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return url; });
              return;
            }
          } else if (r.status === 404) {
            // Camera not registered with Frigate (e.g. Ring) — stop trying.
            skipFrigateRef.current = true;
          }
        } catch { /* Frigate unavailable — try go2rtc */ }
      }
      // Fallback to go2rtc frame.jpeg (works for Ring/on-demand cameras)
      try {
        const r2 = await fetch(`/go2rtc/api/frame.jpeg?src=${stream}&_=${Date.now()}`);
        if (r2.ok && alive) {
          const blob = await r2.blob();
          if (blob.size > 100 && (blob.type.startsWith("image/") || blob.type === "application/octet-stream")) {
            const url = URL.createObjectURL(blob);
            setSnapUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return url; });
          }
        }
      } catch { /* ignore — keep showing last snapshot */ }
    };
    poll();
    const iv = setInterval(poll, 4000);
    return () => {
      alive = false;
      clearInterval(iv);
    };
  }, [stream, status, retryKey, liveReady]);

  // Tracking overlay: poll active tracks and draw boxes on canvas.
  // We poll fast (~250ms) when something is being tracked so the overlay
  // stays close to where the moving subject actually is on screen. The
  // backend caches /tracks for 200ms so this rate is safe.
  const hasTracksRef = useRef(false);
  useEffect(() => { hasTracksRef.current = Object.keys(tracks).length > 0; }, [tracks]);

  useEffect(() => {
    if (!showOverlay || status !== "live") return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const data = await api.get<Record<string, TrackedObject>>(`/api/cameras/${cameraId}/tracks`);
        if (!alive) return;
        const now = performance.now();
        // Stamp every track with the current client time so the draw loop
        // can fade or hide stale boxes (avoids drawing the box where the
        // person *was* a second ago — the main cause of the "box is to the
        // side of the person" bug).
        const stamped: Record<string, TrackedObject> = {};
        for (const [k, v] of Object.entries(data)) {
          stamped[k] = { ...v, received_at: now };
        }
        setTracks(stamped);
      } catch { /* ignore */ }
      if (alive) {
        timer = setTimeout(poll, hasTracksRef.current ? 250 : 1500);
      }
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [showOverlay, status, cameraId]);

  // Draw bounding boxes on canvas using requestAnimationFrame so the overlay
  // is repainted on every video frame instead of only every 500 ms (which
  // visibly lagged behind the moving subject).
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !showOverlay) return;

    let raf = 0;

    const draw = () => {
      const rect = video.getBoundingClientRect();
      if (canvas.width !== Math.round(rect.width)) canvas.width = rect.width;
      if (canvas.height !== Math.round(rect.height)) canvas.height = rect.height;

      const ctx = canvas.getContext("2d");
      if (!ctx) { raf = requestAnimationFrame(draw); return; }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Video natural dimensions
      const vw = video.videoWidth || 1920;
      const vh = video.videoHeight || 1080;

      // object-contain letterboxing: compute the visible area within the element
      const videoAspect = vw / vh;
      const elemAspect = rect.width / rect.height;
      let renderW: number, renderH: number, offsetX: number, offsetY: number;
      if (videoAspect > elemAspect) {
        renderW = rect.width;
        renderH = rect.width / videoAspect;
        offsetX = 0;
        offsetY = (rect.height - renderH) / 2;
      } else {
        renderH = rect.height;
        renderW = rect.height * videoAspect;
        offsetX = (rect.width - renderW) / 2;
        offsetY = 0;
      }

      const COLORS: Record<string, string> = {
        person: "#22c55e", cat: "#f97316", dog: "#f97316", bird: "#f97316",
        car: "#3b82f6", truck: "#3b82f6", bus: "#3b82f6", motorcycle: "#3b82f6",
      };

      const now = performance.now();
      const STALE_HIDE_MS = 2000;   // drop the box entirely after 2s of no update
      const STALE_FADE_MS = 700;    // start fading after 0.7s

      for (const [, t] of Object.entries(tracks)) {
        const age = t.received_at ? now - t.received_at : 0;
        if (age > STALE_HIDE_MS) continue;
        const alpha = age > STALE_FADE_MS
          ? Math.max(0, 1 - (age - STALE_FADE_MS) / (STALE_HIDE_MS - STALE_FADE_MS))
          : 1;

        const [x1, y1, x2, y2] = t.bbox;  // normalized 0-1 from Frigate
        const dx = offsetX + x1 * renderW;
        const dy = offsetY + y1 * renderH;
        const dw = (x2 - x1) * renderW;
        const dh = (y2 - y1) * renderH;
        const color = COLORS[t.class_name] || "#6b7280";

        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(dx, dy, dw, dh);

        // Label
        const label = t.named_object_name || `${t.class_name} ${(t.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 11px sans-serif";
        const tm = ctx.measureText(label);
        const lh = 16;
        ctx.fillStyle = color;
        ctx.fillRect(dx, dy - lh, tm.width + 8, lh);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, dx + 4, dy - 4);
      }
      ctx.globalAlpha = 1;

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [tracks, showOverlay]);

  const retry = () => {
    setLiveReady(false);
    setSnapUrl("");
    go2rtcErrorRef.current = false;
    setRetryKey((k) => k + 1);
  };

  const [showSettings, setShowSettings] = useState(false);

  // Live "detecting" indicator. Whenever a `detection` event arrives over
  // the shared WebSocket for this camera, flash a pill in the corner so the
  // user can see at a glance that motion / object detection is firing —
  // *before* the slower recognition + analysis pipeline confirms anything.
  const { events: wsEvents } = useWebSocket();
  const [detectingObject, setDetectingObject] = useState<string | null>(null);
  useEffect(() => {
    if (!wsEvents.length) return;
    const newest = wsEvents[0] as any;
    if (!newest) return;
    // Match either string or numeric camera_id.
    const camMatches =
      String(newest.camera_id) === String(cameraId) ||
      newest.camera_name === cameraName;
    if (!camMatches) return;
    if (newest.type !== "detection" && newest.type !== "event_started") return;
    const label = newest.object_type || newest.event_type || "object";
    setDetectingObject(label);
    const t = setTimeout(() => setDetectingObject(null), 4000);
    return () => clearTimeout(t);
  }, [wsEvents, cameraId, cameraName]);

  return (
    <div className={`relative bg-black rounded-lg overflow-hidden ${className}`}>
      {snapUrl && !liveReady && (
        <img
          src={snapUrl}
          alt={cameraName}
          className="absolute inset-0 w-full h-full object-contain"
          onError={() => { setSnapUrl(""); }}
        />
      )}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className={`w-full h-full object-contain transition-opacity duration-200 ${liveReady ? "opacity-100" : "opacity-0"}`}
      />
      {showOverlay && status === "live" && liveReady && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
      )}
      {!hideLabel && (
        <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-0.5 rounded text-xs font-medium">
          {cameraName}
        </div>
      )}
      {detectingObject && (
        <div
          className={`absolute ${hideLabel ? "top-2" : "top-9"} left-2 flex items-center gap-1.5 bg-emerald-600/90 backdrop-blur-sm px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider shadow-lg animate-pulse`}
          title={`Detecting ${detectingObject}`}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-white" />
          <span>Detecting {detectingObject}</span>
        </div>
      )}
      {status === "live" && liveReady && (
        <>
          <div className="absolute top-2 right-2 flex items-center gap-1.5">
            <button
              onClick={() => setShowOverlay((v) => !v)}
              className={`p-1 rounded transition-colors ${
                showOverlay ? "bg-blue-600" : "bg-black/60 hover:bg-black/80"
              }`}
              title={showOverlay ? "Hide tracking boxes" : "Show tracking boxes"}
            >
              <Crosshair size={14} />
            </button>
            <div className="bg-red-600 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider">
              Live
            </div>
          </div>
          {showOverlay && (
            <button
              onClick={() => {
                const next = !showSettings;
                setShowSettings(next);
                onSettingsToggle?.(next);
              }}
              className={`absolute top-2 right-24 p-1 rounded transition-colors ${
                showSettings ? "bg-amber-600" : "bg-black/60 hover:bg-black/80"
              }`}
              title="Detection settings"
            >
              <SlidersHorizontal size={14} />
            </button>
          )}
          {showOverlay && Object.keys(tracks).length > 0 && (
            <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded text-[10px] text-slate-300">
              Tracking {Object.keys(tracks).length} object{Object.keys(tracks).length !== 1 ? "s" : ""}
            </div>
          )}
        </>
      )}
      {status === "live" && !liveReady && (
        <div className="absolute top-2 right-2 bg-amber-600 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider">
          Buffering
        </div>
      )}
      {(status === "connecting" || (status === "snapshot" && !snapUrl)) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          <p className="text-[11px] text-slate-400">Starting stream…</p>
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 gap-2">
          <p className="text-red-400 text-sm">Stream unavailable</p>
          <button
            onClick={retry}
            className="text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded flex items-center gap-1"
          >
            <RefreshCw size={12} /> Retry
          </button>
        </div>
      )}
    </div>
  );
}
