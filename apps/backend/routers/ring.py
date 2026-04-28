"""BanusNas — Ring Camera Integration API.

Discovers Ring cameras via ring-mqtt MQTT topics and provides
endpoints for listing devices and adding them to the NVR.
Auth endpoints proxy to ring-mqtt's built-in web UI API (port 55123).
"""

import asyncio
import json
import logging
import os
import threading
import time as _time
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user
from core.config import settings
from models.database import get_session
from models.schemas import Camera, CameraType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ring", tags=["ring"], dependencies=[Depends(get_current_user)])

# Hostname/port of the ring-mqtt web UI inside the docker network.
# Override via RING_MQTT_HOST / RING_MQTT_WEB_PORT env vars when the
# container is renamed (e.g. banusnvr-ring-mqtt vs banusnas-ring-mqtt).
_RING_MQTT_HOST = os.getenv("RING_MQTT_HOST", "ring-mqtt")
_RING_MQTT_WEB_PORT = os.getenv("RING_MQTT_WEB_PORT", "55123")
RING_MQTT_WEB = f"http://{_RING_MQTT_HOST}:{_RING_MQTT_WEB_PORT}"


# ─────────────────────────────────────────────────────────────
# Background MQTT listener — accumulates Ring device data
# ─────────────────────────────────────────────────────────────

class _RingMQTTListener:
    """Persistent MQTT subscriber that builds a Ring device registry.

    Also acts as the Ring ↔ Frigate motion bridge: when ring-mqtt publishes
    `ring/<location>/camera/<id>/motion/state = ON`, we look up the matching
    NVR camera (by `connection_config.ring_device_id`) and publish
    `frigate/<cam>/detect/set = ON` and `frigate/<cam>/recordings/set = ON`
    on the same broker. After `_RING_MOTION_HOLD_SECS` of no further motion,
    we publish `OFF` again so Frigate stops pulling the on-demand stream.
    """

    # How long to keep Frigate detect+record enabled after the last
    # motion-on event for a given Ring camera. Long enough to capture the
    # tail of the event but short enough that Frigate doesn't keep the
    # stream open indefinitely (and burn Ring's per-day stream budget).
    _RING_MOTION_HOLD_SECS = 60

    def __init__(self):
        self._devices: dict[str, dict] = {}
        self._ha_names: dict[str, str] = {}
        self._ha_models: dict[str, str] = {}
        self._lock = threading.Lock()
        self._online = False
        self._last_ring_msg: float = 0.0
        self._started = False
        self._client = None
        # Motion bridge state — populated once main loop is running
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._device_to_cam: dict[str, str] = {}   # ring_device_id → "camera_<id>"
        self._off_tasks: dict[str, asyncio.TimerHandle] = {}  # cam_name → pending OFF call
        self._active: set[str] = set()  # cam_names currently turned ON via bridge

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        # Capture the asyncio loop so the paho thread can schedule async DB
        # lookups + Frigate toggles back on it. Safe to call again later —
        # we only honour the first non-None loop.
        if loop is not None and self._loop is None:
            self._loop = loop
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        logger.info("Ring MQTT background listener starting")

    def _run(self):
        import paho.mqtt.client as mqtt

        while True:
            try:
                self._client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id="banusnas-ring-listener",
                )
                self._client.on_connect = self._on_connect
                self._client.on_message = self._on_message
                self._client.on_disconnect = self._on_disconnect
                self._client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=30)
                self._client.loop_forever()
            except Exception as e:
                logger.warning("Ring MQTT listener error: %s — reconnecting in 10s", e)
                self._online = False
            _time.sleep(10)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe("homeassistant/#", qos=0)
            client.subscribe("ring/#", qos=0)
            logger.info("Ring MQTT listener connected and subscribed")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self._online = False

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode("utf-8", errors="replace")

            with self._lock:
                # ── ring-mqtt status ──
                if topic == "ring/hassmqtt/status":
                    self._online = payload.strip().lower() == "online"
                    return

                # Track any ring/ activity
                if topic.startswith("ring/"):
                    self._last_ring_msg = _time.time()

                # ── HA discovery camera config ──
                if topic.startswith("homeassistant/camera/") and topic.endswith("/config"):
                    data = json.loads(payload)
                    unique_id = data.get("unique_id", data.get("uniq_id", ""))
                    device_info = data.get("device", {})
                    device_name = device_info.get("name", data.get("name", ""))
                    model = device_info.get("model", device_info.get("mdl", ""))
                    device_id = unique_id.rsplit("_", 1)[0] if "_" in unique_id else unique_id
                    if device_id:
                        self._ha_names[device_id] = device_name
                        self._ha_models[device_id] = model
                        dev = self._ensure_device(device_id)
                        dev["name"] = device_name
                        dev["model"] = model
                        dev["manufacturer"] = device_info.get("mfr", device_info.get("manufacturer", "Ring"))
                        dev["unique_id"] = unique_id
                        dev["snapshot_topic"] = data.get("t", data.get("topic", ""))

                # ── HA discovery binary_sensor — ring camera motion sensors ──
                elif topic.startswith("homeassistant/binary_sensor/") and topic.endswith("/config"):
                    data = json.loads(payload)
                    device_class = data.get("device_class", data.get("dev_cla", ""))
                    unique_id = data.get("unique_id", data.get("uniq_id", ""))
                    # Only create device entries for motion sensors (= cameras)
                    if device_class == "motion" and "_motion" in unique_id:
                        device_info = data.get("device", {})
                        device_name = device_info.get("name", "")
                        model = device_info.get("model", device_info.get("mdl", ""))
                        device_id = unique_id.rsplit("_", 1)[0]
                        if device_id and device_name:
                            self._ha_names[device_id] = device_name
                            self._ha_models.setdefault(device_id, model)
                            dev = self._ensure_device(device_id)
                            if dev["name"] == device_id:
                                dev["name"] = device_name
                            if model and not dev["model"]:
                                dev["model"] = model
                            topic_parts = topic.split("/")
                            if len(topic_parts) >= 4:
                                dev.setdefault("location", topic_parts[2])

                # ── Live ring camera topics ──
                elif "/camera/" in topic and topic.startswith("ring/"):
                    parts = topic.split("/")
                    if len(parts) >= 5 and parts[2] == "camera":
                        location = parts[1]
                        device_id = parts[3]
                        dev = self._ensure_device(device_id, location)
                        subtopic = "/".join(parts[4:])
                        if subtopic == "info/state":
                            info = json.loads(payload)
                            dev.setdefault("firmware", info.get("firmwareStatus", ""))
                            dev["location"] = location
                        elif subtopic == "status":
                            dev["status"] = payload.strip()
                        elif subtopic == "motion/state":
                            # Bridge to Frigate. Done outside the lock to
                            # avoid holding it while scheduling on the loop.
                            state = payload.strip().upper()
                            self._dispatch_motion(device_id, state)

        except Exception as e:
            logger.debug("MQTT message parse error: %s", e)

    # ─────── Ring-motion → Frigate bridge ───────

    def _dispatch_motion(self, device_id: str, state: str):
        """Schedule motion handling on the asyncio loop (called from MQTT thread)."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._handle_motion(device_id, state), loop)
        except Exception as e:
            logger.debug("Could not schedule Ring motion handler: %s", e)

    async def _handle_motion(self, device_id: str, state: str):
        cam_name = self._device_to_cam.get(device_id)
        if cam_name is None:
            cam_name = await self._lookup_cam_name(device_id)
            if cam_name is None:
                # Camera with this Ring device_id isn't in our DB — ignore
                return
            self._device_to_cam[device_id] = cam_name

        if state == "ON":
            # Cancel any pending OFF and (re-)enable detect+record
            pending = self._off_tasks.pop(cam_name, None)
            if pending is not None:
                pending.cancel()
            if cam_name not in self._active:
                self._publish_frigate_state(cam_name, True)
                self._active.add(cam_name)
                logger.info("Ring motion ON → Frigate %s detect+record enabled", cam_name)
            # Always (re)arm the OFF timer so detection sticks for the full
            # hold window after the *latest* motion event.
            loop = asyncio.get_running_loop()
            self._off_tasks[cam_name] = loop.call_later(
                self._RING_MOTION_HOLD_SECS,
                lambda c=cam_name: self._motion_off(c),
            )
        elif state == "OFF":
            # Don't immediately turn Frigate off — let the hold timer run.
            # Ring's motion sensor often pulses OFF even mid-event, and the
            # interesting frames are usually the last ones.
            pass

    def _motion_off(self, cam_name: str):
        self._off_tasks.pop(cam_name, None)
        if cam_name in self._active:
            self._publish_frigate_state(cam_name, False)
            self._active.discard(cam_name)
            logger.info("Ring motion hold expired → Frigate %s detect+record disabled", cam_name)

    def _publish_frigate_state(self, cam_name: str, on: bool):
        client = self._client
        if client is None:
            return
        payload = "ON" if on else "OFF"
        try:
            client.publish(f"frigate/{cam_name}/detect/set", payload, qos=0, retain=False)
            client.publish(f"frigate/{cam_name}/recordings/set", payload, qos=0, retain=False)
        except Exception as e:
            logger.warning("Failed to publish Frigate toggle for %s: %s", cam_name, e)

    async def _lookup_cam_name(self, device_id: str) -> Optional[str]:
        """Resolve a Ring device_id → Frigate camera name via DB lookup."""
        try:
            from models.database import async_session
            async with async_session() as session:
                result = await session.execute(
                    select(Camera).where(Camera.camera_type == CameraType.ring)
                )
                for cam in result.scalars().all():
                    cfg = cam.connection_config or {}
                    if cfg.get("ring_device_id") == device_id:
                        return f"camera_{cam.id}"
        except Exception as e:
            logger.debug("Ring cam DB lookup failed for %s: %s", device_id, e)
        return None

    def invalidate_cam_cache(self):
        """Drop the device→cam map (call after a Ring camera is added/removed)."""
        self._device_to_cam.clear()

    def _ensure_device(self, device_id: str, location: str = "") -> dict:
        if device_id not in self._devices:
            self._devices[device_id] = {
                "device_id": device_id,
                "name": self._ha_names.get(device_id, device_id),
                "model": self._ha_models.get(device_id, ""),
                "manufacturer": "Ring",
                "unique_id": f"{device_id}_camera",
                "status": "online",
                "snapshot_topic": "",
            }
            if location:
                self._devices[device_id]["location"] = location
        return self._devices[device_id]

    @property
    def is_online(self) -> bool:
        """ring-mqtt is online if explicit status says so, or recent activity."""
        if self._online:
            return True
        return (_time.time() - self._last_ring_msg) < 120

    def get_devices(self) -> list[dict]:
        with self._lock:
            # Apply HA-discovered names to devices that still use raw IDs
            for did, dev in self._devices.items():
                if dev["name"] == did and did in self._ha_names:
                    dev["name"] = self._ha_names[did]
                if not dev["model"] and did in self._ha_models:
                    dev["model"] = self._ha_models[did]
            return [dict(d) for d in self._devices.values()]

    def get_status(self) -> dict:
        online = self.is_online
        if self._online:
            msg = "online"
        elif online:
            msg = "online (detected via activity)"
        elif self._started:
            msg = "No ring-mqtt activity detected"
        else:
            msg = "MQTT listener not started"
        return {"online": online, "message": msg, "hub_waiting": False}


_ring_listener = _RingMQTTListener()


def _start_ring_listener():
    """Start the background listener (called once at app startup).

    Captures the running asyncio loop so the MQTT thread can dispatch
    motion events back into async DB lookups + Frigate toggle publishes.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    _ring_listener.start(loop=loop)


@router.get("/devices")
async def discover_ring_devices():
    """Return Ring cameras discovered via the background MQTT listener."""
    _ring_listener.start()  # idempotent

    devices = _ring_listener.get_devices()

    # Check which are already added as cameras in our DB
    from models.database import async_session
    async with async_session() as session:
        result = await session.execute(
            select(Camera).where(Camera.camera_type == CameraType.ring)
        )
        existing = result.scalars().all()
        existing_names = {
            (cam.connection_config or {}).get("ring_device_name", "").lower()
            for cam in existing
        }
        existing_ids = {
            (cam.connection_config or {}).get("ring_device_id", "").lower()
            for cam in existing
        }

    for dev in devices:
        dev["already_added"] = (
            dev.get("device_id", "").lower() in existing_ids
            or dev.get("name", "").lower().replace(" ", "_") in existing_names
        )

    return {
        "devices": devices,
        "mqtt_host": settings.mqtt_host,
        "mqtt_port": settings.mqtt_port,
        "ring_rtsp_user": settings.ring_rtsp_user,
    }


class RingAddRequest(BaseModel):
    device_id: str
    name: str
    ring_device_name: str  # For RTSP URL: {name}_live


@router.post("/add-camera")
async def add_ring_camera(
    data: RingAddRequest,
    session: AsyncSession = Depends(get_session),
):
    """Add a discovered Ring camera to the NVR system.

    Note: Ring cameras must also be added to Frigate's config.yml for
    detection/recording to work. This endpoint stores metadata only.
    """
    from routers.cameras import build_source_url

    # Check for duplicates
    result = await session.execute(
        select(Camera).where(Camera.camera_type == CameraType.ring)
    )
    existing = result.scalars().all()
    for cam in existing:
        cfg = cam.connection_config or {}
        if cfg.get("ring_device_id") == data.device_id:
            raise HTTPException(status_code=409, detail="This Ring camera is already added")

    # Create camera
    connection_config = {
        "ring_device_id": data.device_id,
        "ring_device_name": data.ring_device_name,
    }

    camera = Camera(
        name=data.name,
        camera_type=CameraType.ring,
        connection_config=connection_config,
        detection_enabled=True,
        detection_objects=["person", "cat", "dog", "car"],
        detection_confidence=0.5,
    )
    session.add(camera)
    await session.commit()
    await session.refresh(camera)

    # Drop the device→camera cache so the motion bridge picks up this new
    # camera on the next motion event.
    _ring_listener.invalidate_cam_cache()

    source_url = build_source_url("ring", connection_config)
    stream_name = f"camera_{camera.id}"

    return {
        "camera_id": camera.id,
        "name": camera.name,
        "stream_name": stream_name,
        "source_url": source_url,
    }


@router.get("/status")
async def ring_mqtt_status():
    """Check ring-mqtt service status via background MQTT listener."""
    _ring_listener.start()  # idempotent
    return _ring_listener.get_status()


# ─────────────────────────────────────────────────────────────
# Auth Proxy — relay to ring-mqtt web UI at port 55123
# ─────────────────────────────────────────────────────────────

@router.get("/auth/state")
async def get_auth_state():
    """Check ring-mqtt auth state (connected, needs login, etc.).

    ring-mqtt's web UI (port 55123) only runs when authentication is needed.
    Connection refused means the web UI has shut down because a refresh token
    was already generated — i.e. auth is complete.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{RING_MQTT_WEB}/get-state")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        # Web UI not running → auth already complete (web UI shuts down after token saved)
        return {"connected": True, "web_ui_active": False}
    except Exception as e:
        logger.warning("Auth state check failed: %s", e)
        return {"connected": False, "error": str(e)}


class AccountSubmit(BaseModel):
    email: str
    password: str


@router.post("/auth/submit-account")
async def submit_account(data: AccountSubmit):
    """Submit Ring email/password to ring-mqtt for authentication."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{RING_MQTT_WEB}/submit-account",
                data={"email": data.email, "password": data.password},
            )
            result = resp.json()
            if not resp.is_success:
                raise HTTPException(status_code=resp.status_code, detail=result.get("error", "Unknown error"))
            return result
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="ring-mqtt web UI not reachable")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Account submit failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


class CodeSubmit(BaseModel):
    code: str


@router.post("/auth/submit-code")
async def submit_code(data: CodeSubmit):
    """Submit 2FA code to ring-mqtt to complete authentication."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{RING_MQTT_WEB}/submit-code",
                data={"code": data.code},
            )
            result = resp.json()
            if not resp.is_success:
                raise HTTPException(status_code=resp.status_code, detail=result.get("error", "Unknown error"))
            return result
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="ring-mqtt web UI not reachable")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Code submit failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
