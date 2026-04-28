"""BanusNas — Camera Management API (v2: Frigate backend).

Cameras are defined in Frigate's config.yml for detection/recording.
This API manages camera metadata in Postgres and proxies Frigate endpoints.
"""

import asyncio
import base64
import logging
import re
import struct

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user
from core.config import settings
from models.database import get_session
from models.schemas import Camera, CameraCredential, CameraType, RecordingMode, User
from schemas.api_schemas import (
    AutoProbeRequest,
    AutoProbeResult,
    BulkCameraAdd,
    CameraCreate,
    CameraResponse,
    CameraStatusResponse,
    CameraUpdate,
)
from services.stream_manager import stream_manager

logger = logging.getLogger(__name__)

# Extended probe scan paths per camera type (exhaustive discovery)
PROBE_SCAN_PATHS = {
    "tapo": [f"stream{i}" for i in range(1, 9)],
    "hikvision": [
        f"Streaming/Channels/{ch}{q}" for ch in range(1, 5) for q in ("01", "02")
    ],
    "onvif": [f"stream{i}" for i in range(1, 5)]
        + ["MediaInput/h264", "MediaInput/h264/stream_1", "video1", "video2"],
}


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Extract width × height from JPEG bytes by reading SOF marker."""
    try:
        if data[:2] != b"\xff\xd8":
            return None
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                return None
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):  # SOF0 / SOF1 / SOF2
                h = struct.unpack(">H", data[i + 5 : i + 7])[0]
                w = struct.unpack(">H", data[i + 7 : i + 9])[0]
                return (w, h)
            if 0xD0 <= marker <= 0xD9:
                i += 2
            else:
                seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
                i += 2 + seg_len
        return None
    except (IndexError, struct.error):
        return None


async def _probe_single_stream(url: str, timeout: float = 12.0) -> dict:
    """Probe one RTSP URL with ffmpeg, trying TCP then UDP transport.

    Returns the result from whichever transport succeeds first (prefers TCP).
    """
    for transport in ("tcp", "udp"):
        result = await _probe_with_transport(url, transport, timeout)
        if result["available"]:
            return result
    # All transports failed — return last result (has error info)
    return result


async def _probe_with_transport(url: str, transport: str, timeout: float = 12.0) -> dict:
    """Probe one RTSP URL with a specific transport protocol."""
    result = {"snapshot": None, "codec": None, "width": None, "height": None,
              "available": False, "transport": transport}
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-rtsp_transport", transport,
            "-timeout", "6000000",          # 6s RTSP socket-level timeout (µs)
            "-i", url,
            "-frames:v", "1",
            "-f", "image2",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        err_text = stderr.decode(errors="ignore")
        m = re.search(r"Video:\s+(\w+).*?,\s+(\d{3,5})x(\d{3,5})", err_text)
        if m:
            result["codec"] = m.group(1)
            result["width"] = int(m.group(2))
            result["height"] = int(m.group(3))

        if proc.returncode == 0 and stdout and len(stdout) > 100:
            result["snapshot"] = stdout
            result["available"] = True
        elif m:
            # Stream answered (codec/res parsed) but no usable frame
            result["available"] = True
    except asyncio.TimeoutError:
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        pass
    return result


router = APIRouter(prefix="/api/cameras", tags=["cameras"], dependencies=[Depends(get_current_user)])


async def _sync_frigate(session: AsyncSession):
    """Re-generate Frigate config from all cameras and deploy it."""
    from services.frigate_config import deploy_frigate_config
    try:
        result = await session.execute(select(Camera).order_by(Camera.id))
        cameras = result.scalars().all()
        resp = await deploy_frigate_config(cameras)
        if resp["success"]:
            logger.info("Frigate config synced: %s", resp["message"])
        else:
            logger.warning("Frigate config sync issue: %s", resp["message"])
    except Exception as e:
        logger.error("Failed to sync Frigate config: %s", e)

# Camera type → go2rtc source URL templates (kept for reference/test-connection)
CAMERA_TEMPLATES = {
    "tapo": "rtsp://{username}:{password}@{host}:{port}/{stream_path}",
    "hikvision": "rtsp://{username}:{password}@{host}:{port}/{stream_path}",
    "onvif": "rtsp://{username}:{password}@{host}:{port}/{stream_path}",
    "rtsp": "{url}",
    # Ring user/password are URL-encoded in build_source_url before format() runs
    "ring": "rtsp://{ring_rtsp_user}:{ring_rtsp_password}@ring-mqtt:8554/{ring_device_id}_live#timeout=30",
    "other": "{url}",
}


def build_source_url(camera_type: str, config: dict) -> str:
    """Build RTSP source URL from camera type and connection config."""
    from urllib.parse import quote
    config = dict(config)
    template = CAMERA_TEMPLATES.get(camera_type, "{url}")
    if "ip" in config:
        config.setdefault("host", config["ip"])
    if "rtsp_url" in config:
        config.setdefault("url", config["rtsp_url"])
    if "stream_url" in config:
        config.setdefault("url", config["stream_url"])
    config.setdefault("port", "554")
    config.setdefault("channel", "101")
    config.setdefault("path", "stream1")
    config.setdefault("username", "admin")
    config.setdefault("password", "")
    config.setdefault("host", "")
    config.setdefault("url", "")
    config.setdefault("ring_device_name", "")
    if camera_type == "ring":
        config.setdefault("ring_rtsp_user", settings.ring_rtsp_user)
        config.setdefault("ring_rtsp_password", settings.ring_rtsp_password)
        # Ring passwords often contain '@', '!', '#' — URL-encode before
        # substitution into the rtsp:// template so the URL parses correctly.
        config["ring_rtsp_user"] = quote(config["ring_rtsp_user"], safe="")
        config["ring_rtsp_password"] = quote(config["ring_rtsp_password"], safe="")
    config["username"] = quote(config["username"], safe="")
    config["password"] = quote(config["password"], safe="")
    if "stream_path" not in config or not config["stream_path"]:
        if camera_type == "hikvision":
            config["stream_path"] = f"Streaming/Channels/{config['channel']}"
        elif camera_type == "onvif":
            config["stream_path"] = config["path"]
        else:
            config["stream_path"] = "stream1"
    try:
        return template.format(**config)
    except KeyError:
        return config.get("url", "")


async def _frigate_camera_status() -> dict:
    """Fetch camera status from Frigate API."""
    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=10) as client:
            resp = await client.get("/api/stats")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("cameras", {})
    except Exception as e:
        logger.debug("Failed to fetch Frigate camera status: %s", e)
    return {}


@router.get("", response_model=list[CameraStatusResponse])
async def list_cameras(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Camera).order_by(Camera.id))
    cameras = result.scalars().all()

    # Get Frigate status for all cameras
    frigate_status = await _frigate_camera_status()

    responses = []
    for cam in cameras:
        resp = CameraStatusResponse.model_validate(cam)
        stream_name = f"camera_{cam.id}"
        cam_status = frigate_status.get(stream_name, {})
        # Frigate provides camera_fps > 0 when actively detecting
        resp.is_detecting = cam_status.get("camera_fps", 0) > 0
        resp.is_recording = cam.recording_mode != RecordingMode.disabled
        responses.append(resp)
    return responses


@router.get("/{camera_id}", response_model=CameraStatusResponse)
async def get_camera(camera_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    frigate_status = await _frigate_camera_status()
    stream_name = f"camera_{camera.id}"
    cam_status = frigate_status.get(stream_name, {})

    resp = CameraStatusResponse.model_validate(camera)
    resp.is_detecting = cam_status.get("camera_fps", 0) > 0
    resp.is_recording = camera.recording_mode != RecordingMode.disabled
    return resp


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def add_camera(data: CameraCreate, session: AsyncSession = Depends(get_session)):
    connection_config = await _merge_credential(session, dict(data.connection_config), data.credential_id)
    camera = Camera(
        name=data.name,
        camera_type=CameraType(data.camera_type),
        connection_config=connection_config,
        recording_mode=RecordingMode(data.recording_mode),
        detection_enabled=data.detection_enabled,
        detection_objects=data.detection_objects,
        detection_confidence=data.detection_confidence,
        detection_settings=data.detection_settings,
        ptz_mode=data.ptz_mode,
        ptz_config=data.ptz_config,
        zones=data.zones,
    )
    session.add(camera)
    await session.commit()
    await session.refresh(camera)

    # Register stream in go2rtc (non-fatal — camera is already saved)
    source_url = build_source_url(data.camera_type, connection_config)
    stream_name = f"camera_{camera.id}"
    try:
        await stream_manager.add_stream(stream_name, source_url)
    except Exception as e:
        logger.warning("Failed to register stream for camera %s: %s", camera.id, e)

    # Sync Frigate config
    await _sync_frigate(session)

    return camera


async def _merge_credential(session: AsyncSession, config: dict, credential_id: int | None) -> dict:
    """If credential_id is provided, merge its username/password into config.

    Explicit username/password already in config take precedence (so users can
    override on a per-camera basis).
    """
    if not credential_id:
        return config
    result = await session.execute(select(CameraCredential).where(CameraCredential.id == credential_id))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(status_code=404, detail=f"Credential {credential_id} not found")
    if not config.get("username"):
        config["username"] = cred.username
    if not config.get("password"):
        config["password"] = cred.password
    return config


class _ConnectionTestRequest(BaseModel):
    camera_type: str
    connection_config: dict = {}


@router.post("/test-connection")
async def test_connection(data: _ConnectionTestRequest):
    """Test camera connection and return a preview snapshot without saving."""
    source_url = build_source_url(data.camera_type, data.connection_config)
    logger.info("test-connection source_url: %s", source_url)
    if not source_url or source_url in ("", "rtsp://admin:@:554/stream1"):
        raise HTTPException(status_code=400, detail="Invalid connection config")

    # Quick TCP connectivity check for RTSP-based cameras
    host = data.connection_config.get("ip", data.connection_config.get("host", ""))
    port = int(data.connection_config.get("port", "554"))
    tcp_ok = False
    if host:
        import socket
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            tcp_ok = True
            logger.info("TCP connect to %s:%d OK", host, port)
        except Exception as e:
            logger.warning("TCP connect to %s:%d failed: %s", host, port, e)
            return {
                "success": False,
                "message": f"Cannot reach camera at {host}:{port} — {e}",
                "source_url": source_url,
            }

    stream_name = "__preview__"
    try:
        added = await stream_manager.add_stream(stream_name, source_url)
        if not added:
            return {"success": False, "message": "Could not register stream in go2rtc", "source_url": source_url}

        # Check go2rtc stream status for diagnostics
        await asyncio.sleep(3)
        streams = await stream_manager.get_streams()
        stream_info = streams.get(stream_name, {})
        logger.info("go2rtc stream status for %s: %s", stream_name, str(stream_info)[:500])

        # Poll for snapshot (Tapo cameras can be slow — up to 15s)
        snapshot = None
        for attempt in range(6):
            await asyncio.sleep(2.5)
            snapshot = await stream_manager.get_snapshot(stream_name)
            if snapshot:
                break
            logger.info("Waiting for snapshot attempt %d/6...", attempt + 1)

        await stream_manager.remove_stream(stream_name)
        if snapshot:
            return {
                "success": True,
                "snapshot": base64.b64encode(snapshot).decode(),
                "source_url": source_url,
            }

        # Build detailed error message
        producers = stream_info.get("producers") if isinstance(stream_info, dict) else None
        if producers is not None and len(producers) == 0:
            detail = "go2rtc could not connect to camera source — check credentials (Tapo requires Camera Account password from app settings)"
        else:
            detail = "Stream registered but no video frames received"
        return {
            "success": False,
            "message": detail,
            "source_url": source_url,
            "go2rtc_status": stream_info if isinstance(stream_info, dict) else str(stream_info),
        }
    except Exception as e:
        try:
            await stream_manager.remove_stream(stream_name)
        except Exception:
            pass
        logger.warning("Connection test failed for %s: %s", source_url[:80], e)
        return {"success": False, "message": str(e), "source_url": source_url}


@router.post("/probe-streams")
async def probe_streams(data: _ConnectionTestRequest):
    """Probe camera for all available RTSP streams with snapshots & resolution.

    Uses ffmpeg directly for reliable snapshot capture instead of go2rtc's
    frame.jpeg (which returns empty in Frigate 0.17's embedded go2rtc).
    Concurrency is limited to 2 to avoid overwhelming camera RTSP limits.
    """
    scan_paths = PROBE_SCAN_PATHS.get(
        data.camera_type,
        [f"stream{i}" for i in range(1, 5)],
    )
    if not scan_paths:
        return {"streams": [], "message": "No known stream paths for this camera type"}

    # Build probe descriptors
    probes = []
    for path in scan_paths:
        test_config = dict(data.connection_config)
        test_config["stream_path"] = path
        source_url = build_source_url(data.camera_type, test_config)
        probes.append({"path": path, "url": source_url})

    # Probe with limited concurrency (cameras typically support ~2 RTSP sessions)
    sem = asyncio.Semaphore(2)

    async def _guarded_probe(url: str) -> dict:
        async with sem:
            return await _probe_single_stream(url)

    probe_results = await asyncio.gather(
        *[_guarded_probe(p["url"]) for p in probes],
        return_exceptions=True,
    )

    results = []
    for i, probe in enumerate(probes):
        pr = probe_results[i]
        if isinstance(pr, Exception):
            pr = {"snapshot": None, "codec": None, "width": None, "height": None, "available": False}

        snap = pr.get("snapshot")
        width = pr.get("width")
        height = pr.get("height")
        codec = pr.get("codec")
        available = pr.get("available", False)

        if snap:
            status = "ok"
        elif available:
            status = "no_frame"
        else:
            status = "error"

        # Quality tier based on resolution
        quality: str | None = None
        if width and height:
            pixels = max(width, height)
            if pixels >= 1920:
                quality = "hd"
            elif pixels >= 1280:
                quality = "sd"
            elif pixels >= 640:
                quality = "sd"
            else:
                quality = "low"

        results.append({
            "path": probe["path"],
            "available": available,
            "status": status,
            "source_url": probe["url"],
            "snapshot": base64.b64encode(snap).decode() if snap else None,
            "width": width,
            "height": height,
            "codec": codec,
            "quality": quality,
            "transport": pr.get("transport", "tcp"),
        })

    return {"streams": results}


class _RescanRequest(BaseModel):
    camera_type: str
    connection_config: dict = {}
    paths: list[str]


@router.post("/probe-streams/rescan")
async def rescan_streams(data: _RescanRequest):
    """Re-probe specific stream paths (e.g. previously-hidden/error streams)."""
    if not data.paths:
        return {"streams": []}

    probes = []
    for path in data.paths:
        test_config = dict(data.connection_config)
        test_config["stream_path"] = path
        source_url = build_source_url(data.camera_type, test_config)
        probes.append({"path": path, "url": source_url})

    sem = asyncio.Semaphore(2)

    async def _guarded_probe(url: str) -> dict:
        async with sem:
            return await _probe_single_stream(url)

    probe_results = await asyncio.gather(
        *[_guarded_probe(p["url"]) for p in probes],
        return_exceptions=True,
    )

    results = []
    for i, probe in enumerate(probes):
        pr = probe_results[i]
        if isinstance(pr, Exception):
            pr = {"snapshot": None, "codec": None, "width": None, "height": None, "available": False}

        snap = pr.get("snapshot")
        width = pr.get("width")
        height = pr.get("height")
        codec = pr.get("codec")
        available = pr.get("available", False)

        if snap:
            status = "ok"
        elif available:
            status = "no_frame"
        else:
            status = "error"

        quality: str | None = None
        if width and height:
            pixels = max(width, height)
            if pixels >= 1920:
                quality = "hd"
            elif pixels >= 1280:
                quality = "sd"
            elif pixels >= 640:
                quality = "sd"
            else:
                quality = "low"

        results.append({
            "path": probe["path"],
            "available": available,
            "status": status,
            "source_url": probe["url"],
            "snapshot": base64.b64encode(snap).decode() if snap else None,
            "width": width,
            "height": height,
            "codec": codec,
            "quality": quality,
            "transport": pr.get("transport", "tcp"),
        })

    return {"streams": results}


class _ScanLanRequest(BaseModel):
    subnet_prefix: str = ""  # e.g. "192.168.68"


@router.post("/scan-lan")
async def scan_lan(data: _ScanLanRequest, session: AsyncSession = Depends(get_session)):
    """Scan local network for cameras via common RTSP/ONVIF ports."""
    import socket

    subnet = ""

    # 1. Use explicitly provided subnet prefix
    if data.subnet_prefix:
        # Validate: must look like "X.X.X" with valid octets
        octets = data.subnet_prefix.split(".")
        if len(octets) == 3 and all(o.isdigit() and 0 <= int(o) <= 255 for o in octets):
            subnet = data.subnet_prefix + "."
        else:
            return {"devices": [], "error": "Invalid subnet prefix — use format like 192.168.1"}

    # 2. Auto-detect from existing cameras' IPs
    if not subnet:
        result = await session.execute(select(Camera).where(Camera.enabled == True))
        cameras = result.scalars().all()
        for cam in cameras:
            cam_ip = (cam.connection_config or {}).get("ip", "")
            if cam_ip and not cam_ip.startswith("172.") and not cam_ip.startswith("10."):
                parts = cam_ip.rsplit(".", 1)
                if len(parts) == 2:
                    subnet = parts[0] + "."
                    break

    # 3. Fallback: container's own network (Docker bridge — least useful)
    if not subnet:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            subnet = local_ip.rsplit(".", 1)[0] + "."
        except Exception:
            return {"devices": [], "error": "Cannot determine local network — provide subnet_prefix"}

    local_ip = subnet + "1"  # Gateway assumed

    # Known camera ports
    scan_ports = [
        (554, "rtsp", "RTSP"),
        (80, "http", "HTTP (ONVIF/Web UI)"),
        (8080, "http", "HTTP Alt"),
        (8554, "rtsp", "RTSP Alt"),
        (37777, "tcp", "Dahua"),
        (34567, "tcp", "XMEye"),
    ]

    # Known camera MAC OUI prefixes
    CAMERA_OUIS: dict[str, list[str]] = {
        "tapo": ["54:af:97", "98:25:4a", "5c:e9:31", "30:de:4b", "e8:48:b8", "b0:a7:b9"],
        "hikvision": ["c0:56:e3", "44:19:b6", "18:68:cb", "54:c4:15", "bc:ad:28", "c4:2f:90"],
        "dahua": ["3c:ef:8c", "40:2c:76", "a0:bd:1d", "e0:50:8b"],
        "reolink": ["ec:71:db", "b4:6d:83"],
        "amcrest": ["9c:8e:cd"],
        "axis": ["ac:cc:8e", "00:40:8c"],
        "uniview": ["24:24:04"],
    }
    oui_lookup: dict[str, str] = {}
    for brand, ouis in CAMERA_OUIS.items():
        for oui in ouis:
            oui_lookup[oui.lower()] = brand

    devices: list[dict] = []

    async def check_host(ip: str):
        open_ports = []
        for port, proto, label in scan_ports:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, port), timeout=1.0
                )
                writer.close()
                await writer.wait_closed()
                open_ports.append({"port": port, "protocol": proto, "service": label})
            except Exception:
                continue

        if not open_ports:
            return

        has_rtsp = any(p["port"] in (554, 8554) for p in open_ports)
        has_http = any(p["port"] in (80, 8080) for p in open_ports)
        has_dahua = any(p["port"] == 37777 for p in open_ports)
        has_xmeye = any(p["port"] == 34567 for p in open_ports)

        # Try ARP lookup for MAC address
        mac_addr = None
        camera_brand = None
        try:
            with open("/proc/net/arp") as f:
                for line in f:
                    if ip in line:
                        cols = line.split()
                        if len(cols) >= 4:
                            mac_addr = cols[3].lower()
                            oui = ":".join(mac_addr.split(":")[:3])
                            camera_brand = oui_lookup.get(oui)
                        break
        except Exception:
            pass

        inferred_type = "unknown"
        if camera_brand:
            inferred_type = camera_brand
        elif has_dahua:
            inferred_type = "dahua"
        elif has_xmeye:
            inferred_type = "xmeye"
        elif has_rtsp and has_http:
            inferred_type = "ip_camera"
        elif has_rtsp:
            inferred_type = "rtsp"

        # Filter out non-camera hosts (e.g. NAS, routers, PCs) — keep only devices
        # with at least one camera-specific signal: RTSP port, vendor-specific
        # control port, or known camera MAC OUI. HTTP-only hosts get dropped.
        if not (has_rtsp or has_dahua or has_xmeye or camera_brand):
            return

        devices.append({
            "ip": ip,
            "ports": open_ports,
            "mac": mac_addr,
            "brand": camera_brand,
            "inferred_type": inferred_type,
            "has_rtsp": has_rtsp,
            "has_http": has_http,
        })

    # Scan in batches of 25
    ips = [f"{subnet}{i}" for i in range(1, 255)]
    batch_size = 25
    for i in range(0, len(ips), batch_size):
        batch = ips[i:i + batch_size]
        await asyncio.gather(*[check_host(ip) for ip in batch], return_exceptions=True)

    devices.sort(key=lambda d: [int(x) for x in d["ip"].split(".")])

    return {
        "local_ip": local_ip,
        "subnet": f"{subnet}0/24",
        "devices": devices,
        "scanned": len(ips),
    }


# Brand / inferred-type → known camera_type used for path templates
_BRAND_TO_TYPE = {
    "tapo": "tapo",
    "hikvision": "hikvision",
    "dahua": "onvif",
    "reolink": "onvif",
    "amcrest": "onvif",
    "axis": "onvif",
    "uniview": "onvif",
    "ip_camera": "onvif",
    "rtsp": "onvif",
    "xmeye": "onvif",
    "unknown": "onvif",
}


async def _try_credential_against_device(
    ip: str,
    camera_type: str,
    username: str,
    password: str,
    *,
    port: int = 554,
    per_path_timeout: float = 6.0,
) -> dict:
    """Probe known stream paths for `camera_type` with one (user, pass) combo.

    Returns {success, stream_path, source_url, snapshot, width, height, codec, error}.
    Stops at first working path, then probes ALL remaining paths in parallel
    (capped) to find a smaller sub-stream for detection.
    """
    paths = PROBE_SCAN_PATHS.get(camera_type, [f"stream{i}" for i in range(1, 5)])
    base_config = {
        "ip": ip, "host": ip,
        "username": username, "password": password,
        "port": str(port),
    }

    # Probe all paths concurrently with a small semaphore (cameras choke on
    # too many simultaneous RTSP sessions). We use shorter timeout than the
    # interactive probe-streams endpoint so a 24-host scan stays fast.
    sem = asyncio.Semaphore(2)

    async def _probe_one(path: str):
        async with sem:
            cfg = dict(base_config)
            cfg["stream_path"] = path
            url = build_source_url(camera_type, cfg)
            res = await _probe_with_transport(url, "tcp", timeout=per_path_timeout)
            res["path"] = path
            res["url"] = url
            return res

    results = await asyncio.gather(*[_probe_one(p) for p in paths], return_exceptions=True)
    valid = []
    for r in results:
        if isinstance(r, Exception):
            continue
        if r.get("available") and r.get("snapshot"):
            valid.append(r)

    if not valid:
        return {"success": False, "error": "no path returned a frame"}

    # Sort by resolution descending — biggest is "record", smallest is "detect"
    valid.sort(key=lambda r: (r.get("width") or 0) * (r.get("height") or 0), reverse=True)
    main = valid[0]
    sub = valid[-1] if len(valid) > 1 else valid[0]
    return {
        "success": True,
        "stream_path": main["path"],
        "sub_stream_path": sub["path"],
        "source_url": main["url"],
        "snapshot": main["snapshot"],
        "width": main.get("width"),
        "height": main.get("height"),
        "codec": main.get("codec"),
    }


@router.post("/scan-lan/auto-probe", response_model=list[AutoProbeResult])
async def auto_probe_devices(
    data: AutoProbeRequest,
    session: AsyncSession = Depends(get_session),
):
    """Try saved (and optional ad-hoc) credentials against scanned devices.

    For each device, tests every credential whose `camera_type` matches (or has
    no type set, meaning "any"). Returns the first credential that produces a
    decodable RTSP frame, plus the discovered main + sub stream paths and a
    base64 snapshot — ready to be pasted into a bulk-add call.
    """
    # Load saved credentials
    saved: list[CameraCredential] = []
    if data.credential_ids:
        result = await session.execute(
            select(CameraCredential).where(CameraCredential.id.in_(data.credential_ids))
        )
        saved = list(result.scalars().all())

    # Build the list of (id|None, name, camera_type|None, username, password) tuples to try
    cred_list: list[tuple[int | None, str, str | None, str, str]] = [
        (c.id, c.name, c.camera_type, c.username, c.password) for c in saved
    ]
    for ec in data.extra_credentials:
        cred_list.append((None, ec.name, ec.camera_type, ec.username, ec.password or ""))

    if not cred_list:
        raise HTTPException(status_code=400, detail="No credentials provided")

    # Existing camera IPs — skip duplicates
    existing_result = await session.execute(select(Camera))
    existing_ips = set()
    for cam in existing_result.scalars().all():
        ip = (cam.connection_config or {}).get("ip") or (cam.connection_config or {}).get("host")
        if ip:
            existing_ips.add(ip)

    results: list[AutoProbeResult] = []

    # Process devices sequentially (each one already probes paths in parallel,
    # and trying multiple devices at once would overload the LAN).
    for dev in data.devices:
        if dev.ip in existing_ips:
            results.append(AutoProbeResult(
                ip=dev.ip,
                camera_type=dev.camera_type or "unknown",
                success=False,
                error="already added",
            ))
            continue

        camera_type = _BRAND_TO_TYPE.get(dev.camera_type or "unknown", "onvif")

        # Try every selected credential against this device. Order them so
        # type-matching creds are tried first, then untyped ("any") creds, then
        # type-mismatched ones as a last-ditch fallback. We do NOT skip on
        # type-mismatch — brand auto-detection is unreliable (ARP table is often
        # empty inside the container) and the user explicitly picked these creds.
        def _cred_priority(c):
            cred_type = c[2]
            if cred_type == camera_type:
                return 0          # exact type match
            if cred_type is None:
                return 1          # untyped / any
            return 2              # different type — still try as fallback

        ordered_creds = sorted(cred_list, key=_cred_priority)

        last_error = "no credentials worked"
        chosen: AutoProbeResult | None = None
        for cred_id, cred_name, cred_type, username, password in ordered_creds:
            # If the credential is tied to a specific brand and the device's
            # inferred type was the generic "onvif" fallback, probe under the
            # credential's brand instead — its path templates are usually a
            # superset and will still find the stream.
            probe_type = camera_type
            if cred_type and cred_type != camera_type and camera_type == "onvif":
                probe_type = cred_type
            try:
                probe = await _try_credential_against_device(
                    dev.ip, probe_type, username, password, port=dev.port or 554
                )
            except Exception as e:
                last_error = str(e)
                continue
            if probe["success"]:
                chosen = AutoProbeResult(
                    ip=dev.ip,
                    camera_type=probe_type,
                    success=True,
                    credential_id=cred_id,
                    credential_name=cred_name,
                    username=username,
                    stream_path=probe["stream_path"],
                    sub_stream_path=probe["sub_stream_path"],
                    source_url=probe["source_url"],
                    snapshot=base64.b64encode(probe["snapshot"]).decode() if probe.get("snapshot") else None,
                    width=probe.get("width"),
                    height=probe.get("height"),
                    codec=probe.get("codec"),
                    suggested_name=f"Camera {dev.ip}",
                )
                break
            last_error = probe.get("error") or last_error

        if not chosen:
            chosen = AutoProbeResult(
                ip=dev.ip,
                camera_type=camera_type,
                success=False,
                error=last_error,
            )
        results.append(chosen)

    return results


@router.post("/bulk-add", response_model=list[CameraResponse], status_code=status.HTTP_201_CREATED)
async def bulk_add_cameras(
    data: BulkCameraAdd,
    session: AsyncSession = Depends(get_session),
):
    """Create many cameras at once (e.g. from auto-probe results).

    Frigate config is synced once at the end (instead of after each camera).
    Per-camera failures are collected — partial success is allowed.
    """
    if not data.cameras:
        return []

    created: list[Camera] = []
    failures: list[dict] = []

    for cam_data in data.cameras:
        try:
            connection_config = await _merge_credential(session, dict(cam_data.connection_config), cam_data.credential_id)
            camera = Camera(
                name=cam_data.name,
                camera_type=CameraType(cam_data.camera_type),
                connection_config=connection_config,
                recording_mode=RecordingMode(cam_data.recording_mode),
                detection_enabled=cam_data.detection_enabled,
                detection_objects=cam_data.detection_objects,
                detection_confidence=cam_data.detection_confidence,
                detection_settings=cam_data.detection_settings,
                ptz_mode=cam_data.ptz_mode,
                ptz_config=cam_data.ptz_config,
                zones=cam_data.zones,
            )
            session.add(camera)
            await session.commit()
            await session.refresh(camera)
            created.append(camera)

            # Register go2rtc stream
            source_url = build_source_url(cam_data.camera_type, connection_config)
            try:
                await stream_manager.add_stream(f"camera_{camera.id}", source_url)
            except Exception as e:
                logger.warning("Failed to register stream for camera %s: %s", camera.id, e)
        except Exception as e:
            await session.rollback()
            logger.warning("bulk-add: failed to create camera %s: %s", cam_data.name, e)
            failures.append({"name": cam_data.name, "error": str(e)})

    if created:
        await _sync_frigate(session)

    if failures and not created:
        raise HTTPException(status_code=400, detail={"failures": failures})

    return created


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(camera_id: int, data: CameraUpdate, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    update_data = data.model_dump(exclude_unset=True)
    stream_name = f"camera_{camera.id}"

    # Track if stream config changed
    stream_changed = "connection_config" in update_data or "camera_type" in update_data

    for key, value in update_data.items():
        if key == "camera_type":
            setattr(camera, key, CameraType(value))
        elif key == "recording_mode":
            setattr(camera, key, RecordingMode(value))
        else:
            setattr(camera, key, value)

    session.add(camera)
    await session.commit()
    await session.refresh(camera)

    # Re-register stream in go2rtc if connection changed
    if stream_changed:
        try:
            await stream_manager.remove_stream(stream_name)
            source_url = build_source_url(camera.camera_type.value, camera.connection_config)
            await stream_manager.add_stream(stream_name, source_url)
        except Exception as e:
            logger.warning("Failed to re-register stream for camera %s: %s", camera.id, e)

    # Sync Frigate config
    await _sync_frigate(session)

    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(camera_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    stream_name = f"camera_{camera.id}"

    # Remove go2rtc stream (non-fatal — camera will be deleted regardless)
    try:
        await stream_manager.remove_stream(stream_name)
    except Exception as e:
        logger.warning("Failed to remove stream for camera %s: %s", camera.id, e)

    await session.delete(camera)
    await session.commit()

    # Sync Frigate config
    await _sync_frigate(session)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{camera_id}/test")
async def test_camera(camera_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    source_url = build_source_url(camera.camera_type.value, camera.connection_config)
    ok = await stream_manager.test_stream(source_url)
    return {"success": ok, "source_url": source_url}


# Short-lived cache so high-frequency /tracks polling from multiple clients
# (or multiple cameras open) doesn't hammer the Frigate API. ~200ms is well
# below human perception but coalesces bursts of requests into one upstream call.
_tracks_cache: dict = {"ts": 0.0, "events": []}
_tracks_cache_lock = asyncio.Lock()
_TRACKS_CACHE_TTL = 0.2  # seconds


async def _get_cached_active_events():
    from services.frigate_bridge import frigate_bridge
    import time as _time
    now = _time.time()
    if now - _tracks_cache["ts"] < _TRACKS_CACHE_TTL:
        return _tracks_cache["events"]
    async with _tracks_cache_lock:
        # Re-check inside the lock in case another coroutine just refreshed.
        now = _time.time()
        if now - _tracks_cache["ts"] < _TRACKS_CACHE_TTL:
            return _tracks_cache["events"]
        events = await frigate_bridge.get_active_frigate_events()
        _tracks_cache["events"] = events
        _tracks_cache["ts"] = now
        return events


@router.get("/{camera_id}/tracks")
async def get_active_tracks(camera_id: int):
    """Return current tracked objects from Frigate in-progress events.

    bbox values are normalized 0-1 [x1, y1, x2, y2].
    Results are cached briefly (~200ms) to allow ~4 Hz polling without
    overloading Frigate.
    """
    try:
        events = await _get_cached_active_events()
        camera_name = f"camera_{camera_id}"
        result = {}
        for ev in events:
            if ev["frigate_camera"] == camera_name and ev["in_progress"]:
                result[ev["frigate_id"]] = {
                    "class_name": ev["label"],
                    "confidence": ev["score"],
                    "bbox": ev["bbox_norm"],
                    "named_object_name": ev.get("sub_label"),
                }
        return result
    except Exception as e:
        logger.debug("Failed to get Frigate tracks for camera %s: %s", camera_id, e)
    return {}


@router.post("/{camera_id}/detect-preview")
async def detect_preview(
    camera_id: int,
    data: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run YOLO on a live snapshot with provided settings and return detections + image.

    Body: { detection_confidence, detection_settings, detection_objects }
    Returns: { detections: [...], snapshot_b64: "..." }
    """
    from services.object_detector import object_detector
    import cv2
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    stream_name = f"camera_{camera.id}"
    snapshot_bytes = await stream_manager.get_snapshot(stream_name)
    if not snapshot_bytes:
        raise HTTPException(status_code=503, detail="Failed to get snapshot")

    # Decode snapshot
    import numpy as np
    nparr = np.frombuffer(snapshot_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=503, detail="Failed to decode snapshot")

    target_classes = data.get("detection_objects", camera.detection_objects)
    global_conf = data.get("detection_confidence", camera.detection_confidence)
    det_settings = data.get("detection_settings", camera.detection_settings or {})

    # Compute minimum confidence for initial YOLO pass
    per_obj_confs = [det_settings.get(cls, {}).get("confidence", global_conf) for cls in target_classes]
    min_conf = min(per_obj_confs) if per_obj_confs else global_conf

    # Run detection (tiled for better resolution on 2K/4K cameras)
    raw_dets = await object_detector.detect(img, confidence_threshold=min_conf, target_classes=target_classes, tiled=True)

    # Apply per-class confidence and min_area
    filtered = []
    for det in raw_dets:
        obj_s = det_settings.get(det.class_name, {})
        req_conf = obj_s.get("confidence", global_conf)
        min_area = obj_s.get("min_area", 0)
        if det.confidence < req_conf:
            continue
        bbox_area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
        if bbox_area < min_area:
            continue
        filtered.append(det)

    # Encode snapshot as base64
    snapshot_b64 = base64.b64encode(snapshot_bytes).decode("ascii")

    return {
        "detections": [d.to_dict() for d in filtered],
        "snapshot_b64": snapshot_b64,
        "image_width": img.shape[1],
        "image_height": img.shape[0],
    }


@router.get("/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    stream_name = f"camera_{camera.id}"
    snapshot = await stream_manager.get_snapshot(stream_name)
    if not snapshot:
        raise HTTPException(status_code=503, detail="Failed to get snapshot")

    return Response(content=snapshot, media_type="image/jpeg")


@router.post("/discover")
async def discover_cameras(subnet: str = "192.168.1.0/24", user: User = Depends(get_current_user)):
    """ONVIF camera auto-discovery (placeholder — requires onvif-zeep or ws-discovery)."""
    # TODO: Implement ONVIF WS-Discovery scan
    return {"message": "Discovery not yet implemented", "subnet": subnet, "cameras": []}


def _parse_zones(zones: dict | None) -> list | None:
    if not zones:
        return None
    return zones.get("polygons", None)


# --- COCO object classes ---

COCO_CATEGORIES = {
    "People": ["person"],
    "Animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "Outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "Accessories": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "Sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "Kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "Food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "Furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "Electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster"],
    "Indoor": ["sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
    "Other": ["package"],
}


@router.get("/detection-classes")
async def get_detection_classes():
    """Return all available COCO detection classes grouped by category."""
    all_classes = [cls for group in COCO_CATEGORIES.values() for cls in group]
    return {"classes": all_classes, "categories": COCO_CATEGORIES}


@router.get("/{camera_id}/detection-settings")
async def get_detection_settings(camera_id: int, session: AsyncSession = Depends(get_session)):
    """Get per-object detection settings for a camera."""
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {
        "camera_id": camera.id,
        "camera_name": camera.name,
        "detection_enabled": camera.detection_enabled,
        "detection_objects": camera.detection_objects,
        "detection_confidence": camera.detection_confidence,
        "detection_settings": camera.detection_settings or {},
        "ptz_mode": camera.ptz_mode,
    }


@router.put("/{camera_id}/detection-settings")
async def update_detection_settings(
    camera_id: int,
    data: dict,
    session: AsyncSession = Depends(get_session),
):
    """Update per-object detection settings for a camera."""
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # data has: detection_objects, detection_confidence, detection_settings, ptz_mode
    if "detection_objects" in data:
        camera.detection_objects = data["detection_objects"]
    if "detection_confidence" in data:
        camera.detection_confidence = data["detection_confidence"]
    if "detection_settings" in data:
        camera.detection_settings = data["detection_settings"]
    if "ptz_mode" in data:
        camera.ptz_mode = data["ptz_mode"]

    session.add(camera)
    await session.commit()
    await session.refresh(camera)

    # Sync Frigate config
    await _sync_frigate(session)

    return {
        "camera_id": camera.id,
        "detection_objects": camera.detection_objects,
        "detection_confidence": camera.detection_confidence,
        "detection_settings": camera.detection_settings or {},
        "ptz_mode": camera.ptz_mode,
    }


# ── Ignore zones: suppress detections of specific classes in defined regions ──


class IgnoreZone(BaseModel):
    polygon: list[list[int]]  # [[x1,y1], [x2,y2], ...] — at least 3 points
    classes: list[str] = []   # Empty = all classes. e.g. ["cat", "dog"]
    label: str = ""           # Optional label for UI (e.g. "Statue area")
    # Source image dimensions the polygon was drawn against — used to normalize
    # to 0..1 coordinates when synced to Frigate masks.
    image_width: int | None = None
    image_height: int | None = None

from sqlalchemy.orm.attributes import flag_modified


@router.get("/{camera_id}/ignore-zones")
async def get_ignore_zones(camera_id: int, session: AsyncSession = Depends(get_session)):
    """Get ignore zones for a camera."""
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    ds = camera.detection_settings or {}
    return {"camera_id": camera.id, "ignore_zones": ds.get("ignore_zones", [])}


@router.put("/{camera_id}/ignore-zones")
async def set_ignore_zones(
    camera_id: int,
    zones: list[IgnoreZone],
    session: AsyncSession = Depends(get_session),
):
    """Set ignore zones for a camera. Replaces all existing ignore zones."""
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    for z in zones:
        if len(z.polygon) < 3:
            raise HTTPException(status_code=422, detail="Each ignore zone polygon must have at least 3 points")

    ds = dict(camera.detection_settings or {})
    ds["ignore_zones"] = [z.model_dump() for z in zones]
    camera.detection_settings = ds
    flag_modified(camera, "detection_settings")
    session.add(camera)
    await session.commit()

    # Sync Frigate config
    await _sync_frigate(session)

    return {"camera_id": camera.id, "ignore_zones": ds["ignore_zones"]}
