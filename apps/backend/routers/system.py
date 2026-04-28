"""BanusNas — System API: status, storage, global settings, config export/import."""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import httpx

from core.auth import get_current_user
from core.config import settings
from models.database import get_session
from models.schemas import Camera, Event, SystemSettings
from schemas.api_schemas import GlobalSettings, HardwareResources, HardwareWarning, SystemStatus, PerformanceSettings, DashboardLayout, TrainingSettings
from services.storage_manager import storage_manager
from services.frigate_bridge import frigate_bridge

router = APIRouter(prefix="/api/system", tags=["system"], dependencies=[Depends(get_current_user)])

_start_time = time.time()


@router.get("/status", response_model=SystemStatus)
async def get_status(session: AsyncSession = Depends(get_session)):
    total_cameras = await session.scalar(select(func.count()).select_from(Camera)) or 0
    enabled_cameras = await session.scalar(
        select(func.count()).select_from(Camera).where(Camera.enabled == True)
    ) or 0

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    events_today = await session.scalar(
        select(func.count()).select_from(Event).where(Event.started_at >= today_start)
    ) or 0

    disk = storage_manager.get_disk_usage()

    # Get active recording count from Frigate
    active_recordings = 0
    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=5) as client:
            resp = await client.get("/api/stats")
            if resp.status_code == 200:
                stats = resp.json()
                active_recordings = sum(
                    1 for c in stats.get("cameras", {}).values()
                    if c.get("camera_fps", 0) > 0
                )
    except Exception:
        pass

    return SystemStatus(
        cameras_online=enabled_cameras,
        cameras_total=total_cameras,
        active_recordings=active_recordings,
        events_today=events_today,
        storage_used_gb=disk["used_gb"],
        storage_total_gb=disk["total_gb"],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/storage")
async def get_storage():
    return storage_manager.get_disk_usage()


@router.get("/tracking-status")
async def get_tracking_status():
    """Debug endpoint: show Frigate bridge status and presence tracking."""
    return frigate_bridge.get_status()


@router.get("/detection-metrics")
async def get_detection_metrics():
    """Runtime metrics — proxied from Frigate stats."""
    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=5) as client:
            resp = await client.get("/api/stats")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {}


def _detect_gpu() -> tuple[bool, str | None]:
    """Detect Intel iGPU via /dev/dri render nodes."""
    if not Path("/dev/dri").exists():
        return False, None
    try:
        for entry in Path("/dev/dri/by-path").iterdir():
            name = entry.name
            if "pci" in name:
                return True, "Intel iGPU"
    except (FileNotFoundError, PermissionError):
        pass
    # Fallback: if render nodes exist, assume GPU is available
    render_nodes = list(Path("/dev/dri").glob("renderD*"))
    if render_nodes:
        return True, "Intel iGPU"
    return False, None


def _read_gpu_usage() -> float | None:
    """Try to read Intel GPU usage from /sys/class/drm/card0/gt/gt0."""
    try:
        # Try intel_gpu_top-style busy percentage via sysfs
        busy_path = Path("/sys/class/drm/card0/gt/gt0/rcs0/busy")
        total_path = Path("/sys/class/drm/card0/gt/gt0/rcs0/total")
        if busy_path.exists() and total_path.exists():
            busy = int(busy_path.read_text().strip())
            total = int(total_path.read_text().strip())
            if total > 0:
                return round(busy / total * 100, 1)
    except (ValueError, PermissionError, FileNotFoundError):
        pass

    try:
        # Alternative: /sys/kernel/debug/dri/0/i915_engine_info
        # Or parse /sys/devices/pci*/*/drm/card0/gt_act_freq_mhz vs gt_max_freq_mhz
        act_path = None
        for p in Path("/sys/devices").rglob("gt_act_freq_mhz"):
            act_path = p
            break
        if act_path:
            max_path = act_path.parent / "gt_max_freq_mhz"
            if max_path.exists():
                act = int(act_path.read_text().strip())
                mx = int(max_path.read_text().strip())
                if mx > 0:
                    return round(act / mx * 100, 1)
    except (ValueError, PermissionError, FileNotFoundError):
        pass
    return None


def _estimate_capacity(cpu_cores: int, ram_gb: float, has_gpu: bool) -> tuple[int, int]:
    """Estimate max cameras: (relay, transcode) based on detected hardware."""
    # Relay (passthrough): Limited mainly by network/memory, ~4 streams per core
    max_relay = min(cpu_cores * 4, int(ram_gb * 3))
    # Transcode: GPU can handle ~8 simultaneous 1080p transcodes on N100
    # CPU-only: ~1 per 2 cores
    if has_gpu:
        max_transcode = max(8, cpu_cores)  # QSV handles ~8 concurrent 1080p
    else:
        max_transcode = max(1, cpu_cores // 2)
    return max_relay, max_transcode


def _read_proc_stat() -> list[int]:
    """Read aggregate CPU jiffies from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
            # user, nice, system, idle, iowait, irq, softirq, steal
            return [int(x) for x in parts[1:9]]
    except (FileNotFoundError, PermissionError, ValueError):
        return [0, 0, 0, 100, 0, 0, 0, 0]

# Cached CPU percentage to avoid 0.5s sleep on every request
_cpu_cache: dict[str, float] = {"percent": 0.0}
_cpu_prev_stat: list[int] = _read_proc_stat()

# Cache static hardware info (CPU name/cores, GPU) — doesn't change at runtime
_hw_static_cache: dict | None = None


def _get_hw_static() -> dict:
    global _hw_static_cache
    if _hw_static_cache is not None:
        return _hw_static_cache

    cpu_name = "Unknown"
    cpu_cores = 1
    try:
        with open("/proc/cpuinfo") as f:
            core_ids = set()
            for line in f:
                if line.startswith("model name") and cpu_name == "Unknown":
                    cpu_name = line.split(":", 1)[1].strip()
                if line.startswith("processor"):
                    core_ids.add(line.split(":", 1)[1].strip())
            cpu_cores = len(core_ids) or 1
    except (FileNotFoundError, PermissionError):
        pass

    gpu_available, gpu_name = _detect_gpu()

    _hw_static_cache = {
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
    }
    return _hw_static_cache


def _read_meminfo() -> dict[str, int]:
    """Read /proc/meminfo and return values in kB."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return info


@router.get("/resources", response_model=HardwareResources)
async def get_resources(session: AsyncSession = Depends(get_session)):
    """Real-time hardware resource monitoring with adaptive warnings."""
    # CPU usage: use cached value (updated by _refresh_cpu_cache below)
    global _cpu_prev_stat
    s2 = _read_proc_stat()
    s1 = _cpu_prev_stat
    idle1, idle2 = s1[3] + s1[4], s2[3] + s2[4]
    total1, total2 = sum(s1), sum(s2)
    d_total = total2 - total1
    cpu_percent = (1.0 - (idle2 - idle1) / d_total) * 100 if d_total > 0 else _cpu_cache["percent"]
    if d_total > 0:
        _cpu_cache["percent"] = cpu_percent
    _cpu_prev_stat = s2

    # Static hardware info (cached — read once)
    hw = _get_hw_static()
    cpu_name = hw["cpu_name"]
    cpu_cores = hw["cpu_cores"]

    # RAM from /proc/meminfo
    mem = _read_meminfo()
    mem_total_kb = mem.get("MemTotal", 0)
    mem_avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
    mem_used_kb = mem_total_kb - mem_avail_kb
    ram_total_gb = mem_total_kb / (1024 * 1024)
    ram_used_gb = mem_used_kb / (1024 * 1024)
    ram_pct = (mem_used_kb / mem_total_kb * 100) if mem_total_kb > 0 else 0.0

    disk = storage_manager.get_disk_usage()
    gpu_available = hw["gpu_available"]
    gpu_name = hw["gpu_name"]

    # Camera counts
    total_cameras = await session.scalar(
        select(func.count()).select_from(Camera).where(Camera.enabled == True)
    ) or 0
    cameras_transcode = 0
    cameras_relay = total_cameras

    max_relay, max_transcode = _estimate_capacity(cpu_cores, ram_total_gb, gpu_available)

    # Build warnings
    warnings: list[HardwareWarning] = []

    if cpu_percent > 90:
        warnings.append(HardwareWarning(
            level="critical", category="cpu",
            message=f"CPU at {cpu_percent:.0f}% \u2014 system may become unresponsive",
            value=cpu_percent,
        ))
    elif cpu_percent > 70:
        warnings.append(HardwareWarning(
            level="warning", category="cpu",
            message=f"CPU at {cpu_percent:.0f}% \u2014 consider reducing cameras or detection FPS",
            value=cpu_percent,
        ))

    if ram_pct > 90:
        warnings.append(HardwareWarning(
            level="critical", category="ram",
            message=f"RAM at {ram_pct:.0f}% ({ram_used_gb:.1f}/{ram_total_gb:.1f} GB) \u2014 add more RAM or reduce cameras",
            value=ram_pct, limit=ram_total_gb,
        ))
    elif ram_pct > 75:
        warnings.append(HardwareWarning(
            level="warning", category="ram",
            message=f"RAM at {ram_pct:.0f}% \u2014 approaching limit",
            value=ram_pct, limit=ram_total_gb,
        ))

    storage_pct = (disk["used_gb"] / disk["total_gb"] * 100) if disk["total_gb"] > 0 else 0
    if storage_pct > 95:
        warnings.append(HardwareWarning(
            level="critical", category="storage",
            message=f"Storage at {storage_pct:.0f}% \u2014 recordings may fail, reduce retention",
            value=storage_pct, limit=disk["total_gb"],
        ))
    elif storage_pct > 80:
        warnings.append(HardwareWarning(
            level="warning", category="storage",
            message=f"Storage at {storage_pct:.0f}% \u2014 consider reducing retention days",
            value=storage_pct, limit=disk["total_gb"],
        ))

    if total_cameras >= max_relay:
        warnings.append(HardwareWarning(
            level="critical", category="cameras",
            message=f"{total_cameras} cameras at relay limit ({max_relay}) for this hardware",
            value=total_cameras, limit=max_relay,
        ))
    elif total_cameras >= max_relay * 0.75:
        warnings.append(HardwareWarning(
            level="warning", category="cameras",
            message=f"{total_cameras}/{max_relay} camera capacity \u2014 approaching limit",
            value=total_cameras, limit=max_relay,
        ))

    if not gpu_available:
        warnings.append(HardwareWarning(
            level="info", category="gpu",
            message="No GPU detected \u2014 transcoding will use CPU (slower)",
            value=0,
        ))

    # GPU usage
    gpu_percent = _read_gpu_usage() if gpu_available else None

    # Detector status from Frigate (supports Coral, OpenVINO, TensorRT, etc.)
    coral_available = False
    coral_status = None
    gpu_inference_device = None
    detector_type: str | None = None      # "edgetpu" | "openvino" | "cpu" | ...
    detector_devices: list[str] = []      # human-readable accelerator list

    # First, probe the host for physical Coral devices regardless of Frigate's
    # current state — lets the UI show "Coral detected" even before Frigate
    # has finished loading.
    try:
        from services.frigate_config import detect_coral_devices
        coral_hw = detect_coral_devices()
        if coral_hw["usb"]:
            coral_available = True
            detector_devices.append("Coral USB")
        for idx in coral_hw["pcie"]:
            coral_available = True
            detector_devices.append(f"Coral M.2 #{idx}")
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=5) as client:
            resp = await client.get("/api/stats")
            if resp.status_code == 200:
                stats = resp.json()
                detectors = stats.get("detectors", {})
                for name, det in detectors.items():
                    coral_status = det
                    gpu_inference_device = name
                    # Detector name conventions: coral_usb, coral_pci0, openvino, ov_0...
                    if name.startswith("coral"):
                        detector_type = "edgetpu"
                        coral_available = True
                    elif name.startswith("ov") or "openvino" in name:
                        detector_type = detector_type or "openvino"
                    else:
                        detector_type = detector_type or name
    except Exception:
        pass

    if not detector_devices:
        if detector_type == "openvino":
            detector_devices.append("Intel iGPU/CPU (OpenVINO)" if gpu_available else "CPU (OpenVINO)")
        elif gpu_available:
            detector_devices.append(gpu_name or "Intel iGPU")
        else:
            detector_devices.append("CPU")

    return HardwareResources(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_percent=round(cpu_percent, 1),
        ram_total_gb=round(ram_total_gb, 1),
        ram_used_gb=round(ram_used_gb, 1),
        ram_percent=round(ram_pct, 1),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_percent=gpu_percent,
        gpu_inference_device=gpu_inference_device,
        coral_available=coral_available,
        coral_status=coral_status,
        detector_type=detector_type,
        detector_devices=detector_devices,
        storage_used_gb=disk["used_gb"],
        storage_total_gb=disk["total_gb"],
        storage_percent=round(storage_pct, 1),
        cameras_active=total_cameras,
        cameras_relay=cameras_relay,
        cameras_transcode=cameras_transcode,
        estimated_max_cameras_relay=max_relay,
        estimated_max_cameras_transcode=max_transcode,
        uptime_seconds=round(time.time() - _start_time, 1),
        warnings=warnings,
    )


@router.get("/coral")
async def get_coral_status():
    """Detector status — proxied from Frigate stats, shaped for the frontend CoralStatusData interface."""
    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=5) as client:
            resp = await client.get("/api/stats")
            if resp.status_code == 200:
                stats = resp.json()
                detectors = stats.get("detectors", {})
                service = stats.get("service", {})

                if not detectors:
                    return {"enabled": False, "available": False}

                # Use the first (primary) detector
                det_name, det_data = next(iter(detectors.items()))
                inference_speed = det_data.get("inference_speed", 0)
                pid = det_data.get("pid", 0)
                detection_start = det_data.get("detection_start", 0)

                uptime = (time.time() - detection_start) if detection_start else 0

                # Sum detection_fps across all cameras as a detect throughput proxy
                cameras = stats.get("cameras", {})
                total_det_fps = sum(
                    c.get("detection_fps", 0) for c in cameras.values()
                )

                return {
                    "enabled": True,
                    "available": pid > 0,
                    "active_model": det_name,
                    "swap_count": 0,
                    "last_swap_ms": 0,
                    "detect_count": int(total_det_fps),
                    "detect_avg_ms": round(inference_speed, 1),
                    "detect_last_ms": round(inference_speed, 1),
                    "cnn_count": 0,
                    "cnn_avg_ms": 0,
                    "cnn_last_ms": 0,
                    "uptime_seconds": round(uptime, 0),
                    "last_error": None,
                }
    except Exception as e:
        return {"enabled": True, "available": False, "error": str(e)}
    return {"enabled": False, "available": False}


@router.get("/settings", response_model=GlobalSettings)
async def get_settings(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "global"))
    row = result.scalar_one_or_none()

    if row and row.value:
        return GlobalSettings(**row.value)

    return GlobalSettings(
        retention_events_days=settings.retention_events_days,
        retention_continuous_days=settings.retention_continuous_days,
        retention_snapshots_days=settings.retention_snapshots_days,
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        smtp_user=settings.smtp_user,
        smtp_from=settings.smtp_from,
    )


@router.put("/settings", response_model=GlobalSettings)
async def update_settings(data: GlobalSettings, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "global"))
    row = result.scalar_one_or_none()

    if row:
        row.value = data.model_dump()
    else:
        row = SystemSettings(key="global", value=data.model_dump())

    session.add(row)
    await session.commit()
    return data


# ─── Dashboard Layout ───

@router.get("/dashboard-layout", response_model=DashboardLayout)
async def get_dashboard_layout(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "dashboard_layout"))
    row = result.scalar_one_or_none()
    if row and row.value:
        return DashboardLayout(**row.value)
    return DashboardLayout()


@router.put("/dashboard-layout", response_model=DashboardLayout)
async def update_dashboard_layout(data: DashboardLayout, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "dashboard_layout"))
    row = result.scalar_one_or_none()
    if row:
        row.value = data.model_dump()
    else:
        row = SystemSettings(key="dashboard_layout", value=data.model_dump())
    session.add(row)
    await session.commit()
    return data


# ─── Performance Settings ───

def _get_performance_defaults() -> PerformanceSettings:
    """Return baseline (balanced) performance defaults."""
    return PerformanceSettings()


async def _load_performance_settings(session: AsyncSession) -> PerformanceSettings:
    """Load performance settings from DB, falling back to defaults."""
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "performance"))
    row = result.scalar_one_or_none()
    if row and row.value:
        return PerformanceSettings(**row.value)
    return _get_performance_defaults()


def _apply_performance_settings(perf: PerformanceSettings):
    """Push performance settings into running service instances.

    Most detection/tracking settings are now managed by Frigate config.
    Only ML offload settings are still applied at runtime.
    """
    # ML offload
    import services.ml_client as ml_client
    ml_client.ml_offload_enabled = perf.ml_offload_enabled
    ml_client.ml_offload_url = perf.ml_offload_url


@router.get("/performance", response_model=PerformanceSettings)
async def get_performance_settings(session: AsyncSession = Depends(get_session)):
    return await _load_performance_settings(session)


@router.put("/performance", response_model=PerformanceSettings)
async def update_performance_settings(data: PerformanceSettings, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "performance"))
    row = result.scalar_one_or_none()

    if row:
        row.value = data.model_dump()
    else:
        row = SystemSettings(key="performance", value=data.model_dump())

    session.add(row)
    await session.commit()

    # Apply to running services immediately
    _apply_performance_settings(data)

    return data


# ─── Training Settings ───

async def load_training_settings(session: AsyncSession) -> TrainingSettings:
    """Load training settings from DB, falling back to defaults."""
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "training"))
    row = result.scalar_one_or_none()
    if row and row.value:
        return TrainingSettings(**row.value)
    return TrainingSettings()


@router.get("/training-settings", response_model=TrainingSettings)
async def get_training_settings(session: AsyncSession = Depends(get_session)):
    return await load_training_settings(session)


@router.put("/training-settings", response_model=TrainingSettings)
async def update_training_settings(data: TrainingSettings, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(SystemSettings).where(SystemSettings.key == "training"))
    row = result.scalar_one_or_none()
    if row:
        row.value = data.model_dump()
    else:
        row = SystemSettings(key="training", value=data.model_dump())
    session.add(row)
    await session.commit()
    return data


@router.post("/export-config")
async def export_config(session: AsyncSession = Depends(get_session)):
    """Export full system configuration (cameras, objects, rules, settings)."""
    cameras = (await session.execute(select(Camera))).scalars().all()
    from models.schemas import NamedObject, NotificationRule

    objects = (await session.execute(select(NamedObject))).scalars().all()
    rules = (await session.execute(select(NotificationRule))).scalars().all()
    settings_row = (await session.execute(select(SystemSettings))).scalars().all()

    config = {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "cameras": [
            {
                "name": c.name,
                "camera_type": c.camera_type.value,
                "connection_config": c.connection_config,
                "recording_mode": c.recording_mode.value,
                "detection_enabled": c.detection_enabled,
                "detection_objects": c.detection_objects,
                "detection_confidence": c.detection_confidence,
                "zones": c.zones,
            }
            for c in cameras
        ],
        "named_objects": [
            {"name": o.name, "category": o.category.value}
            for o in objects
        ],
        "settings": {s.key: s.value for s in settings_row},
    }

    return config


@router.post("/import-config")
async def import_config(file: UploadFile = File(...), session: AsyncSession = Depends(get_session)):
    """Import system configuration from JSON."""
    content = await file.read()
    try:
        config = json.loads(content)
    except json.JSONDecodeError:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    imported = {"cameras": 0, "objects": 0}

    # Import cameras
    for cam_data in config.get("cameras", []):
        from models.schemas import CameraType, RecordingMode

        camera = Camera(
            name=cam_data["name"],
            camera_type=CameraType(cam_data["camera_type"]),
            connection_config=cam_data.get("connection_config", {}),
            recording_mode=RecordingMode(cam_data.get("recording_mode", "motion")),
            detection_enabled=cam_data.get("detection_enabled", True),
            detection_objects=cam_data.get("detection_objects", ["person"]),
            detection_confidence=cam_data.get("detection_confidence", 0.5),
            zones=cam_data.get("zones"),
        )
        session.add(camera)
        imported["cameras"] += 1

    # Import named objects
    from models.schemas import NamedObject, ObjectCategory

    for obj_data in config.get("named_objects", []):
        obj = NamedObject(
            name=obj_data["name"],
            category=ObjectCategory(obj_data["category"]),
        )
        session.add(obj)
        imported["objects"] += 1

    # Import settings
    for key, value in config.get("settings", {}).items():
        result = await session.execute(select(SystemSettings).where(SystemSettings.key == key))
        existing = result.scalar_one_or_none()
        if existing:
            existing.value = value
        else:
            session.add(SystemSettings(key=key, value=value))

    await session.commit()
    return {"message": "Configuration imported", "imported": imported}


@router.get("/ml-health")
async def ml_health():
    """Check remote ML server health."""
    import services.ml_client as ml_client
    try:
        info = await ml_client.health_check()
        online = info.get("status") != "error"
        return {"enabled": ml_client.ml_offload_enabled, "online": online, **info}
    except Exception as e:
        return {"enabled": ml_client.ml_offload_enabled, "online": False, "error": str(e)}


@router.get("/recognition-agent")
async def recognition_agent_stats():
    """Return recognition bridge diagnostics."""
    return frigate_bridge.get_status()
