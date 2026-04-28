"""BanusNas — Frigate Config Generator.

Auto-generates Frigate config.yml from camera database entries.
Called after camera create/update/delete to keep Frigate in sync.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx
import yaml

from core.config import settings

logger = logging.getLogger(__name__)


def detect_coral_devices() -> dict:
    """Probe the host for Coral Edge TPU devices.

    Returns dict: {
        "usb": bool,         # USB Coral on /sys/bus/usb (Google vendor 1a6e/18d1)
        "pcie": list[int],   # PCIe Coral indices (M.2 / Mini PCIe), via /sys/bus/pci
    }
    Detection works from inside the API container without any /dev passthrough
    — we only read /sys, which is mounted automatically by Docker. Frigate
    runs `privileged: true` so it has direct access to the actual /dev nodes.
    """
    result: dict = {"usb": False, "pcie": []}

    # PCIe / M.2 Coral: Google Edge TPU PCI vendor 1ac1, device 089a.
    # Each board exposes one or more functions — we count unique BDFs.
    try:
        seen = set()
        for dev_dir in Path("/sys/bus/pci/devices").iterdir():
            vendor_file = dev_dir / "vendor"
            device_file = dev_dir / "device"
            if not vendor_file.exists() or not device_file.exists():
                continue
            try:
                vendor = vendor_file.read_text().strip().lower()
                device = device_file.read_text().strip().lower()
            except (OSError, PermissionError):
                continue
            if vendor == "0x1ac1" and device == "0x089a":
                seen.add(dev_dir.name)
        # Sort by BDF for stable indices (apex_0, apex_1, ...)
        for idx, _ in enumerate(sorted(seen)):
            result["pcie"].append(idx)
    except (FileNotFoundError, PermissionError):
        pass

    # USB Coral: 1a6e:089a (unprogrammed) or 18d1:9302 (Google Edge TPU)
    try:
        for dev_dir in Path("/sys/bus/usb/devices").iterdir():
            vendor_file = dev_dir / "idVendor"
            product_file = dev_dir / "idProduct"
            if not vendor_file.exists():
                continue
            try:
                vendor = vendor_file.read_text().strip().lower()
                product = product_file.read_text().strip().lower() if product_file.exists() else ""
            except (OSError, PermissionError):
                continue
            if (vendor == "1a6e" and product == "089a") or \
               (vendor == "18d1" and product == "9302"):
                result["usb"] = True
                break
    except (FileNotFoundError, PermissionError):
        pass

    return result

# Path where Frigate config is mounted (shared between API and Frigate containers)
FRIGATE_CONFIG_DIR = Path("/config/frigate")
FRIGATE_CONFIG_PATH = FRIGATE_CONFIG_DIR / "config.yml"

# Sub-stream path derivation per camera type
SUB_STREAM_MAP = {
    "tapo": {
        "stream1": "stream2", "stream2": "stream2",
        "stream3": "stream4", "stream4": "stream4",
        "stream5": "stream6", "stream6": "stream6",
        "stream7": "stream8", "stream8": "stream8",
    },
    "hikvision": {
        "Streaming/Channels/101": "Streaming/Channels/102",
        "Streaming/Channels/201": "Streaming/Channels/202",
    },
}

# Default main/sub stream paths per camera type
DEFAULT_STREAMS = {
    "tapo": ("stream1", "stream2"),
    "hikvision": ("Streaming/Channels/101", "Streaming/Channels/102"),
    "onvif": ("stream1", "stream2"),
}


def _build_rtsp_url(camera_type: str, conn: dict, stream_path: str) -> str:
    """Build an RTSP URL for a given camera type and stream path."""
    if camera_type == "ring":
        ring_user = conn.get("ring_rtsp_user", settings.ring_rtsp_user)
        ring_pass = conn.get("ring_rtsp_password", settings.ring_rtsp_password)
        device_id = conn.get("ring_device_id", conn.get("ring_device_name", ""))
        return f"rtsp://{ring_user}:{ring_pass}@ring-mqtt:8554/{device_id}_live#timeout=30"

    if camera_type == "rtsp":
        return conn.get("rtsp_url", conn.get("url", ""))

    if camera_type == "other":
        return conn.get("stream_url", conn.get("url", ""))

    host = conn.get("ip", conn.get("host", ""))
    port = conn.get("port", "554")
    username = quote(conn.get("username", "admin"), safe="")
    password = quote(conn.get("password", ""), safe="")

    return f"rtsp://{username}:{password}@{host}:{port}/{stream_path}"


def _get_stream_paths(camera_type: str, conn: dict) -> tuple[str, Optional[str]]:
    """Get main and sub stream paths for a camera."""
    main_path = conn.get("stream_path", "")
    sub_path = conn.get("sub_stream_path", "")

    # Use defaults if not specified
    if not main_path:
        defaults = DEFAULT_STREAMS.get(camera_type)
        if defaults:
            main_path = defaults[0]
            if not sub_path:
                sub_path = defaults[1]

    # Derive sub from main if not set
    if not sub_path and camera_type in SUB_STREAM_MAP:
        sub_path = SUB_STREAM_MAP[camera_type].get(main_path, "")

    return main_path, sub_path or None


def _build_detector_block() -> dict:
    """Pick the best Frigate detector based on detected hardware.

    Returns a partial config dict with `detectors` and `model` keys.
    Coral wins if any device is present (TPU is dramatically faster than iGPU
    for SSD MobileNet). Otherwise we fall back to OpenVINO AUTO, which uses
    Intel iGPU when available and CPU as a last resort.
    """
    coral = detect_coral_devices()
    detectors: dict = {}

    # PCIe / M.2 Coral devices first (one detector per device for parallelism)
    for idx in coral["pcie"]:
        device = "pci" if idx == 0 else f"pci:{idx}"
        detectors[f"coral_pci{idx}"] = {"type": "edgetpu", "device": device}

    # USB Coral
    if coral["usb"]:
        detectors["coral_usb"] = {"type": "edgetpu", "device": "usb"}

    if detectors:
        logger.info(
            "Frigate detector: Coral Edge TPU (usb=%s, pcie=%s)",
            coral["usb"], coral["pcie"],
        )
        # SSD MobileNet V2 EdgeTPU model is bundled in the Frigate image at
        # /edgetpu_model.tflite — no external download required.
        return {
            "detectors": detectors,
            "model": {
                "width": 320,
                "height": 320,
                "input_tensor": "nhwc",
                "input_pixel_format": "rgb",
                "model_type": "ssd",
                "labelmap_path": "/labelmap/coco-80.txt",
            },
        }

    logger.info("Frigate detector: OpenVINO AUTO (no Coral detected)")
    return {
        "detectors": {
            "openvino": {"type": "openvino", "device": "AUTO"},
        },
        "model": {
            "width": 300,
            "height": 300,
            "input_tensor": "nhwc",
            "input_pixel_format": "bgr",
            "model_type": "ssd",
            "labelmap_path": "/labelmap/coco-80.txt",
        },
    }


def generate_frigate_config(cameras: list) -> dict:
    """Generate complete Frigate config dict from camera list.

    cameras: list of Camera ORM objects (or dicts with same fields)
    """
    # Static global sections
    config = {
        "mqtt": {
            "host": "mqtt",
            "port": 1883,
            "topic_prefix": "frigate",
            "stats_interval": 60,
        },
        "database": {"path": "/config/frigate.db"},
        # Detector + model are picked at runtime based on what hardware is
        # actually present — Coral TPU (USB or M.2) wins, otherwise OpenVINO
        # (Intel iGPU/CPU). The bundled SSDLite MobileNet v2 model is used in
        # both cases so no external downloads are required.
        **_build_detector_block(),
        "ffmpeg": {
            "hwaccel_args": "preset-vaapi",
            "output_args": {"record": "preset-record-generic-audio-aac"},
        },
        "objects": {
            "track": ["person", "cat", "dog", "car", "bird"],
            "filters": {
                "person": {"min_area": 1500, "min_score": 0.55, "threshold": 0.72},
                "cat": {"min_area": 500, "min_score": 0.3, "threshold": 0.5},
                "dog": {"min_area": 500, "min_score": 0.2, "threshold": 0.35},
                "car": {"min_area": 2000, "min_score": 0.55, "threshold": 0.72},
                "bird": {"min_area": 400, "min_score": 0.35, "threshold": 0.5},
            },
        },
        "record": {
            "enabled": True,
            "alerts": {"retain": {"days": 30, "mode": "motion"}},
            "detections": {"retain": {"days": 14, "mode": "motion"}},
            "continuous": {"days": 0},
            "motion": {"days": 3},
        },
        "snapshots": {
            "enabled": True,
            "clean_copy": True,
            "timestamp": False,
            "bounding_box": True,
            "retain": {"default": 30},
        },
        "motion": {
            "threshold": 30,
            "contour_area": 20,
            "improve_contrast": True,
        },
        "ui": {
            "time_format": "24hour",
            "timezone": "Europe/London",
        },
        "version": "0.17-0",
    }

    # Generate go2rtc streams and camera sections
    go2rtc_streams = {}
    camera_sections = {}

    for cam in cameras:
        if not cam.enabled:
            continue

        cam_id = cam.id
        cam_name = f"camera_{cam_id}"
        cam_type = cam.camera_type if isinstance(cam.camera_type, str) else cam.camera_type.value
        conn = cam.connection_config or {}

        main_path, sub_path = _get_stream_paths(cam_type, conn)

        # Build go2rtc stream entries
        main_url = _build_rtsp_url(cam_type, conn, main_path)
        if not main_url:
            logger.warning("Skipping camera %s: no stream URL", cam_name)
            continue

        go2rtc_streams[cam_name] = [main_url]

        if sub_path and cam_type not in ("ring", "rtsp", "other"):
            sub_url = _build_rtsp_url(cam_type, conn, sub_path)
            if sub_url and sub_url != main_url:
                go2rtc_streams[f"{cam_name}_sub"] = [sub_url]

        # Ring cameras are on-demand — Frigate must NOT continuously pull
        # frames or it creates a crash loop with ring-mqtt.  Keep go2rtc
        # streams so the frontend can do on-demand live viewing.
        if cam_type == "ring":
            continue

        # Build Frigate camera section
        has_sub = f"{cam_name}_sub" in go2rtc_streams
        cam_section = _build_camera_section(cam, cam_name, has_sub, cam_type)
        camera_sections[cam_name] = cam_section

    config["go2rtc"] = {
        "streams": go2rtc_streams,
        "webrtc": {"candidates": ["192.168.68.59:8555"]},
    }
    config["cameras"] = camera_sections

    return config


def _build_camera_section(cam, cam_name: str, has_sub: bool, cam_type: str) -> dict:
    """Build a single Frigate camera config section."""
    section: dict = {}

    # FFmpeg inputs
    if has_sub:
        section["ffmpeg"] = {
            "inputs": [
                {"path": f"rtsp://127.0.0.1:8554/{cam_name}", "roles": ["record"]},
                {"path": f"rtsp://127.0.0.1:8554/{cam_name}_sub", "roles": ["detect"]},
            ]
        }
    else:
        section["ffmpeg"] = {
            "inputs": [
                {"path": f"rtsp://127.0.0.1:8554/{cam_name}", "roles": ["detect", "record"]},
            ]
        }

    # Ring cameras need retry interval
    if cam_type == "ring":
        section["ffmpeg"]["retry_interval"] = 30

    # ONVIF / PTZ config
    ptz = cam.ptz_config or {}
    if cam.ptz_mode and ptz.get("enabled"):
        protocol = ptz.get("protocol", "onvif")
        if protocol in ("onvif", "tapo"):
            onvif_section = {
                "host": ptz.get("onvif_host", ""),
                "port": ptz.get("onvif_port", 2020 if cam_type == "tapo" else 80),
                "user": ptz.get("onvif_user", ""),
                "password": ptz.get("onvif_password", ""),
            }
            if protocol == "onvif" and ptz.get("autotrack_enabled"):
                autotrack_objs = ptz.get("autotrack_objects", ["person"])
                if "cat" in autotrack_objs and "dog" not in autotrack_objs:
                    autotrack_objs = list(autotrack_objs) + ["dog"]
                onvif_section["autotracking"] = {
                    "enabled": True,
                    "track": autotrack_objs,
                    "timeout": ptz.get("autotrack_timeout", 30),
                    "required_zones": ["full_zone"],
                }
            section["onvif"] = onvif_section

    # Detect config
    det_settings = cam.detection_settings or {}
    objects_track = cam.detection_objects or ["person"]
    has_animals = "cat" in objects_track or "dog" in objects_track
    section["detect"] = {
        "enabled": cam.detection_enabled,
        "width": det_settings.get("width", 640),
        "height": det_settings.get("height", 360),
        "fps": det_settings.get("fps", 5),
        # Keep stationary pets tracked much longer — cats/dogs sit still for ages
        "max_disappeared": 75 if has_animals else 25,
        "stationary": {
            "threshold": 50,
            # max_frames 0 = never remove stationary tracked objects
            "max_frames": {
                "default": 3000,  # ~10 min at 5fps for people
                "objects": (
                    {"cat": 0, "dog": 0} if has_animals else {}
                ),
            },
        },
    }

    # ── User-drawn ignore zones (DetectionSettings UI) → Frigate masks ──
    # Each zone: {polygon: [[x,y], ...], classes: [...], image_width, image_height}
    # If classes is empty → applies camera-wide (motion + objects).
    # If classes specified → applies only as per-object-class filter mask.
    ignore_zones = det_settings.get("ignore_zones") or []
    universal_polys: list[str] = []   # camera-wide masks
    per_class_polys: dict[str, list[str]] = {}  # class → list[polygon str]
    for z in ignore_zones:
        poly = z.get("polygon") or []
        if len(poly) < 3:
            continue
        iw = z.get("image_width") or 0
        ih = z.get("image_height") or 0
        if iw <= 0 or ih <= 0:
            # Older zones without saved dimensions — skip Frigate sync to avoid
            # mis-scaled masks (still consumed by app-side object_tracker).
            continue
        # Normalize to 0..1 and clamp; Frigate format: "x1,y1,x2,y2,..."
        coords = []
        for px, py in poly:
            nx = max(0.0, min(1.0, float(px) / iw))
            ny = max(0.0, min(1.0, float(py) / ih))
            coords.append(f"{nx:.4f}")
            coords.append(f"{ny:.4f}")
        poly_str = ",".join(coords)
        cls_list = z.get("classes") or []
        if not cls_list:
            universal_polys.append(poly_str)
        else:
            for c in cls_list:
                per_class_polys.setdefault(c, []).append(poly_str)

    # Motion config
    motion_threshold = det_settings.get("motion_threshold")
    motion_contour = det_settings.get("motion_contour_area")
    motion_mask = det_settings.get("motion_mask")  # str or list of polygon coord strings
    # Combine raw motion_mask with universal ignore zones
    combined_motion_mask: list[str] = []
    if motion_mask:
        if isinstance(motion_mask, list):
            combined_motion_mask.extend(motion_mask)
        else:
            combined_motion_mask.append(motion_mask)
    combined_motion_mask.extend(universal_polys)
    if motion_threshold or motion_contour or combined_motion_mask:
        motion = {}
        if motion_threshold:
            motion["threshold"] = motion_threshold
        if motion_contour:
            motion["contour_area"] = motion_contour
        if combined_motion_mask:
            motion["mask"] = combined_motion_mask
        section["motion"] = motion

    # Objects config — always pair cat+dog since white cats are often classified as dog
    # (defined early because detect config also needs to know about animal tracking)
    if "cat" in objects_track and "dog" not in objects_track:
        objects_track = list(objects_track) + ["dog"]
    section["objects"] = {"track": objects_track}

    # Camera-wide object mask (raw object_mask + universal ignore zones)
    object_mask = det_settings.get("object_mask")
    combined_object_mask: list[str] = []
    if object_mask:
        if isinstance(object_mask, list):
            combined_object_mask.extend(object_mask)
        else:
            combined_object_mask.append(object_mask)
    combined_object_mask.extend(universal_polys)
    if combined_object_mask:
        section["objects"]["mask"] = combined_object_mask

    # Per-object filters from detection_settings (merge with per-class ignore zones)
    obj_filters = dict(det_settings.get("object_filters", {}) or {})
    # Inject per-class ignore-zone masks
    for cls, polys in per_class_polys.items():
        existing = dict(obj_filters.get(cls, {}) or {})
        existing_mask = existing.get("mask")
        if existing_mask:
            if isinstance(existing_mask, list):
                existing["mask"] = existing_mask + polys
            else:
                existing["mask"] = [existing_mask, *polys]
        else:
            existing["mask"] = polys
        obj_filters[cls] = existing
    if obj_filters:
        section["objects"]["filters"] = {}
        for obj_name, obj_f in obj_filters.items():
            filt = {}
            if "min_area" in obj_f:
                filt["min_area"] = obj_f["min_area"]
            if "min_score" in obj_f:
                filt["min_score"] = obj_f["min_score"]
            if "threshold" in obj_f:
                filt["threshold"] = obj_f["threshold"]
            if "mask" in obj_f:
                # Per-object-class mask
                m = obj_f["mask"]
                filt["mask"] = m if isinstance(m, list) else [m]
            if filt:
                section["objects"]["filters"][obj_name] = filt

    # Recording config based on recording_mode
    rec_mode = cam.recording_mode if isinstance(cam.recording_mode, str) else cam.recording_mode.value
    if rec_mode == "disabled":
        section["record"] = {"enabled": False}
    elif rec_mode == "events":
        section["record"] = {"continuous": {"days": 0}, "motion": {"days": 3}}
    # continuous and motion use global defaults

    # Zones — always include a full-frame zone so autotracking required_zones is satisfied
    det_w = section["detect"].get("width", 640)
    det_h = section["detect"].get("height", 360)
    zone_entries = {
        "full_zone": {
            "coordinates": "0,0,1,0,1,1,0,1",
        }
    }
    # Merge user-defined zones on top
    zones = cam.zones
    if zones and isinstance(zones, dict):
        for zone_name, zone_data in zones.items():
            if isinstance(zone_data, dict) and "coordinates" in zone_data:
                zone_entries[zone_name] = {"coordinates": zone_data["coordinates"]}
    section["zones"] = zone_entries

    return section


def config_to_yaml(config: dict) -> str:
    """Convert config dict to YAML string with comments."""
    header = (
        "# =============================================================================\n"
        "# Frigate NVR — Auto-generated by BanusNVR\n"
        "# DO NOT EDIT MANUALLY — changes will be overwritten on camera save\n"
        "# =============================================================================\n\n"
    )

    # Custom YAML representer to output lists inline for simple values
    class CustomDumper(yaml.SafeDumper):
        pass

    def str_representer(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    CustomDumper.add_representer(str, str_representer)

    return header + yaml.dump(
        config,
        Dumper=CustomDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


async def deploy_frigate_config(cameras: list) -> dict:
    """Generate, write, and reload Frigate config.

    Returns: {"success": bool, "message": str}
    """
    try:
        config = generate_frigate_config(cameras)
        yaml_str = config_to_yaml(config)

        # Write config
        if not FRIGATE_CONFIG_DIR.exists():
            logger.error("Frigate config directory not found: %s", FRIGATE_CONFIG_DIR)
            return {"success": False, "message": "Frigate config directory not mounted"}

        FRIGATE_CONFIG_PATH.write_text(yaml_str)
        logger.info("Frigate config written to %s (%d cameras)", FRIGATE_CONFIG_PATH, len(config.get("cameras", {})))

        # Clean up files that cause issues on restart
        for stale_file in ("go2rtc_homekit.yml", "backup_config.yaml"):
            stale_path = FRIGATE_CONFIG_DIR / stale_file
            if stale_path.exists():
                stale_path.unlink()
                logger.info("Removed stale file: %s", stale_file)

        # Restart Frigate via its API
        try:
            async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=30) as client:
                resp = await client.post("/api/restart")
                if resp.status_code == 200:
                    logger.info("Frigate restart requested successfully")
                    return {"success": True, "message": "Config deployed and Frigate restarting"}
                else:
                    logger.warning("Frigate restart returned %d: %s", resp.status_code, resp.text[:200])
                    return {"success": True, "message": f"Config written but Frigate restart returned {resp.status_code}. Manual restart may be needed."}
        except Exception as e:
            logger.warning("Failed to restart Frigate: %s", e)
            return {"success": True, "message": f"Config written but could not restart Frigate: {e}. Manual restart needed."}

    except Exception as e:
        logger.error("Failed to deploy Frigate config: %s", e, exc_info=True)
        return {"success": False, "message": str(e)}
