"""BanusNas — Deep Hunt: scan continuous recordings for a specific object.

Extracts frames from Frigate MP4 recording segments at configurable intervals,
runs YOLO detection + CNN embedding matching against a target named object,
and streams sightings back in real time.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)

RECORDINGS_ROOT = Path(settings.recordings_path)

# Also check hot storage if available
HOT_ROOT = Path(settings.hot_storage_path) / "recordings" if settings.hot_storage_path else None


@dataclass
class HuntSighting:
    """A single sighting found during a deep hunt."""
    timestamp: float          # Unix epoch
    camera_name: str
    confidence: float         # CNN cosine similarity
    det_confidence: float     # YOLO detection confidence
    class_name: str           # YOLO class (cat, dog, person, car, etc.)
    bbox: tuple[int, int, int, int]
    frame_path: str           # Path to saved thumbnail
    segment_path: str         # Source MP4 segment


@dataclass
class HuntJob:
    """Tracks the state of a running deep hunt."""
    job_id: str
    target_name: str
    target_id: int
    category: str
    camera_ids: list[int]
    start_time: datetime
    end_time: datetime
    frame_interval: float     # seconds between sampled frames
    status: str = "pending"   # pending, running, completed, cancelled, error
    progress: float = 0.0     # 0.0 to 1.0
    segments_total: int = 0
    segments_done: int = 0
    sightings: list[HuntSighting] = field(default_factory=list)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    frames_scanned: int = 0
    detections_total: int = 0       # total YOLO detections (all classes)
    detections_relevant: int = 0    # detections matching target class (cat/dog/person)
    _cancel: bool = False


# In-memory job store (ephemeral)
_jobs: dict[str, HuntJob] = {}


def get_job(job_id: str) -> Optional[HuntJob]:
    return _jobs.get(job_id)


def list_jobs(target_id: int | None = None) -> list[HuntJob]:
    """Return recent jobs, optionally filtered to a specific target."""
    jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    if target_id is not None:
        jobs = [j for j in jobs if j.target_id == target_id]
    return jobs[:20]


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job and job.status == "running":
        job._cancel = True
        return True
    return False


def _find_segments(
    camera_names: list[str],
    start: datetime,
    end: datetime,
) -> list[tuple[str, Path]]:
    """Find all MP4 recording segments in the time range for given cameras.

    Returns list of (camera_name, segment_path) sorted by time.
    """
    segments: list[tuple[str, Path, float]] = []

    # Generate date/hour directories to scan
    from datetime import timedelta
    current = start.replace(minute=0, second=0, microsecond=0)
    while current <= end:
        date_dir = current.strftime("%Y-%m-%d")
        hour_dir = current.strftime("%H")

        for cam_name in camera_names:
            # Check both hot storage and cold storage
            for root in [HOT_ROOT, RECORDINGS_ROOT]:
                if root is None:
                    continue
                seg_dir = root / date_dir / hour_dir / cam_name
                if not seg_dir.exists():
                    continue
                for mp4 in seg_dir.glob("*.mp4"):
                    # Extract timestamp from filename (MM.SS.mp4)
                    try:
                        parts = mp4.stem.split(".")
                        minute = int(parts[0])
                        second = int(parts[1]) if len(parts) > 1 else 0
                        seg_ts = current.replace(minute=minute, second=second).timestamp()
                        # Filter to time range
                        if start.timestamp() <= seg_ts <= end.timestamp() + 60:
                            segments.append((cam_name, mp4, seg_ts))
                    except (ValueError, IndexError):
                        segments.append((cam_name, mp4, current.timestamp()))

        current += timedelta(hours=1)

    # Sort by timestamp, deduplicate paths
    segments.sort(key=lambda s: s[2])
    seen = set()
    result = []
    for cam, path, ts in segments:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append((cam, path))
    return result


def _extract_frames(mp4_path: Path, interval: float) -> list[tuple[float, np.ndarray]]:
    """Extract frames from an MP4 at the given interval (seconds).

    Returns list of (timestamp_offset, frame_bgr).
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(fps * interval))

    frames = []
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        ts_offset = frame_idx / fps
        frames.append((ts_offset, frame))
        frame_idx += frame_step

    cap.release()
    return frames


async def run_hunt(
    job: HuntJob,
    target_embedding: list[float],
    target_classes: list[str],
) -> None:
    """Execute a deep hunt job — scan recordings and find sightings.

    This runs as a background task. Progress and results are stored
    on the HuntJob object for polling / SSE streaming.
    """
    from services.object_detector import object_detector
    from services.recognition_service import recognition_service, CNN_EMBED_DIM
    from services.ml_client import ml_offload_enabled

    job.status = "running"
    logger.info(
        "Deep hunt %s started: target=%s (%s), cameras=%s, range=%s→%s, interval=%.1fs",
        job.job_id, job.target_name, job.category,
        job.camera_ids, job.start_time, job.end_time, job.frame_interval,
    )

    # Prepare camera names
    camera_names = [f"camera_{cid}" for cid in job.camera_ids]

    # Find all relevant segments
    segments = _find_segments(camera_names, job.start_time, job.end_time)
    job.segments_total = len(segments)
    logger.info("Deep hunt %s: found %d recording segments to scan", job.job_id, len(segments))

    if not segments:
        job.status = "completed"
        job.progress = 1.0
        return

    # Ensure detector is available — lazy-init if needed (Frigate normally handles detection)
    if not object_detector._initialized and not ml_offload_enabled:
        logger.info("Deep hunt %s: initializing object detector on demand...", job.job_id)
        try:
            await object_detector.start()
        except Exception as e:
            logger.error("Deep hunt %s: failed to start object detector: %s", job.job_id, e)

    if not object_detector._initialized and not ml_offload_enabled:
        job.status = "error"
        job.error = "Object detector not available (local model failed to load and ML offload disabled)"
        return

    # Prepare target embedding as numpy array
    target_emb = np.array(target_embedding, dtype=np.float64)
    emb_dim = len(target_emb)
    # Loose thresholds for discovery — let users curate via UI
    threshold = 0.35 if emb_dim == CNN_EMBED_DIM else 0.65

    # Verify CNN is available
    if not recognition_service._cnn_ready:
        job.status = "error"
        job.error = "CNN feature extractor not available"
        return

    # Thumbnails directory
    thumb_dir = Path("/tmp/deep_hunt") / job.job_id
    thumb_dir.mkdir(parents=True, exist_ok=True)

    try:
        for seg_idx, (cam_name, seg_path) in enumerate(segments):
            if job._cancel:
                job.status = "cancelled"
                logger.info("Deep hunt %s cancelled at segment %d/%d", job.job_id, seg_idx, len(segments))
                return

            # Extract frames from this segment
            frames = await asyncio.to_thread(
                _extract_frames, seg_path, job.frame_interval
            )

            # Parse segment timestamp from path: /recordings/YYYY-MM-DD/HH/camera_N/MM.SS.mp4
            try:
                parts = seg_path.parts
                date_str = parts[-4]  # YYYY-MM-DD
                hour_str = parts[-3]  # HH
                min_sec = seg_path.stem  # MM.SS
                ms_parts = min_sec.split(".")
                base_dt = datetime.strptime(
                    f"{date_str} {hour_str}:{ms_parts[0]}:{ms_parts[1] if len(ms_parts) > 1 else '00'}",
                    "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                base_ts = base_dt.timestamp()
            except Exception:
                base_ts = time.time()

            for ts_offset, frame in frames:
                if job._cancel:
                    break

                frame_ts = base_ts + ts_offset
                job.frames_scanned += 1

                # Run YOLO detection
                try:
                    detections = await object_detector.detect(frame, confidence_threshold=0.35)
                except Exception as e:
                    logger.debug("Detection failed on frame: %s", e)
                    continue

                job.detections_total += len(detections)

                # Filter to target classes
                relevant = [d for d in detections if d.class_name in target_classes]
                if not relevant:
                    continue
                job.detections_relevant += len(relevant)

                # Check each detection against target embedding
                for det in relevant:
                    try:
                        crop = det.crop
                        if crop is None or crop.size == 0:
                            continue

                        # Compute CNN embedding
                        embedding = recognition_service._compute_embedding_best(crop)
                        emb_array = np.array(embedding, dtype=np.float64)

                        if len(emb_array) != emb_dim:
                            continue

                        # Cosine similarity
                        dot = float(np.dot(emb_array, target_emb))
                        norm = float(np.linalg.norm(emb_array) * np.linalg.norm(target_emb))
                        similarity = dot / (norm + 1e-8)

                        if similarity >= threshold:
                            # Save thumbnail — use clean crop from original frame
                            # (det.crop may have seg-mask black-out for embeddings)
                            x1t, y1t, x2t, y2t = det.bbox
                            clean_crop = frame[y1t:y2t, x1t:x2t]
                            thumb_name = f"{cam_name}_{int(frame_ts)}_{len(job.sightings)}.jpg"
                            thumb_path = thumb_dir / thumb_name
                            await asyncio.to_thread(
                                cv2.imwrite, str(thumb_path), clean_crop
                            )

                            sighting = HuntSighting(
                                timestamp=frame_ts,
                                camera_name=cam_name,
                                confidence=similarity,
                                det_confidence=det.confidence,
                                class_name=det.class_name,
                                bbox=det.bbox,
                                frame_path=str(thumb_path),
                                segment_path=str(seg_path),
                            )
                            job.sightings.append(sighting)
                            logger.info(
                                "Deep hunt %s: sighting #%d — %s on %s @ %.3f (%.2f cos)",
                                job.job_id, len(job.sightings),
                                det.class_name, cam_name, frame_ts, similarity,
                            )
                    except Exception as e:
                        logger.debug("Embedding comparison failed: %s", e)
                        continue

            job.segments_done = seg_idx + 1
            job.progress = job.segments_done / max(job.segments_total, 1)

            # Yield to event loop periodically
            await asyncio.sleep(0)

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        logger.error("Deep hunt %s failed: %s", job.job_id, e, exc_info=True)
        return

    job.status = "completed"
    job.progress = 1.0
    logger.info(
        "Deep hunt %s completed: %d segments, %d frames, %d detections (%d relevant), %d sightings",
        job.job_id, job.segments_total, job.frames_scanned,
        job.detections_total, job.detections_relevant, len(job.sightings),
    )
