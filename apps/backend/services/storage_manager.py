"""BanusNas — Storage Manager: retention, cleanup, disk usage, and clip export."""

import asyncio
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages recording storage, retention policies, and clip export."""

    def __init__(self):
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start periodic cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("StorageManager started — cleanup interval: 1 hour")

    async def stop(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """Run cleanup every hour."""
        while True:
            try:
                await self.run_cleanup()
            except Exception as e:
                logger.error("Cleanup error: %s", e)
            await asyncio.sleep(3600)

    async def run_cleanup(self):
        """Delete old data and migrate hot→cold storage."""
        if settings.hot_storage_path:
            await asyncio.to_thread(self._migrate_hot_to_cold)
        await asyncio.to_thread(self._cleanup_recordings)
        await asyncio.to_thread(self._cleanup_snapshots)
        logger.info("Cleanup completed")

    def _migrate_hot_to_cold(self):
        """Move old recordings and snapshots from SSD hot storage to HDD cold storage."""
        hot_root = Path(settings.hot_storage_path)
        if not hot_root.exists():
            return

        now = datetime.now(timezone.utc)

        # ── Migrate recordings ──
        hot_recordings = hot_root / "recordings"
        if hot_recordings.exists():
            rec_cutoff = now - timedelta(hours=settings.hot_storage_recordings_hours)
            rec_cutoff_date = rec_cutoff.strftime("%Y-%m-%d")
            rec_cutoff_hour = rec_cutoff.strftime("%H")
            moved_rec = 0

            for camera_dir in hot_recordings.iterdir():
                if not camera_dir.is_dir():
                    continue
                cam_id = camera_dir.name
                for date_dir in sorted(camera_dir.iterdir()):
                    if not date_dir.is_dir():
                        continue
                    for hour_dir in sorted(date_dir.iterdir()):
                        if not hour_dir.is_dir():
                            continue
                        # Migrate if older than cutoff
                        if date_dir.name < rec_cutoff_date or (date_dir.name == rec_cutoff_date and hour_dir.name < rec_cutoff_hour):
                            cold_dest = Path(settings.recordings_path) / cam_id / date_dir.name / hour_dir.name
                            cold_dest.mkdir(parents=True, exist_ok=True)
                            for f in hour_dir.iterdir():
                                dest_file = cold_dest / f.name
                                if not dest_file.exists():
                                    shutil.move(str(f), str(dest_file))
                                    moved_rec += 1
                                else:
                                    f.unlink(missing_ok=True)
                            # Remove empty hour dir
                            try:
                                hour_dir.rmdir()
                            except OSError:
                                pass
                    # Remove empty date dir
                    try:
                        date_dir.rmdir()
                    except OSError:
                        pass

            if moved_rec:
                logger.info("Hot→Cold: migrated %d recording files", moved_rec)

        # ── Migrate snapshots ──
        hot_snapshots = hot_root / "snapshots"
        if hot_snapshots.exists():
            snap_cutoff = now - timedelta(hours=settings.hot_storage_snapshots_hours)
            snap_cutoff_ts = snap_cutoff.timestamp()
            moved_snap = 0

            for camera_dir in hot_snapshots.iterdir():
                if not camera_dir.is_dir():
                    continue
                cam_id = camera_dir.name
                for snap_file in list(camera_dir.rglob("*")):
                    if not snap_file.is_file():
                        continue
                    if snap_file.stat().st_mtime < snap_cutoff_ts:
                        # Preserve relative structure under snapshots/{cam}/
                        rel = snap_file.relative_to(hot_snapshots / cam_id)
                        cold_dest = Path(settings.snapshots_path) / cam_id / rel
                        cold_dest.parent.mkdir(parents=True, exist_ok=True)
                        if not cold_dest.exists():
                            shutil.move(str(snap_file), str(cold_dest))
                            moved_snap += 1
                        else:
                            snap_file.unlink(missing_ok=True)

                # Clean empty subdirs
                for sub in sorted(camera_dir.rglob("*"), reverse=True):
                    if sub.is_dir():
                        try:
                            sub.rmdir()
                        except OSError:
                            pass

            if moved_snap:
                logger.info("Hot→Cold: migrated %d snapshot files", moved_snap)

    def _cleanup_recordings(self):
        """Remove recording date-directories older than retention.

        Frigate stores recordings as: recordings_path/YYYY-MM-DD/HH/camera_name/MM.SS.mp4
        """
        recordings_path = Path(settings.recordings_path)
        if not recordings_path.exists():
            return

        cutoff_continuous = datetime.now(timezone.utc) - timedelta(days=settings.retention_continuous_days)
        cutoff_date_str = cutoff_continuous.strftime("%Y-%m-%d")

        import re
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        for entry in recordings_path.iterdir():
            if not entry.is_dir():
                continue
            # Only process directories matching YYYY-MM-DD (Frigate date dirs)
            if not date_pattern.match(entry.name):
                continue
            if entry.name < cutoff_date_str:
                logger.info("Removing old recordings: %s", entry)
                shutil.rmtree(entry, ignore_errors=True)

    def _cleanup_snapshots(self):
        """Remove snapshots older than retention."""
        snapshots_path = Path(settings.snapshots_path)
        if not snapshots_path.exists():
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.retention_snapshots_days)
        cutoff_ts = cutoff.timestamp()

        for camera_dir in snapshots_path.iterdir():
            if not camera_dir.is_dir():
                continue
            for snap_file in camera_dir.rglob("*.jpg"):
                if snap_file.stat().st_mtime < cutoff_ts:
                    snap_file.unlink(missing_ok=True)

    def get_disk_usage(self) -> dict:
        """Get disk usage statistics for recordings (both hot and cold)."""
        recordings_path = Path(settings.recordings_path)
        try:
            usage = shutil.disk_usage(str(recordings_path))
            per_camera = {}

            if recordings_path.exists():
                for camera_dir in recordings_path.iterdir():
                    if camera_dir.is_dir() and camera_dir.name != "snapshots":
                        total_size = sum(f.stat().st_size for f in camera_dir.rglob("*") if f.is_file())
                        per_camera[camera_dir.name] = round(total_size / (1024 ** 3), 2)

            result = {
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "used_gb": round(usage.used / (1024 ** 3), 2),
                "free_gb": round(usage.free / (1024 ** 3), 2),
                "usage_percent": round(usage.used / usage.total * 100, 1),
                "per_camera_gb": per_camera,
            }

            # Add hot storage stats if configured
            if settings.hot_storage_path:
                hot_root = Path(settings.hot_storage_path)
                try:
                    hot_usage = shutil.disk_usage(str(hot_root))
                    result["hot_storage"] = {
                        "total_gb": round(hot_usage.total / (1024 ** 3), 2),
                        "used_gb": round(hot_usage.used / (1024 ** 3), 2),
                        "free_gb": round(hot_usage.free / (1024 ** 3), 2),
                        "usage_percent": round(hot_usage.used / hot_usage.total * 100, 1),
                    }
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.error("Failed to get disk usage: %s", e)
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0, "per_camera_gb": {}}

    def get_recording_timeline(self, camera_id: int, date: str) -> list[dict]:
        """Get which hours have recordings for a camera on a given date."""
        # Merge results from both hot and cold storage
        seen_hours: dict[int, int] = {}  # hour -> segment_count

        search_dirs = []
        if settings.hot_storage_path:
            search_dirs.append(Path(settings.hot_storage_path) / "recordings" / str(camera_id) / date)
        search_dirs.append(Path(settings.recordings_path) / str(camera_id) / date)

        for camera_dir in search_dirs:
            if not camera_dir.exists():
                continue
            for hour_dir in camera_dir.iterdir():
                if not hour_dir.is_dir():
                    continue
                h = int(hour_dir.name)
                count = len(list(hour_dir.glob("segment-*.ts")))
                seen_hours[h] = seen_hours.get(h, 0) + count

        return [
            {"hour": h, "segments": seen_hours[h], "has_recordings": seen_hours[h] > 0}
            for h in sorted(seen_hours)
        ]

    async def export_clip(self, camera_id: int, start: datetime, end: datetime) -> Optional[str]:
        """Export a time range as an MP4 file using FFmpeg concat."""
        segments = self._collect_segments(camera_id, start, end)
        if not segments:
            return None

        export_dir = Path(settings.recordings_path) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = export_dir / f"clip_{camera_id}_{ts}.mp4"

        # Create concat file list
        concat_file = export_dir / f"concat_{camera_id}_{ts}.txt"
        with open(concat_file, "w") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]

        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, timeout=300,
            )
            concat_file.unlink(missing_ok=True)

            if proc.returncode == 0:
                logger.info("Clip exported: %s", output_path)
                return str(output_path)
            else:
                logger.error("Clip export failed: %s", proc.stderr.decode(errors="replace"))
                return None
        except Exception as e:
            logger.error("Clip export error: %s", e)
            concat_file.unlink(missing_ok=True)
            return None

    def _collect_segments(self, camera_id: int, start: datetime, end: datetime) -> list[str]:
        """Collect all .ts segment files within a time range (hot + cold)."""
        segments = []
        current = start.replace(minute=0, second=0, microsecond=0)

        while current <= end:
            found = False
            # Check hot storage first
            if settings.hot_storage_path:
                hour_dir = Path(settings.hot_storage_path) / "recordings" / str(camera_id) / current.strftime("%Y-%m-%d") / current.strftime("%H")
                if hour_dir.exists():
                    for ts_file in sorted(hour_dir.glob("segment-*.ts")):
                        segments.append(str(ts_file))
                    found = True
            # Then cold storage
            hour_dir = Path(settings.recordings_path) / str(camera_id) / current.strftime("%Y-%m-%d") / current.strftime("%H")
            if hour_dir.exists():
                for ts_file in sorted(hour_dir.glob("segment-*.ts")):
                    if not found or str(ts_file) not in segments:
                        segments.append(str(ts_file))
            current += timedelta(hours=1)

        return segments

    def serve_segment(self, camera_id: int, segment_path: str) -> Optional[str]:
        """Resolve a segment relative path to an absolute path, with validation.

        Searches hot storage (SSD) first, then cold storage (HDD).
        """
        search_roots = []
        if settings.hot_storage_path:
            search_roots.append(Path(settings.hot_storage_path) / "recordings" / str(camera_id))
        search_roots.append(Path(settings.recordings_path) / str(camera_id))

        for base in search_roots:
            full_path = (base / segment_path).resolve()
            # Path traversal protection
            if not str(full_path).startswith(str(base.resolve())):
                logger.warning("Path traversal attempt: %s", segment_path)
                return None
            if full_path.exists() and full_path.suffix == ".ts":
                return str(full_path)

        return None


# Singleton
storage_manager = StorageManager()
