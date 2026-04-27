"""BanusNas — Recording Engine: FFmpeg-based HLS recording pipeline."""

import asyncio
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.config import settings
from services.stream_manager import stream_manager

logger = logging.getLogger(__name__)


class CameraRecorder:
    """Manages an FFmpeg process that records a single camera to HLS segments."""

    def __init__(self, camera_id: int, camera_name: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def is_recording(self) -> bool:
        return self._running and self._process is not None and self._process.poll() is None

    def _get_segment_dir(self) -> Path:
        now = datetime.now(timezone.utc)
        # Write to SSD hot storage if available, otherwise HDD
        root = settings.hot_storage_path if settings.hot_storage_path else settings.recordings_path
        base = Path(root) / "recordings" / str(self.camera_id) / now.strftime("%Y-%m-%d") / now.strftime("%H") if settings.hot_storage_path else Path(settings.recordings_path) / str(self.camera_id) / now.strftime("%Y-%m-%d") / now.strftime("%H")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _build_ffmpeg_cmd(self, rtsp_url: str, output_dir: Path) -> list[str]:
        playlist = str(output_dir / "playlist.m3u8")
        segment_pattern = str(output_dir / "segment-%05d.ts")
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-fflags", "+genpts+discardcorrupt",
            "-rtsp_transport", "tcp",
            "-timeout", "5000000",  # 5s RTSP timeout (microseconds)
            "-use_wallclock_as_timestamps", "1",
            "-i", rtsp_url,
            "-c:v", "copy",
            "-c:a", "aac",
            "-ac", "1",            # Mono audio saves ~50% audio bitrate
            "-b:a", "48k",
            "-f", "hls",
            "-hls_time", "4",
            "-hls_list_size", "0",
            "-hls_flags", "append_list+temp_file",  # temp_file prevents partial reads
            "-hls_segment_filename", segment_pattern,
            "-strftime_mkdir", "1",
            playlist,
        ]

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._recording_loop())
        logger.info("Recording started for camera %s (%s)", self.camera_id, self.camera_name)

    async def _recording_loop(self):
        """Run FFmpeg in a loop, restarting on crash with backoff."""
        backoff = 5
        while self._running:
            rtsp_url = stream_manager.get_rtsp_url(self.camera_name)
            output_dir = self._get_segment_dir()
            cmd = self._build_ffmpeg_cmd(rtsp_url, output_dir)

            try:
                self._process = await asyncio.to_thread(
                    subprocess.Popen,
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                logger.info("FFmpeg PID %d recording camera %s to %s", self._process.pid, self.camera_name, output_dir)

                # Wait for process to finish (or crash)
                while self._running and self._process.poll() is None:
                    await asyncio.sleep(10)
                    # Rotate output dir every hour
                    new_dir = self._get_segment_dir()
                    if new_dir != output_dir:
                        logger.info("Hour rolled over — restarting FFmpeg for camera %s", self.camera_name)
                        self._stop_process()
                        await asyncio.sleep(0.5)  # Brief pause to let the process fully exit
                        break

                if not self._running:
                    self._stop_process()  # Reap on exit
                    break

                returncode = self._process.poll()
                if returncode is not None and returncode != 0:
                    stderr_out = ""
                    if self._process.stderr:
                        stderr_out = self._process.stderr.read().decode(errors="replace")[-500:]
                        self._process.stderr.close()
                    self._process = None  # Mark as reaped
                    logger.warning(
                        "FFmpeg exited with code %d for camera %s: %s",
                        returncode, self.camera_name, stderr_out,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                else:
                    backoff = 5  # Reset on clean exit (hour rotation)

            except Exception as e:
                logger.error("Recording error for camera %s: %s", self.camera_name, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _stop_process(self):
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()  # SIGTERM — cleaner than SIGINT
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)  # Reap the zombie
            except OSError:
                pass
            finally:
                # Ensure stderr is closed to release file descriptors
                if self._process and self._process.stderr:
                    self._process.stderr.close()
                self._process = None
        elif self._process:
            # Process already exited — reap it to prevent zombies
            try:
                self._process.wait(timeout=1)
                if self._process.stderr:
                    self._process.stderr.close()
            except Exception:
                pass
            self._process = None

    async def stop(self):
        self._running = False
        self._stop_process()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Recording stopped for camera %s", self.camera_name)


class RecordingEngine:
    """Manages recording processes for all cameras."""

    def __init__(self):
        self._recorders: dict[int, CameraRecorder] = {}

    async def start_recording(self, camera_id: int, camera_name: str):
        if camera_id in self._recorders and self._recorders[camera_id].is_recording:
            return
        recorder = CameraRecorder(camera_id, camera_name)
        self._recorders[camera_id] = recorder
        await recorder.start()

    async def stop_recording(self, camera_id: int):
        recorder = self._recorders.pop(camera_id, None)
        if recorder:
            await recorder.stop()

    async def stop_all(self):
        for camera_id in list(self._recorders.keys()):
            await self.stop_recording(camera_id)

    def is_recording(self, camera_id: int) -> bool:
        recorder = self._recorders.get(camera_id)
        return recorder.is_recording if recorder else False

    def active_count(self) -> int:
        return sum(1 for r in self._recorders.values() if r.is_recording)

    def get_recording_path(self, camera_id: int, dt: datetime) -> Path:
        return Path(settings.recordings_path) / str(camera_id) / dt.strftime("%Y-%m-%d") / dt.strftime("%H")

    def _get_recording_paths(self, camera_id: int, dt: datetime) -> list[Path]:
        """Return candidate recording paths — hot (SSD) first, then cold (HDD)."""
        paths = []
        if settings.hot_storage_path:
            paths.append(Path(settings.hot_storage_path) / "recordings" / str(camera_id) / dt.strftime("%Y-%m-%d") / dt.strftime("%H"))
        paths.append(Path(settings.recordings_path) / str(camera_id) / dt.strftime("%Y-%m-%d") / dt.strftime("%H"))
        return paths

    def generate_playlist(self, camera_id: int, start: datetime, end: datetime) -> Optional[str]:
        """Generate an HLS playlist spanning a time range with accurate segment durations."""
        from datetime import timedelta

        all_entries: list[dict] = []
        current = start.replace(minute=0, second=0, microsecond=0)
        base_path = Path(settings.recordings_path) / str(camera_id)

        while current <= end:
            # Search both hot and cold storage for this hour
            for hour_dir in self._get_recording_paths(camera_id, current):
                if not hour_dir.exists():
                    continue
                # Determine the base_path relative to this hour_dir's root
                # e.g. /livenvr/recordings/{cam} or /recordings/{cam}
                bp = hour_dir.parent.parent.parent
                m3u8_path = hour_dir / "playlist.m3u8"
                if m3u8_path.exists():
                    entries = self._parse_hour_playlist(m3u8_path, current, bp, camera_id)
                else:
                    entries = self._infer_hour_segments(hour_dir, current, bp, camera_id)
                for e in entries:
                    if e["seg_end"] > start and e["seg_start"] < end:
                        all_entries.append(e)
                break  # Found this hour — don't double-count from cold
            current = current + timedelta(hours=1)

        if not all_entries:
            return None

        max_dur = max(int(e["duration"]) + 1 for e in all_entries)
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{max_dur}", "#EXT-X-MEDIA-SEQUENCE:0"]
        for e in all_entries:
            lines.append(f"#EXTINF:{e['duration']:.3f},")
            lines.append(e["url"])
        lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines)

    def _parse_hour_playlist(self, m3u8_path: Path, hour_start: datetime, base_path: Path, camera_id: int) -> list[dict]:
        """Parse an FFmpeg-generated m3u8 for accurate segment durations."""
        from datetime import timedelta

        entries = []
        current_duration = None
        target_duration = 60.0
        hour_dir = m3u8_path.parent
        raw_entries: list[tuple[str, float]] = []

        try:
            with open(m3u8_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#EXT-X-TARGETDURATION:"):
                        try:
                            target_duration = float(line.split(":")[1])
                        except ValueError:
                            pass
                    elif line.startswith("#EXTINF:"):
                        try:
                            current_duration = float(line.split(":")[1].rstrip(","))
                        except ValueError:
                            current_duration = None
                    elif not line.startswith("#") and line and current_duration is not None:
                        raw_entries.append((line, current_duration))
                        current_duration = None
        except Exception:
            return []

        if not raw_entries:
            return []

        # Determine start offset from first segment's index (handles truncated m3u8)
        first_name = Path(raw_entries[0][0]).name
        try:
            first_idx = int(first_name.split("-")[1].split(".")[0])
        except (IndexError, ValueError):
            first_idx = 0
        elapsed = first_idx * target_duration

        for seg_line, duration in raw_entries:
            # Handle both absolute and relative paths in m3u8
            seg_filename = Path(seg_line).name
            seg_path = hour_dir / seg_filename
            if seg_path.exists():
                seg_start = hour_start + timedelta(seconds=elapsed)
                seg_end = seg_start + timedelta(seconds=duration)
                rel = seg_path.relative_to(base_path)
                entries.append({
                    "url": f"/api/recordings/{camera_id}/segments/{rel}",
                    "duration": duration,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                })
            elapsed += duration

        return entries

    def _infer_hour_segments(self, hour_dir: Path, hour_start: datetime, base_path: Path, camera_id: int) -> list[dict]:
        """Fallback: infer segment timing from filename indices."""
        from datetime import timedelta

        ts_files = sorted(hour_dir.glob("segment-*.ts"))
        if not ts_files:
            return []

        # Detect segment duration: if many segments exist, they're short (4s); few = old 60s
        seg_duration = 60.0 if len(ts_files) <= 90 else 4.0
        entries = []
        for ts_file in ts_files:
            try:
                idx = int(ts_file.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            seg_start = hour_start + timedelta(seconds=idx * seg_duration)
            seg_end = seg_start + timedelta(seconds=seg_duration)
            rel = ts_file.relative_to(base_path)
            entries.append({
                "url": f"/api/recordings/{camera_id}/segments/{rel}",
                "duration": seg_duration,
                "seg_start": seg_start,
                "seg_end": seg_end,
            })
        return entries


# Singleton instance
recording_engine = RecordingEngine()
