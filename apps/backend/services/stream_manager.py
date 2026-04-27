"""BanusNas — Stream Manager: go2rtc HTTP API stream management."""

import asyncio
import logging
from typing import Optional

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages camera streams via go2rtc HTTP API (embedded in Frigate)."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = settings.go2rtc_api_url

    async def start(self):
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=10.0)
        logger.info("StreamManager started — go2rtc at %s", self._base_url)

    async def stop(self):
        if self._client:
            await self._client.aclose()
            logger.info("StreamManager stopped")

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Auto-start if not explicitly started (lazy init)
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=10.0)
        return self._client

    async def add_stream(self, name: str, source_url: str) -> bool:
        """Register a camera stream via go2rtc PUT /api/streams."""
        logger.info("Registering stream %s → %s", name, source_url[:80])
        try:
            resp = await self.client.put(
                "/api/streams",
                params={"src": name, "url": source_url},
            )
            if resp.status_code in (200, 201):
                logger.info("Stream %s registered successfully", name)
                return True
            logger.error("Failed to register stream %s: HTTP %d — %s", name, resp.status_code, resp.text[:200])
            return False
        except Exception as e:
            logger.error("Failed to add stream %s: %s", name, e)
            return False

    async def remove_stream(self, name: str) -> bool:
        """Remove a camera stream via go2rtc DELETE /api/streams."""
        try:
            resp = await self.client.delete(
                "/api/streams",
                params={"src": name},
            )
            if resp.status_code in (200, 204):
                logger.info("Stream removed: %s", name)
                return True
            logger.warning("Remove stream %s: HTTP %d", name, resp.status_code)
            return True  # Non-fatal — stream may not exist
        except Exception as e:
            logger.error("Failed to remove stream %s: %s", name, e)
            return False

    async def get_streams(self) -> dict:
        """List all active streams from go2rtc."""
        try:
            resp = await self.client.get("/api/streams")
            if resp.status_code == 200:
                return resp.json()
            return {}
        except httpx.HTTPError as e:
            logger.error("Failed to list streams: %s", e)
            return {}

    async def get_snapshot(self, camera_name: str) -> Optional[bytes]:
        """Get a JPEG snapshot frame from a camera via go2rtc."""
        try:
            resp = await self.client.get(
                "/api/frame.jpeg",
                params={"src": camera_name},
                timeout=15.0,
            )
            if resp.status_code == 200 and len(resp.content) > 100:
                return resp.content
            return None
        except httpx.HTTPError as e:
            logger.error("Failed to get snapshot for %s: %s", camera_name, e)
            return None

    async def test_stream(self, source_url: str, name: str = "__test__") -> bool:
        """Test if a camera stream URL is reachable via go2rtc."""
        try:
            added = await self.add_stream(name, source_url)
            if not added:
                return False
            await asyncio.sleep(2)
            snapshot = await self.get_snapshot(name)
            await self.remove_stream(name)
            return snapshot is not None
        except Exception as e:
            logger.error("Stream test failed: %s", e)
            await self.remove_stream(name)
            return False

    def get_rtsp_url(self, camera_name: str) -> str:
        """Return the go2rtc RTSP restream URL for a camera."""
        # Derive RTSP host from the API URL (same host, port 8554)
        from urllib.parse import urlparse
        parsed = urlparse(self._base_url)
        rtsp_host = parsed.hostname or "host.docker.internal"
        return f"rtsp://{rtsp_host}:8554/{camera_name}"

    def get_webrtc_url(self, camera_name: str) -> str:
        """Return the go2rtc WebRTC API URL for a camera."""
        return f"{self._base_url}/api/webrtc?src={camera_name}"

    def get_hls_url(self, camera_name: str) -> str:
        """Return the go2rtc HLS stream URL for a camera."""
        return f"{self._base_url}/api/stream.m3u8?src={camera_name}"

    async def health_check(self) -> bool:
        """Check if go2rtc is responsive."""
        try:
            resp = await self.client.get("/api/streams")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


# Singleton instance
stream_manager = StreamManager()
