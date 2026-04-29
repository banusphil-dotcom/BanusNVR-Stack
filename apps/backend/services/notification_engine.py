"""BanusNas — Notification Engine: Web Push + Email notifications with rules engine."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import aiosmtplib
from pywebpush import WebPushException, webpush
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.database import async_session
from models.schemas import NotificationPreference, NotificationRule, PushSubscription, SentNotification, User

logger = logging.getLogger(__name__)


# Maps a raw detector class to logical groups the user can mute. The raw
# class itself is also considered (e.g. muting "car" specifically works).
_OBJECT_TYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "person": ("person",),
    "cat": ("pet",),
    "dog": ("pet",),
    "bird": ("pet",),
    "car": ("vehicle",),
    "truck": ("vehicle",),
    "bus": ("vehicle",),
    "motorcycle": ("vehicle",),
    "bicycle": ("vehicle",),
    "motion": ("motion",),
}


def _is_muted(object_type: str, muted: list[str]) -> bool:
    if not muted or not object_type:
        return False
    muted_set = {m.lower() for m in muted}
    ot = object_type.lower()
    if ot in muted_set:
        return True
    for grp in _OBJECT_TYPE_GROUPS.get(ot, ()):
        if grp in muted_set:
            return True
    return False


def _in_quiet_hours(pref: "NotificationPreference") -> bool:
    if not pref.quiet_hours_enabled or not pref.quiet_start or not pref.quiet_end:
        return False
    local = datetime.now(timezone.utc) + timedelta(minutes=pref.timezone_offset_minutes or 0)
    current = local.strftime("%H:%M")
    start, end = pref.quiet_start, pref.quiet_end
    if start == end:
        return False
    if start < end:
        return start <= current < end
    # Overnight window (e.g. 22:00 -> 07:00)
    return current >= start or current < end


class NotificationEngine:
    """Dispatches push and email notifications based on user-defined rules."""

    def __init__(self):
        # Debounce cache: debounce_key -> last_notify_time
        self._debounce_cache: dict[str, float] = {}
        # Repeat counter: debounce_key -> consecutive_notify_count
        self._debounce_repeat: dict[str, int] = {}
        # Smart grouping: camera_id -> {names: set, timer_task, rule, user, ...}
        self._pending_groups: dict[str, dict] = {}
        self._group_window = 8.0  # seconds to wait and group detections
        # Extended presence debounce for named objects (30 min base, escalates)
        self._named_presence_debounce = 1800.0
        # After N repeats of the same named+camera, escalate to longer debounce
        self._escalation_threshold = 2
        self._escalated_debounce = 7200.0  # 2 hours after repeated notifications

    async def _fetch_subscriptions(self, user_id: int) -> list[tuple[dict, int]]:
        """Return [(webpush-payload, subscription_row_id), ...] for the user."""
        async with async_session() as session:
            rows = (await session.execute(
                select(PushSubscription).where(PushSubscription.user_id == user_id)
            )).scalars().all()
            return [
                (
                    {"endpoint": r.endpoint, "keys": {"p256dh": r.p256dh, "auth": r.auth}},
                    r.id,
                )
                for r in rows
            ]

    async def _fetch_preferences(self, user_id: int) -> NotificationPreference:
        """Return user prefs, creating a default row on first access.

        On any failure (e.g. table missing on a freshly-upgraded install
        before `Base.metadata.create_all` has run, or a transient DB error)
        we fall back to a default in-memory instance so notifications are
        never silently dropped because the prefs lookup failed.
        """
        try:
            async with async_session() as session:
                pref = (await session.execute(
                    select(NotificationPreference).where(NotificationPreference.user_id == user_id)
                )).scalar_one_or_none()
                if pref is None:
                    pref = NotificationPreference(user_id=user_id)
                    session.add(pref)
                    await session.commit()
                    await session.refresh(pref)
                return pref
        except Exception as e:
            logger.warning("NotificationPreference fetch failed for user=%d: %s — using defaults", user_id, e)
            return NotificationPreference(
                user_id=user_id,
                push_enabled=True,
                email_enabled=True,
                muted_object_types=[],
                quiet_hours_enabled=False,
                timezone_offset_minutes=0,
            )

    async def evaluate_and_notify(
        self,
        camera_id: int,
        camera_name: str,
        object_type: str,
        named_object_id: Optional[int],
        named_object_name: Optional[str],
        event_id: int,
        snapshot_path: Optional[str],
    ):
        """Evaluate all notification rules and dispatch matching notifications."""
        async with async_session() as session:
            result = await session.execute(
                select(NotificationRule).where(NotificationRule.enabled == True)
            )
            rules = result.scalars().all()

            for rule in rules:
                if not self._matches_rule(rule, camera_id, object_type, named_object_id):
                    continue

                if not self._check_schedule(rule):
                    continue

                # Debounce — respects rule.debounce_seconds. Named objects no
                # longer get a hidden 30 min floor (that suppressed legitimate
                # notifications and looked like a broken pipeline). Escalation
                # only kicks in after several repeats *within* the rule window.
                debounce_key = f"{rule.id}_{named_object_id or object_type}_{camera_id}"
                now = time.monotonic()
                last = self._debounce_cache.get(debounce_key, 0)
                repeat_count = self._debounce_repeat.get(debounce_key, 0)
                base_debounce = rule.debounce_seconds
                if named_object_id and repeat_count >= self._escalation_threshold:
                    effective_debounce = max(base_debounce, self._escalated_debounce)
                else:
                    effective_debounce = base_debounce
                if (now - last) < effective_debounce:
                    logger.debug(
                        "Notification debounced (rule=%d key=%s remaining=%.0fs)",
                        rule.id, debounce_key, effective_debounce - (now - last),
                    )
                    continue
                self._debounce_cache[debounce_key] = now
                # Track repeat count — reset if >4hr since last notification
                if last > 0 and (now - last) < 14400:
                    self._debounce_repeat[debounce_key] = repeat_count + 1
                else:
                    self._debounce_repeat[debounce_key] = 1

                # Get user for this rule
                user_result = await session.execute(
                    select(User).where(User.id == rule.user_id)
                )
                user = user_result.scalar_one_or_none()
                if not user:
                    continue

                # Smart grouping: buffer detections on the same camera to group them
                # Collapse cat/dog/bird → "Pet" so the notification matches the
                # UI label (Frigate's detector regularly mis-labels pets).
                if named_object_name:
                    display_name = named_object_name
                elif object_type and object_type.lower() in ("cat", "dog", "bird"):
                    display_name = "Pet"
                else:
                    display_name = object_type.title() if object_type else "Detection"
                group_key = f"{rule.id}_{camera_id}"

                if group_key in self._pending_groups:
                    # Add to existing group
                    grp = self._pending_groups[group_key]
                    grp["names"].append(display_name)
                    # Keep the latest snapshot and event_id
                    grp["event_id"] = event_id
                    grp["snapshot_path"] = snapshot_path or grp["snapshot_path"]
                    continue

                # Start a new group
                self._pending_groups[group_key] = {
                    "names": [display_name],
                    "camera_name": camera_name,
                    "camera_id": camera_id,
                    "event_id": event_id,
                    "snapshot_path": snapshot_path,
                    "rule": rule,
                    "user": user,
                    "object_type": object_type,
                }

                # Schedule the flush after the group window
                asyncio.create_task(self._flush_group_after_delay(group_key))

    async def _flush_group_after_delay(self, group_key: str):
        """Wait for group window then dispatch the grouped notification."""
        await asyncio.sleep(self._group_window)
        grp = self._pending_groups.pop(group_key, None)
        if not grp:
            return

        names = grp["names"]
        camera_name = grp["camera_name"]
        camera_id = grp["camera_id"]
        event_id = grp["event_id"]
        snapshot_path = grp["snapshot_path"]
        rule = grp["rule"]
        user = grp["user"]
        object_type = grp["object_type"]

        # Build smart title and body
        title, body = self._build_smart_message(names, camera_name, object_type)

        # Apply per-user preferences: object-type mute + quiet hours act as
        # global gates *before* per-rule channel logic.
        pref = await self._fetch_preferences(user.id)
        if _is_muted(object_type, pref.muted_object_types or []):
            logger.debug("Notification suppressed by user mute (user=%d type=%s)", user.id, object_type)
            return
        if _in_quiet_hours(pref):
            logger.debug("Notification suppressed by quiet hours (user=%d)", user.id)
            return

        channels = rule.channels or {}

        if channels.get("push") and pref.push_enabled:
            # Dispatch to every registered device for this user. Fall back to
            # the legacy single-subscription column for users who haven't
            # re-subscribed since the multi-device migration.
            subs = (await self._fetch_subscriptions(user.id))
            logger.info(
                "Push dispatch: user=%d devices=%d legacy=%s title=%r",
                user.id, len(subs), bool(user.push_subscription), title,
            )
            sent_any = False
            for sub_payload, sub_id in subs:
                asyncio.create_task(
                    self._send_push(sub_payload, title, body, event_id, camera_id, user.id, snapshot_path, subscription_id=sub_id)
                )
                sent_any = True
            if not sent_any and user.push_subscription:
                asyncio.create_task(
                    self._send_push(user.push_subscription, title, body, event_id, camera_id, user.id, snapshot_path)
                )
                sent_any = True
            if sent_any:
                await self._log_notification(
                    user.id, event_id, "push", title, body, camera_name, object_type
                )

        if channels.get("email") and pref.email_enabled and user.email:
            asyncio.create_task(
                self._send_email(user.email, title, body, event_id, snapshot_path)
            )
            await self._log_notification(
                user.id, event_id, "email", title, body, camera_name, object_type
            )

    @staticmethod
    def _build_smart_message(names: list[str], camera_name: str, object_type: str) -> tuple[str, str]:
        """Build a human-friendly notification title and body.

        Examples:
          - ["Bob"] -> "Bob seen in Kitchen"
          - ["Bob", "Tangie"] -> "Bob and Tangie seen in Kitchen"
          - ["Bob", "Tangie", "Bobs Phone"] -> "Bob, Tangie and Bobs Phone seen in Kitchen"
          - ["Person", "Person"] -> "2 people seen in Kitchen"
        """
        # Deduplicate while preserving order
        seen = set()
        unique: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)

        # Count generic types vs named objects
        generic = [n for n in unique if n.lower() in ("person", "car", "cat", "dog", "motion", "truck", "bus", "motorcycle", "bicycle")]
        named = [n for n in unique if n not in generic]

        if named:
            if len(named) == 1:
                who = named[0]
            elif len(named) == 2:
                who = f"{named[0]} and {named[1]}"
            else:
                who = ", ".join(named[:-1]) + f" and {named[-1]}"

            # Add generic counts if mixed
            if generic:
                who += f" (+{len(generic)} other{'s' if len(generic) > 1 else ''})"

            title = f"👤 {who} seen in {camera_name}"
            body = f"{who} {'was' if len(named) == 1 else 'were'} spotted on {camera_name}"
        elif len(unique) == 1:
            title = f"🔔 {unique[0]} detected in {camera_name}"
            body = f"{unique[0]} seen on {camera_name}"
        else:
            count = len(names)
            label = "people" if all(n.lower() == "person" for n in names) else "objects"
            title = f"🔔 {count} {label} detected in {camera_name}"
            body = f"{count} {label} seen on {camera_name}"

        return title, body

    def _matches_rule(
        self,
        rule: NotificationRule,
        camera_id: int,
        object_type: str,
        named_object_id: Optional[int],
    ) -> bool:
        """Check if an event matches a notification rule's filters."""
        # Camera filter
        if rule.camera_ids and camera_id not in rule.camera_ids:
            return False

        # Object type filter
        if rule.object_types and object_type not in rule.object_types:
            return False

        # Named object filter
        if rule.named_object_ids:
            if named_object_id is None or named_object_id not in rule.named_object_ids:
                return False

        return True

    def _check_schedule(self, rule: NotificationRule) -> bool:
        """Check if current time falls within the rule's schedule (quiet hours)."""
        schedule = rule.schedule
        if not schedule:
            return True

        now = datetime.now(timezone.utc)
        quiet_start = schedule.get("quiet_start")  # e.g., "23:00"
        quiet_end = schedule.get("quiet_end")  # e.g., "07:00"

        if quiet_start and quiet_end:
            current_time = now.strftime("%H:%M")
            if quiet_start <= quiet_end:
                if quiet_start <= current_time <= quiet_end:
                    return False
            else:  # Overnight quiet hours
                if current_time >= quiet_start or current_time <= quiet_end:
                    return False

        return True

    async def _send_push(
        self, subscription: dict, title: str, body: str, event_id: int,
        camera_id: int = 0, user_id: int = 0, snapshot_path: Optional[str] = None,
        subscription_id: Optional[int] = None,
    ):
        """Send a Web Push notification with optional snapshot image.

        If `subscription_id` is provided and the push service returns 410
        Gone, only that subscription row is deleted (other devices are kept).
        Otherwise (legacy code path) the user's `push_subscription` column is
        cleared.
        """
        try:
            from core.auth import create_access_token
            token = create_access_token(user_id, "push")
            image_url = None
            if snapshot_path and event_id:
                image_url = f"/api/events/{event_id}/snapshot?token={token}"

            payload = json.dumps({
                "title": title,
                "body": body,
                "url": f"/camera/{camera_id}",
                "image": image_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            await asyncio.to_thread(
                webpush,
                subscription,
                payload,
                vapid_private_key=settings.vapid_private_key,
                vapid_claims={"sub": settings.vapid_claim_email},
            )
            logger.info("Push notification sent (user=%d sub=%s): %s", user_id, subscription_id, title)
        except WebPushException as e:
            if "410" in str(e):
                logger.info("Push subscription expired (410 Gone) for user %d", user_id)
                try:
                    async with async_session() as session:
                        if subscription_id is not None:
                            sub = (await session.execute(
                                select(PushSubscription).where(PushSubscription.id == subscription_id)
                            )).scalar_one_or_none()
                            if sub is not None:
                                await session.delete(sub)
                                await session.commit()
                        else:
                            user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
                            if user:
                                user.push_subscription = None
                                await session.commit()
                except Exception:
                    pass
            else:
                logger.error("Push notification failed: %s", e)
        except Exception as e:
            logger.error("Push notification error: %s", e)

    async def _send_email(
        self, to_email: str, title: str, body: str, event_id: int, snapshot_path: Optional[str]
    ):
        """Send an email notification with optional snapshot attachment."""
        if not settings.smtp_host or not settings.smtp_user:
            return

        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = title
            msg["From"] = settings.smtp_from
            msg["To"] = to_email

            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px;">
                <div style="max-width: 500px; margin: 0 auto; background: #16213e; border-radius: 12px; overflow: hidden;">
                    <div style="background: #0f3460; padding: 16px 20px;">
                        <h2 style="margin: 0; color: #e94560;">🎥 BanusNas Alert</h2>
                    </div>
                    <div style="padding: 20px;">
                        <h3 style="color: #e94560; margin-top: 0;">{title}</h3>
                        <p>{body}</p>
                        <p style="color: #888; font-size: 12px;">{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                        {"<img src='cid:snapshot' style='width: 100%; border-radius: 8px; margin-top: 10px;' />" if snapshot_path else ""}
                    </div>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(html, "html"))

            if snapshot_path and Path(snapshot_path).exists():
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header("Content-ID", "<snapshot>")
                    msg.attach(img)

            await aiosmtplib.send(
                msg,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_user,
                password=settings.smtp_password,
                use_tls=settings.smtp_tls,
            )
            logger.info("Email notification sent to %s: %s", to_email, title)
        except Exception as e:
            logger.error("Email notification failed: %s", e)

    async def send_test_push(self, subscription: dict) -> bool:
        """Send a test push notification."""
        try:
            await self._send_push(subscription, "🔔 BanusNas Test", "Push notifications are working!", 0)
            return True
        except Exception:
            return False

    async def send_test_email(self, to_email: str) -> bool:
        """Send a test email."""
        try:
            await self._send_email(to_email, "🔔 BanusNas Test", "Email notifications are working!", 0, None)
            return True
        except Exception:
            return False


    async def _log_notification(
        self,
        user_id: int,
        event_id: int,
        channel: str,
        title: str,
        body: str,
        camera_name: str | None,
        object_type: str | None,
    ):
        """Persist a sent notification for the history view."""
        try:
            async with async_session() as session:
                entry = SentNotification(
                    user_id=user_id,
                    event_id=event_id,
                    channel=channel,
                    title=title,
                    body=body,
                    camera_name=camera_name,
                    object_type=object_type,
                )
                session.add(entry)
                await session.commit()
        except Exception as e:
            logger.error("Failed to log notification: %s", e)


# Singleton
notification_engine = NotificationEngine()
