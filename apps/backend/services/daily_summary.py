"""BanusNas — Daily Summary Service: AI-generated activity summaries."""

import asyncio
import base64
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.database import async_session
from models.schemas import Camera, DailySummary, Event, EventType, NamedObject, ObjectCategory

logger = logging.getLogger(__name__)

_PERIOD_RANGES = [
    ("night",    0,  6,  "Night (00:00–06:00)"),
    ("morning",  6, 12,  "Morning (06:00–12:00)"),
    ("afternoon",12, 18, "Afternoon (12:00–18:00)"),
    ("evening", 18, 24,  "Evening (18:00–00:00)"),
]


class DailySummaryService:
    """Generates and schedules daily activity summaries."""

    def __init__(self):
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the daily summary scheduler."""
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Daily summary scheduler started")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Daily summary scheduler stopped")

    async def _scheduler_loop(self):
        """Run forever, checking every minute if it's time for a summary."""
        MORNING_HOUR, MORNING_MINUTE = 6, 0
        EVENING_HOUR, EVENING_MINUTE = 19, 0

        last_morning: str | None = None
        last_evening: str | None = None

        while True:
            try:
                now = datetime.now()  # Uses container TZ (Europe/London via TZ env)
                today = now.strftime("%Y-%m-%d")

                # Morning summary at 6:00 — covers the previous day
                if (now.hour == MORNING_HOUR and now.minute == MORNING_MINUTE
                        and last_morning != today):
                    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                    await self.generate_summary(yesterday, "morning")
                    last_morning = today

                # Evening summary at 19:00 — covers today so far
                if (now.hour == EVENING_HOUR and now.minute == EVENING_MINUTE
                        and last_evening != today):
                    await self.generate_summary(today, "evening")
                    last_evening = today

            except Exception as e:
                logger.error("Summary scheduler error: %s", e, exc_info=True)

            await asyncio.sleep(60)

    async def generate_summary(self, date_str: str, summary_type: str) -> dict:
        """Generate a summary for the given date and type.

        Args:
            date_str: YYYY-MM-DD
            summary_type: 'morning' (full day) or 'evening' (up to now)
        """
        logger.info("Generating %s summary for %s", summary_type, date_str)

        async with async_session() as session:
            # Check if already generated
            existing = await session.execute(
                select(DailySummary).where(
                    DailySummary.date == date_str,
                    DailySummary.summary_type == summary_type,
                )
            )
            if existing.scalar_one_or_none():
                # Re-generate (update)
                pass

            data = await self._build_summary_data(session, date_str, summary_type)

            # Upsert
            result = await session.execute(
                select(DailySummary).where(
                    DailySummary.date == date_str,
                    DailySummary.summary_type == summary_type,
                )
            )
            record = result.scalar_one_or_none()
            if record:
                record.data = data
                record.generated_at = datetime.now(timezone.utc)
            else:
                record = DailySummary(
                    date=date_str,
                    summary_type=summary_type,
                    data=data,
                )
                session.add(record)

            await session.commit()
            logger.info("Summary generated: %s %s (%d people, %d pets)",
                        date_str, summary_type,
                        len(data.get("people", [])), len(data.get("pets", [])))

            # Send push notification
            await self._send_summary_push(data, date_str, summary_type)

            return data

    async def _build_summary_data(
        self, session: AsyncSession, date_str: str, summary_type: str,
    ) -> dict:
        """Build structured summary data from events."""
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start = date_obj
        if summary_type == "evening":
            end = datetime.now(timezone.utc)
        else:
            end = date_obj + timedelta(days=1)

        # Get all events in range
        result = await session.execute(
            select(Event)
            .where(Event.started_at >= start, Event.started_at < end)
            .order_by(Event.started_at)
        )
        events = result.scalars().all()

        # Get cameras
        cam_result = await session.execute(select(Camera))
        cameras = {c.id: c.name for c in cam_result.scalars().all()}

        # Get named objects
        no_result = await session.execute(select(NamedObject))
        named_objects = {n.id: n for n in no_result.scalars().all()}

        # Aggregate people, pets, vehicles with multiple snapshots + time-period breakdown
        people: dict[str, dict] = {}
        pets: dict[str, dict] = {}
        vehicles: dict[str, dict] = {}
        unknown_counts: dict[str, int] = {}

        def _get_period(hour: int) -> str:
            for name, h_start, h_end, _ in _PERIOD_RANGES:
                if h_start <= hour < h_end:
                    return name
            return "night"

        for ev in events:
            obj_type = ev.object_type or "unknown"
            camera_name = cameras.get(ev.camera_id, f"Camera {ev.camera_id}")
            time_str = ev.started_at.strftime("%H:%M")
            period = _get_period(ev.started_at.hour)

            if ev.named_object_id and ev.named_object_id in named_objects:
                no = named_objects[ev.named_object_id]
                category = no.category.value if no.category else "other"

                target = None
                if category == "person":
                    target = people
                elif category == "pet":
                    target = pets
                elif category == "vehicle":
                    target = vehicles

                if target is not None:
                    entry = target.setdefault(no.name, {
                        "name": no.name,
                        "first_seen": time_str,
                        "last_seen": time_str,
                        "locations": [],
                        "event_count": 0,
                        "snapshot_event_id": None,
                        "species": obj_type if category == "pet" else None,
                        "snapshots": [],  # [{event_id, time, camera, snapshot_path}]
                        "by_period": {"night": 0, "morning": 0, "afternoon": 0, "evening": 0},
                    })
                    entry["last_seen"] = time_str
                    entry["event_count"] += 1
                    entry["by_period"][period] += 1
                    if camera_name not in entry["locations"]:
                        entry["locations"].append(camera_name)

                    # Collect up to 3 best snapshots (spaced across the day)
                    if ev.snapshot_path and Path(ev.snapshot_path).exists():
                        existing_times = [s["time"] for s in entry["snapshots"]]
                        # Only add if we have < 3 or this is in a different period
                        if len(entry["snapshots"]) < 3 or period not in [
                            _get_period(int(t.split(":")[0])) for t in existing_times
                        ]:
                            snap = {
                                "event_id": ev.id,
                                "time": time_str,
                                "camera": camera_name,
                                "snapshot_path": ev.snapshot_path,
                                "description": "",  # filled by vision model
                            }
                            if len(entry["snapshots"]) >= 3:
                                # Replace the one closest in time to another
                                entry["snapshots"][-1] = snap
                            else:
                                entry["snapshots"].append(snap)

                    # Best single snapshot for the card
                    if ev.snapshot_path and (
                        entry["snapshot_event_id"] is None
                        or (ev.confidence or 0) > 0.5
                    ):
                        entry["snapshot_event_id"] = ev.id
            else:
                unknown_counts[obj_type] = unknown_counts.get(obj_type, 0) + 1

        # Build activity timeline (hourly buckets)
        hourly: dict[int, dict[str, int]] = {}
        for ev in events:
            hour = ev.started_at.hour
            bucket = hourly.setdefault(hour, {})
            obj_type = ev.object_type or "unknown"
            bucket[obj_type] = bucket.get(obj_type, 0) + 1

        timeline = []
        for hour in sorted(hourly.keys()):
            timeline.append({
                "hour": hour,
                "label": f"{hour:02d}:00",
                "counts": hourly[hour],
            })

        # Build period summaries
        activity_periods = []
        for period_name, h_start, h_end, label in _PERIOD_RANGES:
            period_events = [e for e in events if h_start <= e.started_at.hour < h_end]
            if period_events:
                p_people = set()
                p_pets = set()
                for e in period_events:
                    if e.named_object_id and e.named_object_id in named_objects:
                        no = named_objects[e.named_object_id]
                        cat = no.category.value if no.category else "other"
                        if cat == "person":
                            p_people.add(no.name)
                        elif cat == "pet":
                            p_pets.add(no.name)
                activity_periods.append({
                    "period": period_name,
                    "label": label,
                    "total_events": len(period_events),
                    "people": sorted(p_people),
                    "pets": sorted(p_pets),
                })

        # Summary stats
        total_events = len(events)
        total_people = sum(1 for e in events if e.object_type == "person")
        total_pets = sum(1 for e in events if e.object_type in ("cat", "dog"))

        # Greeting text
        if summary_type == "morning":
            greeting = "Good morning! Here's what happened yesterday."
        else:
            greeting = "Good evening! Here's today's activity so far."

        summary_data = {
            "greeting": greeting,
            "date": date_str,
            "summary_type": summary_type,
            "total_events": total_events,
            "total_people_events": total_people,
            "total_pet_events": total_pets,
            "people": list(people.values()),
            "pets": list(pets.values()),
            "vehicles": list(vehicles.values()),
            "unknown_counts": unknown_counts,
            "timeline": timeline,
            "activity_periods": activity_periods,
            "narrative": "",
        }

        # Analyze snapshots with vision model (skip for local — too slow on CPU)
        # Vision descriptions are added during deep generation only
        narrative, source = await self._generate_narrative(summary_data)
        summary_data["narrative"] = narrative
        summary_data["narrative_source"] = source
        if source != "fallback":
            summary_data["greeting"] = ""

        # Strip internal snapshot_path from output (not needed by frontend)
        for collection in [summary_data["people"], summary_data["pets"], summary_data["vehicles"]]:
            for entry in collection:
                for snap in entry.get("snapshots", []):
                    snap.pop("snapshot_path", None)

        return summary_data

    async def _analyze_snapshots(self, data: dict) -> None:
        """Use a vision model to describe key snapshots for each person/pet."""
        import httpx

        all_snaps: list[dict] = []
        for collection in [data["people"], data["pets"], data["vehicles"]]:
            for entry in collection:
                for snap in entry.get("snapshots", []):
                    if snap.get("snapshot_path"):
                        snap["_entity_name"] = entry["name"]
                        all_snaps.append(snap)

        if not all_snaps:
            return

        # Limit total snapshots to avoid excessive processing time on N100
        all_snaps = all_snaps[:4]
        logger.info("Analyzing %d snapshots with vision model (%s)", len(all_snaps), settings.ollama_vision_model)

        async with httpx.AsyncClient(timeout=180.0) as client:
            for snap in all_snaps:
                try:
                    img_path = Path(snap["snapshot_path"])
                    if not img_path.exists():
                        continue

                    img_bytes = await asyncio.to_thread(img_path.read_bytes)
                    img_b64 = base64.b64encode(img_bytes).decode("ascii")

                    # Use Ollama native API with images field (more reliable for vision)
                    response = await client.post(
                        f"{settings.ollama_url}/api/chat",
                        json={
                            "model": settings.ollama_vision_model,
                            "messages": [{
                                "role": "user",
                                "content": (
                                    f"This is a CCTV security camera image from the '{snap['camera']}' camera "
                                    f"at {snap['time']}. The detected subject is '{snap['_entity_name']}'. "
                                    "In one short sentence, describe what the person or animal is DOING — "
                                    "their action (e.g., walking, entering, sitting, leaving). "
                                    "Do not describe their appearance or the scene. Keep it under 15 words."
                                ),
                                "images": [img_b64],
                            }],
                            "stream": False,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    description = result.get("message", {}).get("content", "").strip()
                    snap["description"] = description
                    logger.info("Vision: %s @ %s — %s", snap["_entity_name"], snap["time"], description[:80])

                except Exception as e:
                    logger.warning("Vision analysis failed for %s @ %s: %s",
                                   snap.get("_entity_name"), snap.get("time"), e,
                                   exc_info=True)

        # Clean up internal fields
        for snap in all_snaps:
            snap.pop("_entity_name", None)

    async def generate_deep_narrative(self, date_str: str, summary_type: str) -> dict:
        """Re-generate summary narrative using the deep ML endpoint."""
        async with async_session() as session:
            result = await session.execute(
                select(DailySummary).where(
                    DailySummary.date == date_str,
                    DailySummary.summary_type == summary_type,
                )
            )
            record = result.scalar_one_or_none()
            if not record:
                # Generate base summary first
                data = await self.generate_summary(date_str, summary_type)
            else:
                data = dict(record.data)

            # TODO: Run vision analysis when deep ML server has a GPU
            # await self._analyze_snapshots(data)

            # Call deep ML endpoint
            narrative = await self._call_llm(
                url=f"{settings.deep_ml_url}/v1/chat/completions",
                model="",  # server decides
                data=data,
                timeout=120.0,
            )
            if narrative is not None:
                data["narrative"] = narrative
                data["narrative_source"] = "deep"
                data["greeting"] = ""
            else:
                data["narrative_source"] = "deep_failed"

            # Update record
            result2 = await session.execute(
                select(DailySummary).where(
                    DailySummary.date == date_str,
                    DailySummary.summary_type == summary_type,
                )
            )
            record = result2.scalar_one_or_none()
            if record:
                record.data = data
                record.generated_at = datetime.now(timezone.utc)
            else:
                record = DailySummary(date=date_str, summary_type=summary_type, data=data)
                session.add(record)
            await session.commit()
            return data

    async def _generate_narrative(self, data: dict) -> tuple[str, str]:
        """Generate a blog/news-style AI narrative using local Ollama.
        Returns (narrative_text, source) where source is 'local' or 'fallback'.
        """
        result = await self._call_llm(
            url=f"{settings.ollama_url}/v1/chat/completions",
            model=settings.ollama_model,
            data=data,
        )
        if result is not None:
            return result, "local"
        return self._generate_fallback_narrative(data), "fallback"

    async def _call_llm(self, url: str, model: str, data: dict, timeout: float = 120.0) -> str | None:
        """Call an OpenAI-compatible chat completions endpoint. Returns None on failure."""
        try:
            import httpx

            prompt = self._build_narrative_prompt(data)
            body: dict = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a smart home security reporter writing a detailed, engaging daily activity "
                            "summary for a family's home CCTV system. Write in a warm, personal tone — like a "
                            "friendly neighbourhood newsletter or a Samsung Now Brief.\n\n"
                            "Structure the summary as follows:\n"
                            "1. An opening overview paragraph with the headline numbers.\n"
                            "2. Break the day into time periods (Morning, Afternoon, Evening) with a blank line "
                            "between each. Start each period section with the period name in bold like **Morning**.\n"
                            "3. Within each period, describe who was seen, what they appeared to be doing "
                            "(based on any photo descriptions provided), and where.\n"
                            "4. End with a short closing observation about the day.\n\n"
                            "Use specific details from the photo descriptions when available — mention clothing, "
                            "posture, activity. Use a few relevant emoji sparingly. Don't be overly formal. "
                            "Do NOT use markdown bullet lists — write flowing prose paragraphs only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 800,
            }
            if model:
                body["model"] = model

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=body,
                )
                response.raise_for_status()
                result = response.json()
                narrative = result["choices"][0]["message"]["content"].strip()
                logger.info("AI narrative generated via %s (%d chars)", url, len(narrative))
                return narrative

        except Exception as e:
            logger.error("AI narrative generation failed (%s): %s", url, e)
            return None

    def _build_narrative_prompt(self, data: dict) -> str:
        """Build a structured prompt for the AI from summary data, including vision descriptions."""
        date_str = data["date"]
        summary_type = data["summary_type"]
        period = "yesterday" if summary_type == "morning" else "today so far"

        lines = [f"Generate a detailed daily activity summary for {period} ({date_str})."]
        lines.append(f"Total events detected: {data['total_events']}")

        # Time period breakdown
        for ap in data.get("activity_periods", []):
            parts = [f"{ap['total_events']} events"]
            if ap["people"]:
                parts.append(f"people: {', '.join(ap['people'])}")
            if ap["pets"]:
                parts.append(f"pets: {', '.join(ap['pets'])}")
            lines.append(f"\n{ap['label']}: {'; '.join(parts)}")

        if data["people"]:
            lines.append("\nPeople seen:")
            for p in data["people"]:
                locs = ", ".join(p["locations"])
                by_p = p.get("by_period", {})
                period_detail = ", ".join(f"{k}: {v}" for k, v in by_p.items() if v > 0)
                lines.append(
                    f"  - {p['name']}: first seen {p['first_seen']}, last seen {p['last_seen']}, "
                    f"spotted {p['event_count']} times in {locs} (by period: {period_detail})"
                )
                for snap in p.get("snapshots", []):
                    if snap.get("description"):
                        lines.append(f"    [Photo at {snap['time']} in {snap['camera']}]: {snap['description']}")

        if data["pets"]:
            lines.append("\nPets seen:")
            for p in data["pets"]:
                species = p.get("species", "pet")
                locs = ", ".join(p["locations"])
                by_p = p.get("by_period", {})
                period_detail = ", ".join(f"{k}: {v}" for k, v in by_p.items() if v > 0)
                lines.append(
                    f"  - {p['name']} ({species}): first seen {p['first_seen']}, last seen {p['last_seen']}, "
                    f"spotted {p['event_count']} times in {locs} (by period: {period_detail})"
                )
                for snap in p.get("snapshots", []):
                    if snap.get("description"):
                        lines.append(f"    [Photo at {snap['time']} in {snap['camera']}]: {snap['description']}")

        if data["vehicles"]:
            lines.append("\nVehicles:")
            for v in data["vehicles"]:
                locs = ", ".join(v["locations"])
                lines.append(f"  - {v['name']}: seen {v['event_count']} times in {locs}")
                for snap in v.get("snapshots", []):
                    if snap.get("description"):
                        lines.append(f"    [Photo at {snap['time']} in {snap['camera']}]: {snap['description']}")

        if data["unknown_counts"]:
            unknown_parts = [f"{count} {obj_type}(s)" for obj_type, count in data["unknown_counts"].items()]
            lines.append(f"\nUnidentified detections: {', '.join(unknown_parts)}")

        if data["timeline"]:
            busy_hours = sorted(data["timeline"], key=lambda t: sum(t["counts"].values()), reverse=True)[:3]
            busy_str = ", ".join(f"{h['label']} ({sum(h['counts'].values())} events)" for h in busy_hours)
            lines.append(f"\nBusiest hours: {busy_str}")

        lines.append(
            "\nWrite the summary broken down by time period (morning, afternoon, evening). "
            "Reference what each person or pet was doing based on the photo descriptions provided. "
            "Mention specific observations from the photos to make the summary vivid and accurate."
        )

        return "\n".join(lines)

    @staticmethod
    def _generate_fallback_narrative(data: dict) -> str:
        """Generate a simple narrative without AI when no API key is available."""
        date_str = data["date"]
        summary_type = data["summary_type"]
        total = data["total_events"]

        if total == 0:
            return "It was a quiet day with no activity detected across any cameras."

        period = "Yesterday" if summary_type == "morning" else "Today so far"
        parts = [f"{period} saw {total} events across your cameras."]

        people = data.get("people", [])
        if people:
            names = [p["name"] for p in people]
            if len(names) == 1:
                p = people[0]
                parts.append(
                    f"{p['name']} was spotted {p['event_count']} times, "
                    f"first appearing at {p['first_seen']} in {', '.join(p['locations'])}."
                )
            else:
                parts.append(f"{', '.join(names[:-1])} and {names[-1]} were seen around the house.")

        pets = data.get("pets", [])
        if pets:
            for p in pets:
                species = p.get("species", "pet")
                parts.append(
                    f"{p['name']} the {species} was active between {p['first_seen']} and {p['last_seen']}, "
                    f"spotted {p['event_count']} times in {', '.join(p['locations'])}."
                )

        unknown = data.get("unknown_counts", {})
        if unknown:
            unid_parts = [f"{c} {t}{'s' if c > 1 else ''}" for t, c in unknown.items()]
            parts.append(f"There were also {', '.join(unid_parts)} detected but not identified.")

        return " ".join(parts)

    async def _send_summary_push(self, data: dict, date_str: str, summary_type: str):
        """Send a push notification with summary highlights."""
        try:
            from services.notification_engine import notification_engine

            people_names = [p["name"] for p in data.get("people", [])]
            pet_names = [p["name"] for p in data.get("pets", [])]

            parts = []
            if people_names:
                parts.append(f"People: {', '.join(people_names[:4])}")
            if pet_names:
                parts.append(f"Pets: {', '.join(pet_names[:4])}")
            if not parts:
                parts.append(f"{data.get('total_events', 0)} events detected")

            period = "Yesterday" if summary_type == "morning" else "Today so far"
            title = f"📊 {period}'s Activity Summary"
            body = " · ".join(parts)

            # Send to all users with push subscriptions
            from models.schemas import User as UserModel
            from sqlalchemy import select as sel

            async with async_session() as session:
                users = (await session.execute(
                    sel(UserModel).where(UserModel.push_subscription.isnot(None))
                )).scalars().all()

                for user in users:
                    if user.push_subscription:
                        await notification_engine._send_push(
                            user.push_subscription, title, body,
                            event_id=0, camera_id=0, user_id=user.id,
                        )

        except Exception as e:
            logger.error("Failed to send summary push: %s", e)

    async def get_summary(self, date_str: str, summary_type: str | None = None) -> list[dict]:
        """Retrieve stored summary(ies) for a date."""
        async with async_session() as session:
            query = select(DailySummary).where(DailySummary.date == date_str)
            if summary_type:
                query = query.where(DailySummary.summary_type == summary_type)
            query = query.order_by(DailySummary.generated_at.desc())

            result = await session.execute(query)
            records = result.scalars().all()

            return [
                {
                    "id": r.id,
                    "date": r.date,
                    "summary_type": r.summary_type,
                    "data": r.data,
                    "generated_at": r.generated_at.isoformat(),
                }
                for r in records
            ]

    async def get_available_dates(self, limit: int = 30) -> list[str]:
        """Get dates that have summaries, most recent first."""
        async with async_session() as session:
            result = await session.execute(
                select(DailySummary.date)
                .distinct()
                .order_by(DailySummary.date.desc())
                .limit(limit)
            )
            return [r[0] for r in result.all()]


# Singleton
daily_summary_service = DailySummaryService()
