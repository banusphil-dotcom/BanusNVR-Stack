"""BanusNas — Notifications API: push subscriptions, rules, history."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import func

from core.auth import get_current_user
from core.config import settings
from models.database import get_session
from models.schemas import (
    NotificationPreference,
    NotificationRule,
    PushSubscription as PushSubscriptionRow,
    SentNotification,
    User,
)
from schemas.api_schemas import (
    NotificationHistoryPage,
    NotificationPreferenceResponse,
    NotificationPreferenceUpdate,
    NotificationRuleCreate,
    NotificationRuleResponse,
    NotificationRuleUpdate,
    PushSubscription,
    PushSubscriptionRename,
    PushSubscriptionResponse,
    SentNotificationResponse,
)
from services.notification_engine import notification_engine

router = APIRouter(prefix="/api/notifications", tags=["notifications"], dependencies=[Depends(get_current_user)])


@router.get("/vapid-key")
async def get_vapid_key():
    """Return the VAPID public key for push subscription."""
    return {"vapid_public_key": settings.vapid_public_key}


@router.post("/subscribe")
async def subscribe_push(
    data: PushSubscription,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Register a Web Push subscription for the current user's device.

    Each device produces a unique `endpoint`. We upsert by endpoint so a
    re-registration from the same browser refreshes keys/last_used_at
    instead of duplicating a row. Other devices for the same user are
    untouched.
    """
    p256dh = (data.keys or {}).get("p256dh") or ""
    auth_k = (data.keys or {}).get("auth") or ""
    if not data.endpoint or not p256dh or not auth_k:
        raise HTTPException(status_code=400, detail="Invalid push subscription payload")

    user_agent = (request.headers.get("user-agent") or "")[:500] or None

    existing = (await session.execute(
        select(PushSubscriptionRow).where(PushSubscriptionRow.endpoint == data.endpoint)
    )).scalar_one_or_none()

    if existing is not None:
        # If the endpoint was previously registered to a different user, take
        # ownership for the current user (browser may have switched accounts).
        existing.user_id = user.id
        existing.p256dh = p256dh
        existing.auth = auth_k
        if data.device_name:
            existing.device_name = data.device_name
        if user_agent:
            existing.user_agent = user_agent
        existing.last_used_at = datetime.now(timezone.utc)
        sub = existing
    else:
        sub = PushSubscriptionRow(
            user_id=user.id,
            endpoint=data.endpoint,
            p256dh=p256dh,
            auth=auth_k,
            device_name=data.device_name,
            user_agent=user_agent,
        )
        session.add(sub)

    # Clear the legacy single-subscription column so the engine's multi-device
    # path is the sole source of truth.
    user.push_subscription = None
    session.add(user)

    await session.commit()
    await session.refresh(sub)
    return {"message": "Push subscription registered", "id": sub.id}


@router.delete("/subscribe")
async def unsubscribe_push(
    endpoint: str | None = None,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Remove the current device's push subscription.

    If `endpoint` is provided, removes that specific subscription. Otherwise
    (legacy behaviour) removes ALL the user's subscriptions.
    """
    from sqlalchemy import delete as sql_delete
    if endpoint:
        await session.execute(
            sql_delete(PushSubscriptionRow).where(
                PushSubscriptionRow.user_id == user.id,
                PushSubscriptionRow.endpoint == endpoint,
            )
        )
    else:
        await session.execute(
            sql_delete(PushSubscriptionRow).where(PushSubscriptionRow.user_id == user.id)
        )
        user.push_subscription = None
        session.add(user)
    await session.commit()
    return {"message": "Push subscription removed"}


@router.get("/subscriptions", response_model=list[PushSubscriptionResponse])
async def list_subscriptions(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """List all push subscriptions (devices) for the current user.

    `is_current` is set to true on the row whose endpoint matches the
    `X-Push-Endpoint` request header (sent by the frontend so the UI can
    label the current device).
    """
    current_endpoint = request.headers.get("x-push-endpoint") or ""
    rows = (await session.execute(
        select(PushSubscriptionRow)
        .where(PushSubscriptionRow.user_id == user.id)
        .order_by(desc(PushSubscriptionRow.last_used_at))
    )).scalars().all()
    out: list[PushSubscriptionResponse] = []
    for r in rows:
        item = PushSubscriptionResponse.model_validate(r)
        item.is_current = bool(current_endpoint and current_endpoint == r.endpoint)
        out.append(item)
    return out


@router.patch("/subscriptions/{sub_id}", response_model=PushSubscriptionResponse)
async def rename_subscription(
    sub_id: int,
    data: PushSubscriptionRename,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    sub = (await session.execute(
        select(PushSubscriptionRow).where(
            PushSubscriptionRow.id == sub_id,
            PushSubscriptionRow.user_id == user.id,
        )
    )).scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    sub.device_name = data.device_name.strip()
    session.add(sub)
    await session.commit()
    await session.refresh(sub)
    return PushSubscriptionResponse.model_validate(sub)


@router.delete("/subscriptions/{sub_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_subscription(
    sub_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    sub = (await session.execute(
        select(PushSubscriptionRow).where(
            PushSubscriptionRow.id == sub_id,
            PushSubscriptionRow.user_id == user.id,
        )
    )).scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    await session.delete(sub)
    await session.commit()


# --- Per-user preferences ---


async def _get_or_create_preferences(session: AsyncSession, user_id: int) -> NotificationPreference:
    pref = (await session.execute(
        select(NotificationPreference).where(NotificationPreference.user_id == user_id)
    )).scalar_one_or_none()
    if pref is None:
        pref = NotificationPreference(user_id=user_id)
        session.add(pref)
        await session.commit()
        await session.refresh(pref)
    return pref


@router.get("/preferences", response_model=NotificationPreferenceResponse)
async def get_preferences(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    pref = await _get_or_create_preferences(session, user.id)
    return NotificationPreferenceResponse.model_validate(pref)


@router.put("/preferences", response_model=NotificationPreferenceResponse)
async def update_preferences(
    data: NotificationPreferenceUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    pref = await _get_or_create_preferences(session, user.id)
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(pref, key, value)
    session.add(pref)
    await session.commit()
    await session.refresh(pref)
    return NotificationPreferenceResponse.model_validate(pref)


@router.get("/rules", response_model=list[NotificationRuleResponse])
async def list_rules(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(NotificationRule)
        .where(NotificationRule.user_id == user.id)
        .order_by(desc(NotificationRule.created_at))
    )
    return result.scalars().all()


@router.post("/rules", response_model=NotificationRuleResponse, status_code=status.HTTP_201_CREATED)
async def create_rule(
    data: NotificationRuleCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    rule = NotificationRule(
        user_id=user.id,
        name=data.name,
        object_types=data.object_types,
        named_object_ids=data.named_object_ids,
        camera_ids=data.camera_ids,
        schedule=data.schedule,
        channels=data.channels,
        debounce_seconds=data.debounce_seconds,
    )
    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    return rule


@router.put("/rules/{rule_id}", response_model=NotificationRuleResponse)
async def update_rule(
    rule_id: int,
    data: NotificationRuleUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(NotificationRule).where(
            NotificationRule.id == rule_id, NotificationRule.user_id == user.id
        )
    )
    rule = result.scalar_one_or_none()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(rule, key, value)

    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(
    rule_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(NotificationRule).where(
            NotificationRule.id == rule_id, NotificationRule.user_id == user.id
        )
    )
    rule = result.scalar_one_or_none()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    await session.delete(rule)
    await session.commit()


# --- Notification History ---


@router.get("/history", response_model=NotificationHistoryPage)
async def get_notification_history(
    page: int = 1,
    page_size: int = 30,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Paginated notification history for the current user."""
    base = select(SentNotification).where(SentNotification.user_id == user.id)
    total_q = await session.execute(select(func.count()).select_from(base.subquery()))
    total = total_q.scalar() or 0

    result = await session.execute(
        base.order_by(desc(SentNotification.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return NotificationHistoryPage(
        items=result.scalars().all(),
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/unread-count")
async def unread_count(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(func.count()).where(
            SentNotification.user_id == user.id,
            SentNotification.read == False,
        )
    )
    return {"count": result.scalar() or 0}


@router.post("/mark-read")
async def mark_all_read(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Mark all notifications as read for the current user."""
    from sqlalchemy import update
    await session.execute(
        update(SentNotification)
        .where(SentNotification.user_id == user.id, SentNotification.read == False)
        .values(read=True)
    )
    await session.commit()
    return {"message": "All notifications marked as read"}


@router.delete("/history")
async def clear_history(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Delete all notification history for the current user."""
    from sqlalchemy import delete as sql_delete
    await session.execute(
        sql_delete(SentNotification).where(SentNotification.user_id == user.id)
    )
    await session.commit()
    return {"message": "Notification history cleared"}


@router.post("/test")
async def test_notification(
    channel: str = "push",
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Send a test notification to all of the user's registered devices."""
    if channel == "push":
        subs = (await session.execute(
            select(PushSubscriptionRow).where(PushSubscriptionRow.user_id == user.id)
        )).scalars().all()
        if not subs and not user.push_subscription:
            raise HTTPException(status_code=400, detail="No push subscription registered")
        ok_any = False
        for sub in subs:
            payload = {"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}}
            if await notification_engine.send_test_push(payload):
                ok_any = True
        if not subs and user.push_subscription:
            ok_any = await notification_engine.send_test_push(user.push_subscription)
        return {"success": ok_any, "devices": len(subs) or (1 if user.push_subscription else 0)}
    elif channel == "email":
        ok = await notification_engine.send_test_email(user.email)
    else:
        raise HTTPException(status_code=400, detail="Invalid channel. Use 'push' or 'email'")

    return {"success": ok}
