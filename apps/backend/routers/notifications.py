"""BanusNas — Notifications API: push subscriptions, rules, history."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import func

from core.auth import get_current_user
from core.config import settings
from models.database import get_session
from models.schemas import NotificationRule, SentNotification, User
from schemas.api_schemas import (
    NotificationHistoryPage,
    NotificationRuleCreate,
    NotificationRuleResponse,
    NotificationRuleUpdate,
    PushSubscription,
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
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Register a Web Push subscription for the current user."""
    user.push_subscription = {"endpoint": data.endpoint, "keys": data.keys}
    session.add(user)
    await session.commit()
    return {"message": "Push subscription registered"}


@router.delete("/subscribe")
async def unsubscribe_push(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Remove Web Push subscription."""
    user.push_subscription = None
    session.add(user)
    await session.commit()
    return {"message": "Push subscription removed"}


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
):
    """Send a test notification."""
    if channel == "push":
        if not user.push_subscription:
            raise HTTPException(status_code=400, detail="No push subscription registered")
        ok = await notification_engine.send_test_push(user.push_subscription)
    elif channel == "email":
        ok = await notification_engine.send_test_email(user.email)
    else:
        raise HTTPException(status_code=400, detail="Invalid channel. Use 'push' or 'email'")

    return {"success": ok}
