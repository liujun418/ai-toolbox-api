"""Admin management endpoints. All routes require admin role."""

import logging
import uuid
from datetime import datetime, UTC, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Task, TaskStatus, Transaction
from app.schemas.admin import (
    AdminStatsResponse,
    AdminUserResponse,
    AdminUserListResponse,
    AdminUserDetailResponse,
    AdjustCreditsRequest,
    SetUserRoleRequest,
    AdminTaskItem,
    AdminTaskListResponse,
    AdminTransactionItem,
    AdminTransactionListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


def require_admin(user: User = Depends(lambda: None)) -> User:
    """Dependency that enforces admin role. Must be used with get_current_user."""
    if user.role != "admin":
        logger.warning("Non-admin user %s attempted admin access", user.email)
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _admin_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "credits": user.credits,
        "email_verified": user.email_verified,
        "created_at": user.created_at,
    }


# --- Helper to inject get_current_user into require_admin ---
# We compose dependencies at the endpoint level, not at router level.

def _get_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db),
) -> User:
    """Composite: authenticate + check admin role."""
    from app.routers.auth import get_current_user
    user = get_current_user(credentials=credentials, db=db)
    if user.role != "admin":
        logger.warning("Non-admin user %s attempted admin access", user.email)
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# --- Dashboard Stats ---

@router.get("/dashboard", response_model=AdminStatsResponse)
def get_dashboard_stats(admin: User = Depends(_get_admin_user), db: Session = Depends(get_db)):
    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    # User counts
    total_users = db.query(User).count()
    new_today = db.query(User).filter(User.created_at >= today_start).count()
    new_week = db.query(User).filter(User.created_at >= week_start).count()
    new_month = db.query(User).filter(User.created_at >= month_start).count()

    # Revenue / credits
    total_revenue = db.query(func.sum(Transaction.amount)).filter(
        Transaction.type == "purchase"
    ).scalar() or 0
    total_sold = db.query(func.sum(Transaction.amount)).filter(
        Transaction.type == "purchase"
    ).scalar() or 0
    total_consumed = db.query(func.sum(Transaction.amount)).filter(
        Transaction.type == "deduction"
    ).scalar() or 0
    # consumed is negative, make positive for display
    if total_consumed < 0:
        total_consumed = abs(total_consumed)

    # Task counts
    tasks_today = db.query(Task).filter(
        Task.status == TaskStatus.COMPLETED,
        Task.created_at >= today_start,
    ).count()
    tasks_week = db.query(Task).filter(
        Task.status == TaskStatus.COMPLETED,
        Task.created_at >= week_start,
    ).count()
    failed = db.query(Task).filter(Task.status == TaskStatus.FAILED).count()

    # Top tools
    top_tools_rows = (
        db.query(Task.tool_type, func.count(Task.id).label("count"))
        .filter(Task.status == TaskStatus.COMPLETED)
        .group_by(Task.tool_type)
        .order_by(func.count(Task.id).desc())
        .limit(5)
        .all()
    )
    top_tools = [{"tool_type": r[0], "count": r[1]} for r in top_tools_rows]

    # Top users by credits
    top_users_rows = (
        db.query(User.id, User.email, User.credits)
        .order_by(User.credits.desc())
        .limit(10)
        .all()
    )
    top_users = [{"id": r[0], "email": r[1], "credits": r[2]} for r in top_users_rows]

    return AdminStatsResponse(
        total_users=total_users,
        new_users_today=new_today,
        new_users_this_week=new_week,
        new_users_this_month=new_month,
        total_revenue=total_revenue,
        total_credits_sold=total_sold,
        total_credits_consumed=total_consumed,
        tasks_today=tasks_today,
        tasks_this_week=tasks_week,
        failed_tasks=failed,
        top_tools=top_tools,
        top_users=top_users,
    )


# --- User List ---

@router.get("/users", response_model=AdminUserListResponse)
def list_users(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    admin: User = Depends(_get_admin_user),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if search:
        s = f"%{search}%"
        query = query.filter(User.email.ilike(s) | User.name.ilike(s))

    total = query.count()

    # Sort
    sort_col = {"created_at": User.created_at, "credits": User.credits, "email": User.email}.get(sort_by, User.created_at)
    if sort_order == "asc":
        query = query.order_by(sort_col.asc())
    else:
        query = query.order_by(sort_col.desc())

    users = query.offset((page - 1) * size).limit(size).all()

    return AdminUserListResponse(
        users=[AdminUserResponse.model_validate(u) for u in users],
        total=total,
        page=page,
        size=size,
    )


# --- User Detail ---

@router.get("/users/{user_id}")
def get_user_detail(
    user_id: str,
    admin: User = Depends(_get_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    recent_tasks = (
        db.query(Task)
        .filter(Task.user_id == user_id)
        .order_by(Task.created_at.desc())
        .limit(20)
        .all()
    )
    recent_txns = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.created_at.desc())
        .limit(20)
        .all()
    )

    return {
        "user": AdminUserResponse.model_validate(user),
        "recent_tasks": [
            {
                "id": t.id, "tool_type": t.tool_type, "status": t.status.value if isinstance(t.status, TaskStatus) else t.status,
                "credits_cost": t.credits_cost, "created_at": t.created_at, "completed_at": t.completed_at,
                "error_message": t.error_message,
            }
            for t in recent_tasks
        ],
        "recent_transactions": [
            {
                "id": t.id, "type": t.type, "amount": t.amount,
                "description": t.description, "created_at": t.created_at,
            }
            for t in recent_txns
        ],
    }


# --- Adjust Credits ---

@router.patch("/users/{user_id}/credits")
def adjust_credits(
    user_id: str,
    req: AdjustCreditsRequest,
    admin: User = Depends(_get_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if req.amount == 0:
        raise HTTPException(status_code=400, detail="Amount cannot be zero")

    if req.amount < 0 and user.credits + req.amount < 0:
        raise HTTPException(status_code=400, detail="Insufficient credits for deduction")

    user.credits += req.amount

    txn = Transaction(
        id=str(uuid.uuid4()),
        user_id=user.id,
        type="deduction" if req.amount < 0 else "purchase",
        amount=req.amount,
        description=f"Admin adjustment by {admin.email}: {req.reason}",
    )
    db.add(txn)
    db.commit()
    db.refresh(user)

    logger.info(
        "Admin %s adjusted credits for %s: %+.1f (reason: %s). New balance: %.1f",
        admin.email, user.email, req.amount, req.reason, user.credits,
    )

    return {"message": f"Credits adjusted by {req.amount:+.1f}", "new_balance": user.credits}


# --- Set User Role ---

@router.patch("/users/{user_id}/role")
def set_user_role(
    user_id: str,
    req: SetUserRoleRequest,
    admin: User = Depends(_get_admin_user),
    db: Session = Depends(get_db),
):
    if req.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role. Must be 'user' or 'admin'")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    old_role = user.role
    user.role = req.role
    db.commit()

    logger.info("Admin %s changed role for %s: %s -> %s", admin.email, user.email, old_role, req.role)

    return {"message": f"Role changed from '{old_role}' to '{req.role}'"}


# --- All Tasks ---

@router.get("/tasks", response_model=AdminTaskListResponse)
def list_tasks(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    tool_type: str | None = Query(None),
    user_id: str | None = Query(None),
    admin: User = Depends(_get_admin_user),
    db: Session = Depends(get_db),
):
    query = db.query(Task).join(User, Task.user_id == User.id)

    if status:
        query = query.filter(Task.status == status)
    if tool_type:
        query = query.filter(Task.tool_type == tool_type)
    if user_id:
        query = query.filter(Task.user_id == user_id)

    total = query.count()
    tasks = query.order_by(Task.created_at.desc()).offset((page - 1) * size).limit(size).all()

    return AdminTaskListResponse(
        tasks=[
            AdminTaskItem(
                id=t.id, user_id=t.user_id, user_email=t.User.email if hasattr(t, "User") else "",
                tool_type=t.tool_type,
                status=t.status.value if isinstance(t.status, TaskStatus) else t.status,
                credits_cost=t.credits_cost, error_message=t.error_message,
                created_at=t.created_at, completed_at=t.completed_at,
            )
            for t in tasks
        ],
        total=total, page=page, size=size,
    )


# --- All Transactions ---

@router.get("/transactions", response_model=AdminTransactionListResponse)
def list_transactions(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    type: str | None = Query(None),
    user_id: str | None = Query(None),
    admin: User = Depends(_get_admin_user),
    db: Session = Depends(get_db),
):
    query = db.query(Transaction).join(User, Transaction.user_id == User.id)

    if type:
        query = query.filter(Transaction.type == type)
    if user_id:
        query = query.filter(Transaction.user_id == user_id)

    total = query.count()
    txns = query.order_by(Transaction.created_at.desc()).offset((page - 1) * size).limit(size).all()

    return AdminTransactionListResponse(
        transactions=[
            AdminTransactionItem(
                id=t.id, user_id=t.user_id, user_email=t.User.email if hasattr(t, "User") else "",
                type=t.type, amount=t.amount, description=t.description,
                created_at=t.created_at,
            )
            for t in txns
        ],
        total=total, page=page, size=size,
    )
