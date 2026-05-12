"""Admin API request/response schemas."""

from datetime import datetime

from pydantic import BaseModel


# --- Dashboard Stats ---

class AdminStatsResponse(BaseModel):
    total_users: int
    new_users_today: int
    new_users_this_week: int
    new_users_this_month: int
    total_revenue: float
    total_credits_sold: float
    total_credits_consumed: float
    tasks_today: int
    tasks_this_week: int
    failed_tasks: int
    top_tools: list[dict]  # [{tool_type, count}, ...]
    top_users: list[dict]  # [{id, email, credits}, ...]


# --- User Management ---

class AdminUserResponse(BaseModel):
    id: str
    email: str
    name: str | None
    role: str
    credits: float
    email_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class AdminUserListResponse(BaseModel):
    users: list[AdminUserResponse]
    total: int
    page: int
    size: int


class AdminUserDetailResponse(BaseModel):
    user: AdminUserResponse
    recent_tasks: list[dict]
    recent_transactions: list[dict]


class AdjustCreditsRequest(BaseModel):
    amount: float  # positive=gift, negative=deduct
    reason: str


class SetUserRoleRequest(BaseModel):
    role: str  # "user" or "admin"


# --- Task/Transaction Lists ---

class AdminTaskItem(BaseModel):
    id: str
    user_id: str
    user_email: str
    tool_type: str
    status: str
    credits_cost: float
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class AdminTaskListResponse(BaseModel):
    tasks: list[AdminTaskItem]
    total: int
    page: int
    size: int


class AdminTransactionItem(BaseModel):
    id: str
    user_id: str
    user_email: str
    type: str
    amount: float
    description: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class AdminTransactionListResponse(BaseModel):
    transactions: list[AdminTransactionItem]
    total: int
    page: int
    size: int
