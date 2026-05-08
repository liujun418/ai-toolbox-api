from datetime import datetime

from pydantic import BaseModel, Field


# --- Schemas for Tasks ---

class TaskCreate(BaseModel):
    tool_type: str
    """avatar-generator, background-remover, watermark-remover, photo-restorer, pdf-to-word"""


class TaskResponse(BaseModel):
    id: str
    user_id: str
    tool_type: str
    status: str
    output_file_url: str | None = None
    error_message: str | None = None
    credits_cost: float = 0
    created_at: datetime
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


# --- Schemas for Users ---

class UserResponse(BaseModel):
    id: str
    email: str
    name: str | None = None
    credits: float
    created_at: datetime

    class Config:
        from_attributes = True


# --- Schemas for Transactions ---

class TransactionResponse(BaseModel):
    id: str
    type: str
    amount: float
    description: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Schemas for Payments ---

class CreateCheckoutSession(BaseModel):
    price_id: str = Field(..., description="Stripe Price ID")


# --- Schemas for Auth ---

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    user: UserResponse


# --- Pricing constants ---

CREDIT_COSTS: dict[str, float] = {
    "avatar-generator": 5,
    "background-remover": 2,
    "watermark-remover": 3,
    "photo-restorer": 5,
    "pdf-to-word": 1,  # +1 if >10 pages (handled dynamically)
}
