import uuid
from datetime import datetime, UTC, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, UserRole, Task, Transaction
from app.schemas import (
    UserResponse, LoginRequest, RegisterRequest, TokenResponse,
    ForgotPasswordRequest, ResetPasswordRequest, ChangePasswordRequest,
    UpdateProfileRequest, TaskListResponse, TransactionListResponse,
    TaskResponse, TransactionResponse,
)
from app.services.auth import (
    hash_password, verify_password, create_access_token, decode_access_token,
)
from app.services.email import send_verification_email, send_password_reset_email

router = APIRouter(prefix="/api/auth", tags=["auth"])

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    user_id = decode_access_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    token = str(uuid.uuid4())
    user = User(
        id=str(uuid.uuid4()),
        email=req.email,
        name=req.name or None,
        hashed_password=hash_password(req.password),
        role=UserRole.USER,
        verification_token=token,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Send verification email (non-blocking — don't fail registration if it fails)
    try:
        send_verification_email(user.email, token)
    except Exception:
        pass  # TODO: log in production

    access_token = create_access_token(user.id)
    return TokenResponse(access_token=access_token, user=UserResponse.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return UserResponse.model_validate(user)


# --- Profile ---

@router.patch("/profile", response_model=UserResponse)
def update_profile(
    req: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if req.name is not None:
        user.name = req.name or None
    if req.email is not None and req.email != user.email:
        existing = db.query(User).filter(User.email == req.email).first()
        if existing:
            raise HTTPException(status_code=409, detail="Email already in use")
        user.email = req.email
        user.email_verified = False
        token = str(uuid.uuid4())
        user.verification_token = token
        try:
            send_verification_email(user.email, token)
        except Exception:
            pass
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


# --- Password ---

@router.patch("/change-password")
def change_password(
    req: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(req.old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    user.hashed_password = hash_password(req.new_password)
    db.commit()
    return {"message": "Password updated"}


@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        # Don't reveal if email exists
        return {"message": "If the email is registered, a reset link has been sent"}
    token = str(uuid.uuid4())
    user.reset_token = token
    user.reset_token_expires = datetime.now(UTC) + timedelta(hours=1)
    db.commit()
    try:
        send_password_reset_email(user.email, token)
    except Exception:
        pass
    return {"message": "If the email is registered, a reset link has been sent"}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        User.reset_token == req.token,
        User.reset_token_expires > datetime.now(UTC),
    ).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    user.hashed_password = hash_password(req.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()
    return {"message": "Password has been reset"}


# --- Email Verification ---

@router.post("/send-verification")
def send_verification(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.email_verified:
        raise HTTPException(status_code=400, detail="Email already verified")
    token = str(uuid.uuid4())
    user.verification_token = token
    db.commit()
    try:
        send_verification_email(user.email, token)
    except Exception:
        pass
    return {"message": "Verification email sent"}


@router.get("/verify-email")
def verify_email(token: str = Query(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    user.email_verified = True
    user.verification_token = None
    db.commit()
    return {"message": "Email verified", "email": user.email}


# --- User Data ---

@router.get("/me/tasks", response_model=TaskListResponse)
def get_my_tasks(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total = db.query(Task).filter(Task.user_id == user.id).count()
    tasks = (
        db.query(Task)
        .filter(Task.user_id == user.id)
        .order_by(Task.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return TaskListResponse(
        tasks=[TaskResponse.model_validate(t) for t in tasks],
        total=total,
    )


@router.get("/me/transactions", response_model=TransactionListResponse)
def get_my_transactions(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total = db.query(Transaction).filter(Transaction.user_id == user.id).count()
    transactions = (
        db.query(Transaction)
        .filter(Transaction.user_id == user.id)
        .order_by(Transaction.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return TransactionListResponse(
        transactions=[TransactionResponse.model_validate(t) for t in transactions],
        total=total,
    )
