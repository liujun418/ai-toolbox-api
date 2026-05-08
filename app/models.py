from datetime import datetime, UTC
from enum import Enum

from sqlalchemy import Column, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, relationship

Base = DeclarativeBase


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.USER)
    credits = Column(Float, default=5.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    tasks = relationship("Task", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    tool_type = Column(String(50), nullable=False)  # avatar-generator, background-remover, etc.
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING)
    input_file_url = Column(Text, nullable=True)
    output_file_url = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    credits_cost = Column(Float, default=0)
    replicate_id = Column(String(255), nullable=True)  # Replicate prediction ID
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="tasks")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    type = Column(String(20), nullable=False)  # purchase, spend, bonus, refund
    amount = Column(Float, nullable=False)
    description = Column(String(255), nullable=True)
    stripe_payment_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    user = relationship("User", back_populates="transactions")
