"""Stripe payment integration for credit top-ups."""

import json
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Transaction, User
from app.routers.auth import get_current_user
from app.schemas import CreateCheckoutSession, TransactionResponse

router = APIRouter(prefix="/api/payments", tags=["payments"])

# One-time credit packages: Stripe Price ID -> credits
CREDIT_PACKAGES: dict[str, float] = {
    "small_credits": 10,
    "standard_credits": 50,
    "value_credits": 200,
}

# Monthly subscription packages: Stripe Price ID -> credits/month
SUBSCRIPTION_PACKAGES: dict[str, float] = {
    "basic_monthly": 40,
    "pro_monthly": 120,
}


def _get_stripe_client():
    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY
    return stripe


@router.post("/create-checkout-session")
async def create_checkout_session(
    body: CreateCheckoutSession,
    user: User = Depends(get_current_user),
):
    """Create a Stripe Checkout session for credit purchase."""
    stripe = _get_stripe_client()

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            customer_email=user.email,
            metadata={
                "user_id": user.id,
                "price_id": body.price_id,
            },
            line_items=[{"price": body.price_id, "quantity": 1}],
            success_url=f"{settings.FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/payment/cancel",
            payment_intent_data={
                "metadata": {
                    "user_id": user.id,
                    "price_id": body.price_id,
                },
            },
        )
        return {"checkout_url": session.url}
    except Exception as e:
        logger.exception("Stripe checkout session creation failed")
        raise HTTPException(status_code=500, detail="Payment service temporarily unavailable")


@router.post("/create-subscription-session")
async def create_subscription_session(
    body: CreateCheckoutSession,
    user: User = Depends(get_current_user),
):
    """Create a Stripe Checkout session for monthly subscription."""
    stripe = _get_stripe_client()

    if body.price_id not in SUBSCRIPTION_PACKAGES:
        raise HTTPException(status_code=400, detail=f"Unknown subscription: {body.price_id}")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=user.email,
            metadata={
                "user_id": user.id,
                "price_id": body.price_id,
            },
            line_items=[{"price": body.price_id, "quantity": 1}],
            success_url=f"{settings.FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/payment/cancel",
            subscription_data={
                "metadata": {
                    "user_id": user.id,
                    "price_id": body.price_id,
                },
            },
        )
        return {"checkout_url": session.url}
    except Exception as e:
        logger.exception("Stripe subscription session creation failed")
        raise HTTPException(status_code=500, detail="Payment service temporarily unavailable")


def _add_credits_to_user(db: Session, user_id: str, credits_amount: float, price_id: str, payment_intent: str | None = None) -> dict:
    """Helper: add credits and record transaction. Idempotent — skips duplicate stripe_payment_id."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"status": "skipped", "reason": "user not found"}

    if payment_intent:
        existing = db.query(Transaction).filter(
            Transaction.stripe_payment_id == payment_intent
        ).first()
        if existing:
            return {"status": "skipped", "reason": "duplicate webhook"}

    user.credits += credits_amount
    is_sub = price_id in SUBSCRIPTION_PACKAGES
    tx = Transaction(
        id=str(uuid.uuid4()),
        user_id=user_id,
        type="purchase",
        amount=credits_amount,
        description=f"{'Subscription' if is_sub else 'Purchased'} {credits_amount} credits via Stripe",
        stripe_payment_id=payment_intent,
        created_at=datetime.now(UTC),
    )
    db.add(tx)

    if is_sub:
        from datetime import timedelta
        user.subscription_tier = price_id
        user.subscription_end_date = datetime.now(UTC) + timedelta(days=32)

    db.commit()
    return {"status": "success", "user_id": user_id, "credits_added": credits_amount}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events to credit user accounts."""
    import stripe

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except (ValueError, stripe.SignatureVerificationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid signature: {e}")

    # One-time purchase completed
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        if session.get("mode") == "subscription":
            return {"status": "ignored", "reason": "subscription handled by invoice.paid"}
        user_id = session["metadata"].get("user_id")
        price_id = session["metadata"].get("price_id")
        if not user_id or not price_id:
            return {"status": "skipped", "reason": "missing metadata"}
        credits_amount = CREDIT_PACKAGES.get(price_id, 0)
        if credits_amount <= 0:
            return {"status": "skipped", "reason": "unknown price_id"}
        return _add_credits_to_user(db, user_id, credits_amount, price_id, session.get("payment_intent"))

    # Subscription payment (initial + recurring)
    if event["type"] == "invoice.paid":
        invoice = event["data"]["object"]
        subscription_id = invoice.get("subscription")
        if not subscription_id:
            return {"status": "skipped", "reason": "no subscription id"}
        # Get subscription metadata via Stripe API
        stripe_client = _get_stripe_client()
        try:
            sub = stripe_client.Subscription.retrieve(subscription_id)
        except Exception:
            return {"status": "skipped", "reason": "subscription not found"}
        user_id = sub["metadata"].get("user_id")
        price_id = sub["metadata"].get("price_id")
        if not user_id or not price_id:
            return {"status": "skipped", "reason": "missing metadata"}
        credits_amount = SUBSCRIPTION_PACKAGES.get(price_id, 0)
        if credits_amount <= 0:
            return {"status": "skipped", "reason": "unknown price_id"}
        return _add_credits_to_user(db, user_id, credits_amount, price_id, invoice.get("id"))

    return {"status": "ignored"}
