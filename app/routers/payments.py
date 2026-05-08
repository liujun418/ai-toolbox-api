"""Stripe payment integration for credit top-ups."""

import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Transaction, User
from app.routers.auth import get_current_user
from app.schemas import CreateCheckoutSession, TransactionResponse

router = APIRouter(prefix="/api/payments", tags=["payments"])

# Credit packages: Stripe Price ID -> credits
CREDIT_PACKAGES: dict[str, float] = {
    "50_credits": 50,
    "100_credits": 100,
    "500_credits": 500,
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
        raise HTTPException(status_code=500, detail=str(e))


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

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"].get("user_id")
        price_id = session["metadata"].get("price_id")

        if not user_id or not price_id:
            return {"status": "skipped", "reason": "missing metadata"}

        credits_amount = CREDIT_PACKAGES.get(price_id, 0)
        if credits_amount <= 0:
            return {"status": "skipped", "reason": "unknown price_id"}

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"status": "skipped", "reason": "user not found"}

        # Add credits
        user.credits += credits_amount

        # Record transaction
        tx = Transaction(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type="purchase",
            amount=credits_amount,
            description=f"Purchased {credits_amount} credits via Stripe",
            stripe_payment_id=session.get("payment_intent"),
            created_at=datetime.now(UTC),
        )
        db.add(tx)
        db.commit()

        return {"status": "success", "user_id": user_id, "credits_added": credits_amount}

    return {"status": "ignored"}
