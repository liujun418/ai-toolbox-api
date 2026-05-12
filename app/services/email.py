"""Email service using Resend API."""

import httpx

from app.config import settings


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.RESEND_API_KEY}",
        "Content-Type": "application/json",
    }


def send_verification_email(to: str, token: str) -> None:
    """Send email verification link. Synchronous — caller should await in FastAPI context."""
    url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            "https://api.resend.com/emails",
            headers=_headers(),
            json={
                "from": "AI ToolBox <onboarding@resend.com>",
                "to": [to],
                "subject": "Verify your email — AI ToolBox",
                "html": (
                    f"<p>Click the link below to verify your email:</p>"
                    f'<p><a href="{url}">Verify Email</a></p>'
                    f"<p>This link expires in 24 hours.</p>"
                ),
            },
        )
        resp.raise_for_status()


def send_password_reset_email(to: str, token: str) -> None:
    """Send password reset link. Synchronous — caller should await in FastAPI context."""
    url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            "https://api.resend.com/emails",
            headers=_headers(),
            json={
                "from": "AI ToolBox <onboarding@resend.com>",
                "to": [to],
                "subject": "Reset your password — AI ToolBox",
                "html": (
                    f"<p>Click the link below to reset your password:</p>"
                    f'<p><a href="{url}">Reset Password</a></p>'
                    f"<p>This link expires in 1 hour. If you did not request this, ignore this email.</p>"
                ),
            },
        )
        resp.raise_for_status()
