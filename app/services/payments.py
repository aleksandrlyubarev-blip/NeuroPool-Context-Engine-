"""Stripe Payment Link + webhook credit fulfilment (ТЗ п.3, п.6).

Flow: the frontend appends ?client_reference_id=<token> to the Payment Link.
Stripe calls POST /api/stripe/webhook with checkout.session.completed; we
verify the signature, compute credits from the amount and top up the token.
Only stripe_id / credits / hashed_email are stored — never raw email (п.5).
"""

import hashlib
import logging
import time

from app.config import get_settings
from app.services.store import Store

logger = logging.getLogger("polingvo.payments")


def verify_and_parse_event(payload: bytes, sig_header: str) -> dict:
    """Verify the Stripe-Signature header and return the event as a plain dict."""
    import json

    from stripe import WebhookSignature

    settings = get_settings()
    WebhookSignature.verify_header(
        payload.decode("utf-8"), sig_header, settings.stripe_webhook_secret
    )
    return json.loads(payload)


def handle_checkout_completed(event: dict, store: Store) -> dict:
    """Fulfil a completed checkout session: add credits, write ledger event."""
    settings = get_settings()
    session = event["data"]["object"]
    token = session.get("client_reference_id")
    stripe_id = session.get("id", "")
    amount_total = int(session.get("amount_total") or 0)  # agorot
    currency = (session.get("currency") or "ils").lower()
    email = (session.get("customer_details") or {}).get("email") or ""
    hashed_email = hashlib.sha256(email.lower().encode()).hexdigest() if email else ""

    if not token:
        logger.warning("checkout %s without client_reference_id, skipping", stripe_id)
        return {"ok": False, "reason": "no_token"}

    price_agorot = settings.price_ils * 100
    credits = max(1, amount_total // price_agorot) if amount_total else 1

    added = store.add_paid_credits(token, credits, stripe_id, hashed_email)
    if added:
        store.record_ledger_event(
            {
                "type": "revenue",
                "stripe_id": stripe_id,
                "amount": amount_total / 100,
                "currency": currency,
                "credits": credits,
                "created_at": time.time(),
            }
        )
        logger.info(
            "credited %s credits for stripe session %s", credits, stripe_id
        )
    return {"ok": True, "credits": credits if added else 0, "duplicate": not added}
