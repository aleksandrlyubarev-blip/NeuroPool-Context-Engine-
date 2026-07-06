"""API contract tests (ТЗ п.6) with the pipeline mocked out."""

import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.pipeline.extract import PipelineOutcome
from app.pipeline.schemas import Card, Refusal
from app.services.ratelimit import RateLimiter
from app.services import store as store_module


@pytest.fixture()
def client(monkeypatch):
    store_module.reset_store()
    monkeypatch.setattr(main, "_rate_limiter", RateLimiter(limit=5))
    with TestClient(main.app) as c:
        yield c
    store_module.reset_store()


def _card_outcome() -> PipelineOutcome:
    return PipelineOutcome(
        result=Card(
            sender_ru="Ирия Хайфы", doc_type_ru="требование об оплате",
            demand_summary_ru="Оплатите долг.", amount_value=150.0,
            amount_currency="ILS", deadline="2026-08-15",
            recommended_actions_ru=["Оплатите на сайте"],
        ),
        verdicts={"sender": "confirm", "amount": "confirm", "deadline": "confirm"},
        doc_type_ru="требование об оплате", tokens_in=100, tokens_out=50,
    )


def _refusal_outcome() -> PipelineOutcome:
    return PipelineOutcome(
        result=Refusal(reason_code="mismatch_amount", reason_ru="Не удалось..."),
        verdicts={"sender": "confirm", "amount": "mismatch", "deadline": "confirm"},
        doc_type_ru=None,
    )


TOKEN = {"X-Client-Token": "test-token-123"}
FILE = {"file": ("letter.jpg", b"\xff\xd8fake", "image/jpeg")}


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_index_and_privacy_served(client):
    assert client.get("/").status_code == 200
    r = client.get("/privacy")
    assert r.status_code == 200 and "не сохраняются" in r.text


def test_analyze_requires_token(client):
    resp = client.post("/api/analyze", files=FILE)
    assert resp.status_code == 400


def test_analyze_rejects_bad_mime(client):
    resp = client.post(
        "/api/analyze",
        files={"file": ("x.gif", b"GIF89a", "image/gif")},
        headers=TOKEN,
    )
    assert resp.status_code == 415


def test_analyze_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_letter", lambda *a, **k: _card_outcome())
    big = b"x" * (11 * 1024 * 1024)
    resp = client.post(
        "/api/analyze", files={"file": ("big.jpg", big, "image/jpeg")}, headers=TOKEN
    )
    assert resp.status_code == 413


def test_analyze_card_consumes_free_credit(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_letter", lambda *a, **k: _card_outcome())
    resp = client.post("/api/analyze", files=FILE, headers=TOKEN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "card"
    assert body["credit_used"] == "free"
    assert body["card"]["amount_value"] == 150.0
    # free credit is now used up → next call is payment_required
    resp2 = client.post("/api/analyze", files=FILE, headers=TOKEN)
    assert resp2.status_code == 402


def test_refusal_does_not_consume_credit(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_letter", lambda *a, **k: _refusal_outcome())
    resp = client.post("/api/analyze", files=FILE, headers=TOKEN)
    assert resp.status_code == 200
    assert resp.json()["status"] == "refusal"
    credits = client.get("/api/credits", params={"token": "test-token-123"}).json()
    assert credits["free_available"] is True


def test_rate_limit_applies_to_free_tier(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_letter", lambda *a, **k: _refusal_outcome())
    for _ in range(5):
        assert client.post("/api/analyze", files=FILE, headers=TOKEN).status_code == 200
    assert client.post("/api/analyze", files=FILE, headers=TOKEN).status_code == 429


def test_credits_endpoint(client):
    body = client.get("/api/credits", params={"token": "abcdefgh"}).json()
    assert body == {"token": "abcdefgh", "paid": 0, "free_available": True}


def _stripe_signed(payload: bytes, secret: str) -> str:
    ts = str(int(time.time()))
    signed = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256)
    return f"t={ts},v1={signed.hexdigest()}"


def test_stripe_webhook_credits_and_ledger(client, monkeypatch):
    secret = "whsec_test"
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", secret)
    event = {
        "id": "evt_test_1",
        "object": "event",
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": "cs_test_1", "client_reference_id": "test-token-123",
            "amount_total": 1000, "currency": "ils",
            "customer_details": {"email": "buyer@example.com"},
        }},
    }
    payload = json.dumps(event).encode()
    resp = client.post(
        "/api/stripe/webhook", content=payload,
        headers={"Stripe-Signature": _stripe_signed(payload, secret)},
    )
    assert resp.status_code == 200 and resp.json() == {"ok": True}

    credits = client.get("/api/credits", params={"token": "test-token-123"}).json()
    assert credits["paid"] == 1

    mem = store_module.get_store()
    revenue = [e for e in mem.ledger if e["type"] == "revenue"]
    purchases = [e for e in mem.ledger if e["type"] == "purchase"]
    assert revenue[0]["amount"] == 10.0
    # raw email never stored — only its hash (ТЗ п.5)
    assert purchases[0]["hashed_email"] == hashlib.sha256(b"buyer@example.com").hexdigest()

    # idempotency: same event again does not double-credit
    resp2 = client.post(
        "/api/stripe/webhook", content=payload,
        headers={"Stripe-Signature": _stripe_signed(payload, secret)},
    )
    assert resp2.status_code == 200
    assert client.get("/api/credits", params={"token": "test-token-123"}).json()["paid"] == 1


def test_stripe_webhook_rejects_bad_signature(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    resp = client.post(
        "/api/stripe/webhook", content=b"{}",
        headers={"Stripe-Signature": "t=1,v1=deadbeef"},
    )
    assert resp.status_code == 400
