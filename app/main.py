"""PoLingvo v1.0 — «Что от меня хотят?»

FastAPI service: photo/PDF of an Israeli official letter → structured card in
Russian. See docs/ and ТЗ for the invariants; the load-bearing ones (privacy
п.5, LoFrayer boundary п.6.4, value locking п.4.6) are enforced in code.
"""

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Header, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.pipeline.extract import analyze_letter
from app.privacy.pii import mask_pii
from app.services.ratelimit import RateLimiter
from app.services.store import get_store
from app.services import payments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("polingvo")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
I18N_DIR = BASE_DIR / "i18n"

ALLOWED_MIME = {"image/jpeg": "image/jpeg", "image/png": "image/png",
                "application/pdf": "application/pdf"}

app = FastAPI(title="PoLingvo", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/i18n", StaticFiles(directory=I18N_DIR), name="i18n")

_rate_limiter = RateLimiter(limit=get_settings().rate_limit_per_hour)


def _i18n() -> dict:
    return json.loads((I18N_DIR / "ru.json").read_text(encoding="utf-8"))


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/privacy", response_class=HTMLResponse)
def privacy() -> FileResponse:
    return FileResponse(STATIC_DIR / "privacy.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def api_config() -> dict:
    settings = get_settings()
    return {
        "payment_link": settings.stripe_payment_link_url,
        "price_ils": settings.price_ils,
    }


@app.get("/api/credits")
def api_credits(token: str = Query(min_length=8, max_length=128)) -> dict:
    credits = get_store().get_credits(token)
    return {"token": token, **credits}


@app.post("/api/analyze")
async def api_analyze(
    request: Request,
    file: UploadFile = File(...),
    x_client_token: str = Header(default=""),
):
    settings = get_settings()
    store = get_store()
    request_id = uuid.uuid4().hex[:12]
    strings = _i18n()["errors"]

    token = x_client_token.strip()
    if not (8 <= len(token) <= 128):
        return JSONResponse(
            {"status": "error", "error_ru": strings["no_token"], "request_id": request_id},
            status_code=400,
        )

    mime = (file.content_type or "").lower()
    if mime not in ALLOWED_MIME:
        return JSONResponse(
            {"status": "error", "error_ru": strings["bad_type"], "request_id": request_id},
            status_code=415,
        )

    # In-memory only: the bytes live inside this request and are never written
    # to disk or any bucket (ТЗ п.5 — блокер мержа).
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        return JSONResponse(
            {"status": "error", "error_ru": strings["too_big"], "request_id": request_id},
            status_code=413,
        )

    credits = store.get_credits(token)
    has_credit = credits["paid"] > 0 or credits["free_available"]
    if not has_credit:
        return JSONResponse(
            {
                "status": "payment_required",
                "error_ru": strings["no_credits"],
                "payment_link": settings.stripe_payment_link_url,
                "request_id": request_id,
            },
            status_code=402,
        )
    # Rate limit by IP applies to non-paying usage (ТЗ п.6: 5/час без кредитов).
    if credits["paid"] == 0 and not _rate_limiter.allow(_client_ip(request)):
        return JSONResponse(
            {"status": "error", "error_ru": strings["rate_limited"], "request_id": request_id},
            status_code=429,
        )

    started = time.monotonic()
    try:
        outcome = analyze_letter(data, ALLOWED_MIME[mime])
    except Exception as exc:  # noqa: BLE001 — user gets a safe message, log is masked
        logger.error("request_id=%s pipeline_error=%s", request_id, mask_pii(str(exc)))
        return JSONResponse(
            {"status": "error", "error_ru": strings["internal"], "request_id": request_id},
            status_code=502,
        )
    latency_ms = int((time.monotonic() - started) * 1000)

    # PII-free metrics only (ТЗ п.5): request_id, doc_type, latency, tokens,
    # verdicts, refusal_flag. No sender names, no letter content.
    metric = {
        "request_id": request_id,
        "doc_type": mask_pii(outcome.doc_type_ru or "")[:64] or None,
        "latency_ms": latency_ms,
        "tokens_in": outcome.tokens_in,
        "tokens_out": outcome.tokens_out,
        "verdicts": outcome.verdicts,
        "refusal_flag": outcome.is_refusal,
        "sanitizer_flagged": outcome.sanitizer_flagged,
        "lock_regenerated": outcome.lock_regenerated,
        "mode": "2-pass",
        "created_at": time.time(),
    }
    store.record_metric(metric)
    logger.info("metric %s", json.dumps(metric, ensure_ascii=False))

    if outcome.is_refusal:
        # Refusal never consumes a credit (ТЗ п.4.3).
        return {
            "status": "refusal",
            "refusal": outcome.result.model_dump(),
            "request_id": request_id,
        }

    used = store.consume_credit(token)
    remaining = store.get_credits(token)
    return {
        "status": "card",
        "card": outcome.result.model_dump(),
        "credit_used": used,
        "credits": remaining,
        "request_id": request_id,
    }


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(default="")):
    payload = await request.body()
    try:
        event = payments.verify_and_parse_event(payload, stripe_signature)
    except Exception:  # noqa: BLE001 — invalid signature/payload
        return JSONResponse({"ok": False}, status_code=400)
    if event["type"] == "checkout.session.completed":
        payments.handle_checkout_completed(event, get_store())
    return {"ok": True}
