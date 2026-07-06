"""Application settings, read from environment variables.

All knobs live here so the rest of the code never touches os.environ directly.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    google_cloud_project: str
    stripe_webhook_secret: str
    stripe_payment_link_url: str
    price_ils: int
    free_credits: int
    max_upload_mb: int
    rate_limit_per_hour: int
    confidence_threshold: float
    use_firestore: bool


def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        # NOTE (ТЗ п.3): модель задаётся через env; дефолт — актуальный Flash-алиас.
        # Апгрейд на новую модель = смена одной env-переменной.
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
        google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET", ""),
        stripe_payment_link_url=os.environ.get("STRIPE_PAYMENT_LINK_URL", ""),
        price_ils=int(os.environ.get("PRICE_ILS", "10")),
        free_credits=int(os.environ.get("FREE_CREDITS", "1")),
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "10")),
        rate_limit_per_hour=int(os.environ.get("RATE_LIMIT_PER_HOUR", "5")),
        confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")),
        use_firestore=os.environ.get("USE_FIRESTORE", "").lower() in ("1", "true", "yes"),
    )
