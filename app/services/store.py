"""Credit / ledger / metrics storage (ТЗ п.3, п.5).

Firestore holds ONLY: purchases (stripe_id, credits, created_at, hashed_email),
credits per anonymous token, aggregated metrics and ledger events.
Letter content is never stored anywhere.

MemoryStore is the local-dev/test fallback (no GCP project configured).
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from app.config import get_settings


class Store(ABC):
    @abstractmethod
    def get_credits(self, token: str) -> dict:
        """Returns {"paid": int, "free_available": bool}."""

    @abstractmethod
    def consume_credit(self, token: str) -> Optional[str]:
        """Consume one credit (free first). Returns 'free' | 'paid' | None."""

    @abstractmethod
    def add_paid_credits(
        self, token: str, credits: int, stripe_id: str, hashed_email: str
    ) -> bool:
        """Idempotent by stripe_id. Returns False if this event was already processed."""

    @abstractmethod
    def record_metric(self, data: dict) -> None: ...

    @abstractmethod
    def record_ledger_event(self, data: dict) -> None: ...


class MemoryStore(Store):
    def __init__(self, free_credits: int = 1):
        self._lock = threading.Lock()
        self._free_credits = free_credits
        self._tokens: dict[str, dict] = {}
        self._stripe_ids: set[str] = set()
        self.metrics: list[dict] = []
        self.ledger: list[dict] = []

    def _token(self, token: str) -> dict:
        return self._tokens.setdefault(token, {"paid": 0, "free_used": 0})

    def get_credits(self, token: str) -> dict:
        with self._lock:
            rec = self._token(token)
            return {
                "paid": rec["paid"],
                "free_available": rec["free_used"] < self._free_credits,
            }

    def consume_credit(self, token: str) -> Optional[str]:
        with self._lock:
            rec = self._token(token)
            if rec["free_used"] < self._free_credits:
                rec["free_used"] += 1
                return "free"
            if rec["paid"] > 0:
                rec["paid"] -= 1
                return "paid"
            return None

    def add_paid_credits(
        self, token: str, credits: int, stripe_id: str, hashed_email: str
    ) -> bool:
        with self._lock:
            if stripe_id in self._stripe_ids:
                return False
            self._stripe_ids.add(stripe_id)
            self._token(token)["paid"] += credits
            self.ledger.append(
                {
                    "type": "purchase",
                    "stripe_id": stripe_id,
                    "credits": credits,
                    "hashed_email": hashed_email,
                    "created_at": time.time(),
                }
            )
            return True

    def record_metric(self, data: dict) -> None:
        with self._lock:
            self.metrics.append(data)

    def record_ledger_event(self, data: dict) -> None:
        with self._lock:
            self.ledger.append(data)


class FirestoreStore(Store):
    """Firestore layout:
    credits/{token}        -> {paid, free_used}
    purchases/{stripe_id}  -> {token, credits, hashed_email, created_at}
    metrics/{auto}         -> anonymous per-request aggregates (no PII)
    ledger_events/{auto}   -> revenue evidence events
    """

    def __init__(self, project: str, free_credits: int = 1):
        from google.cloud import firestore  # lazy: not needed for local dev

        self._db = firestore.Client(project=project)
        self._free_credits = free_credits
        self._firestore = firestore

    def get_credits(self, token: str) -> dict:
        snap = self._db.collection("credits").document(token).get()
        data = snap.to_dict() or {}
        return {
            "paid": int(data.get("paid", 0)),
            "free_available": int(data.get("free_used", 0)) < self._free_credits,
        }

    def consume_credit(self, token: str) -> Optional[str]:
        from google.cloud import firestore

        ref = self._db.collection("credits").document(token)

        @firestore.transactional
        def _tx(transaction):
            snap = ref.get(transaction=transaction)
            data = snap.to_dict() or {"paid": 0, "free_used": 0}
            if int(data.get("free_used", 0)) < self._free_credits:
                transaction.set(
                    ref, {"free_used": int(data.get("free_used", 0)) + 1}, merge=True
                )
                return "free"
            if int(data.get("paid", 0)) > 0:
                transaction.set(ref, {"paid": int(data["paid"]) - 1}, merge=True)
                return "paid"
            return None

        return _tx(self._db.transaction())

    def add_paid_credits(
        self, token: str, credits: int, stripe_id: str, hashed_email: str
    ) -> bool:
        purchase_ref = self._db.collection("purchases").document(stripe_id)
        if purchase_ref.get().exists:
            return False
        purchase_ref.set(
            {
                "token": token,
                "credits": credits,
                "hashed_email": hashed_email,
                "created_at": self._firestore.SERVER_TIMESTAMP,
            }
        )
        self._db.collection("credits").document(token).set(
            {"paid": self._firestore.Increment(credits)}, merge=True
        )
        return True

    def record_metric(self, data: dict) -> None:
        self._db.collection("metrics").add(data)

    def record_ledger_event(self, data: dict) -> None:
        self._db.collection("ledger_events").add(data)


_store: Optional[Store] = None


def get_store() -> Store:
    global _store
    if _store is None:
        settings = get_settings()
        if settings.use_firestore and settings.google_cloud_project:
            _store = FirestoreStore(
                settings.google_cloud_project, settings.free_credits
            )
        else:
            _store = MemoryStore(settings.free_credits)
    return _store


def reset_store() -> None:
    """Test helper."""
    global _store
    _store = None
