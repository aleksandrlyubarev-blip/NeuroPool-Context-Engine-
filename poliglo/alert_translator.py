"""
AlertTranslator — real-time machine alert translation for RomeoFlexVision.

Machine alerts arrive in English from Robo QC / sensors.
AlertTranslator pushes them instantly to operators in their language (ru/he).

Severity levels map to consistent translated prefixes so operators
immediately grasp urgency without reading the full message.

Usage:
    translator = AlertTranslator(poliglo_agent)

    # Single alert
    result = translator.translate_alert("Defect threshold exceeded on line 3", lang="ru")
    print(result.translated)
    # → "[КРИТИЧНО] Превышен порог дефектов на линии 3"

    # Batch (e.g. dashboard refresh)
    results = translator.translate_batch(alerts, lang="he")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .agent import PoligloAgent, ChatMessage


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"
    OK       = "OK"


# Severity prefix labels per language
_SEVERITY_PREFIX: dict[str, dict[Severity, str]] = {
    "ru": {
        Severity.CRITICAL: "⛔ КРИТИЧНО",
        Severity.WARNING:  "⚠️ ПРЕДУПРЕЖДЕНИЕ",
        Severity.INFO:     "ℹ️ ИНФО",
        Severity.OK:       "✅ ОК",
    },
    "he": {
        Severity.CRITICAL: "⛔ קריטי",
        Severity.WARNING:  "⚠️ אזהרה",
        Severity.INFO:     "ℹ️ מידע",
        Severity.OK:       "✅ תקין",
    },
    "en": {
        Severity.CRITICAL: "⛔ CRITICAL",
        Severity.WARNING:  "⚠️ WARNING",
        Severity.INFO:     "ℹ️ INFO",
        Severity.OK:       "✅ OK",
    },
}

# Keywords used to auto-detect severity from raw English alert text
_SEVERITY_KEYWORDS: list[tuple[Severity, list[str]]] = [
    (Severity.CRITICAL, ["critical", "emergency", "failure", "fault", "exceeded", "stopped", "down"]),
    (Severity.WARNING,  ["warning", "warn", "degraded", "threshold", "overload", "delay", "slow"]),
    (Severity.OK,       ["resolved", "restored", "back online", "cleared", "ok", "normal"]),
]


@dataclass
class AlertResult:
    """
    Result of a single alert translation.

    Fields:
        original:   Raw English alert text.
        translated: Translated text with severity prefix.
        lang:       Target language ("ru" or "he").
        severity:   Detected or provided severity level.
        error:      Non-None if translation failed (translated = original in this case).
    """
    original:   str
    translated: str
    lang:       str
    severity:   Severity
    error:      Optional[str] = None

    def ok(self) -> bool:
        return self.error is None


class AlertTranslator:
    """
    Translates Robo QC / sensor alerts to operator languages in real time.

    Wraps PoligloAgent.handle_message with:
    - Automatic severity detection from alert text
    - Localised severity prefix prepended to every translated alert
    - Batch translation for dashboard / bulk updates
    """

    def __init__(self, agent: PoligloAgent):
        self._agent = agent

    # ── Main API ─────────────────────────────────────────────────────────────

    def translate_alert(
        self,
        alert: str,
        lang: str,
        severity: Optional[Severity] = None,
        source_lang: str = "en",
    ) -> AlertResult:
        """
        Translate a single machine alert.

        Args:
            alert:       Raw alert string (typically English from Robo QC).
            lang:        Target language: "ru", "he", or "en".
            severity:    Explicit severity. If None, auto-detected from text.
            source_lang: Source language, default "en".

        Returns:
            AlertResult with translated text and severity prefix.
        """
        if severity is None:
            severity = self._detect_severity(alert)

        msg = ChatMessage(
            sender="robo_qc",
            text=alert,
            target_lang=lang,
            source_lang=source_lang,
            context="factory floor machine alert, quality control system",
        )
        reply = self._agent.handle_message(msg)

        if reply.error:
            return AlertResult(
                original=alert,
                translated=alert,
                lang=lang,
                severity=severity,
                error=reply.error,
            )

        prefix = _SEVERITY_PREFIX.get(lang, _SEVERITY_PREFIX["en"])[severity]
        translated = f"{prefix}: {reply.translated}"

        return AlertResult(
            original=alert,
            translated=translated,
            lang=lang,
            severity=severity,
        )

    def translate_batch(
        self,
        alerts: list[str],
        lang: str,
        source_lang: str = "en",
    ) -> list[AlertResult]:
        """
        Translate multiple alerts (e.g. dashboard refresh).

        Args:
            alerts:      List of raw alert strings.
            lang:        Target language for all alerts.
            source_lang: Source language, default "en".

        Returns:
            List of AlertResult in the same order as input.
        """
        return [
            self.translate_alert(alert, lang=lang, source_lang=source_lang)
            for alert in alerts
        ]

    # ── Severity detection ───────────────────────────────────────────────────

    @staticmethod
    def _detect_severity(text: str) -> Severity:
        """
        Heuristic severity detection from English alert text.
        Falls back to INFO if no keywords match.
        """
        lower = text.lower()
        for severity, keywords in _SEVERITY_KEYWORDS:
            if any(kw in lower for kw in keywords):
                return severity
        return Severity.INFO
