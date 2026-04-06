"""
OperatorBridge — language bridge between factory floor operators and Nemotron.

Operators (Russian / Hebrew speaking) interact in their native language.
OperatorBridge silently translates every message going to/from Nemotron,
so neither side needs to know what language the other speaks.

Flow:
    Operator (ru/he) --[operator_to_nemotron]--> English --> Nemotron
    Nemotron (en)    --[nemotron_to_operator]--> ru/he   --> Operator

Usage:
    bridge = OperatorBridge(poliglo_agent)

    session = bridge.create_session(operator_id="op-42", lang="ru")

    en_text = bridge.operator_to_nemotron(session, "Линия 3 остановлена")
    # → "Line 3 has stopped"

    reply_for_op = bridge.nemotron_to_operator(session, "Line 3 restarting in 30s")
    # → "Линия 3 перезапускается через 30 секунд"
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid

from .agent import PoligloAgent, ChatMessage, SUPPORTED_LANGS


@dataclass
class OperatorSession:
    """Represents one operator's active session."""
    operator_id: str
    lang: str                          # "ru" or "he"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[str] = "factory floor operator communication"


class OperatorBridge:
    """
    Bi-directional language bridge for factory operators ↔ Nemotron.

    All translation is delegated to a PoligloAgent instance.
    The bridge itself is stateless beyond holding a reference to that agent.
    """

    def __init__(self, agent: PoligloAgent):
        self._agent = agent

    # ── Session management ───────────────────────────────────────────────────

    def create_session(self, operator_id: str, lang: str) -> OperatorSession:
        """
        Create a new operator session.

        Args:
            operator_id: Unique operator identifier (e.g. badge number).
            lang:        Operator's language — "ru" or "he".

        Returns:
            OperatorSession ready to pass to operator_to_nemotron / nemotron_to_operator.

        Raises:
            ValueError: If lang is not "ru" or "he".
        """
        if lang not in {"ru", "he"}:
            raise ValueError(
                f"Operator language must be 'ru' or 'he', got '{lang}'. "
                f"Nemotron speaks English natively — use lang='en' sessions are not needed."
            )
        return OperatorSession(operator_id=operator_id, lang=lang)

    # ── Translation helpers ──────────────────────────────────────────────────

    def operator_to_nemotron(self, session: OperatorSession, text: str) -> str:
        """
        Translate operator message → English for Nemotron.

        Args:
            session: Active OperatorSession.
            text:    Raw operator input in session.lang.

        Returns:
            English text ready to send to Nemotron.
            On failure, returns original text so the pipeline isn't blocked.
        """
        msg = ChatMessage(
            sender=session.operator_id,
            text=text,
            target_lang="en",
            source_lang=session.lang,
            context=session.context,
        )
        reply = self._agent.handle_message(msg)
        return reply.translated  # falls back to original on error

    def nemotron_to_operator(self, session: OperatorSession, text: str) -> str:
        """
        Translate Nemotron English response → operator's language.

        Args:
            session: Active OperatorSession.
            text:    English text from Nemotron.

        Returns:
            Text in session.lang for display to operator.
        """
        msg = ChatMessage(
            sender="nemotron",
            text=text,
            target_lang=session.lang,
            source_lang="en",
            context=session.context,
        )
        reply = self._agent.handle_message(msg)
        return reply.translated

    def relay(
        self,
        session: OperatorSession,
        operator_text: str,
        nemotron_fn,
    ) -> str:
        """
        Full round-trip helper: translate in, call Nemotron, translate out.

        Args:
            session:       Active OperatorSession.
            operator_text: Raw operator input.
            nemotron_fn:   Callable(english_text: str) -> str.
                           Should call the actual Nemotron endpoint.

        Returns:
            Nemotron's response translated back to operator's language.

        Example:
            def call_nemotron(text):
                return nemotron.chat(text)

            reply = bridge.relay(session, "Линия 3 стоит", call_nemotron)
            print(reply)  # "Линия 3 перезапускается через 30 секунд"
        """
        en_input = self.operator_to_nemotron(session, operator_text)
        en_output = nemotron_fn(en_input)
        return self.nemotron_to_operator(session, en_output)
