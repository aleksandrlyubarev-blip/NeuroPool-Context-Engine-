"""
poliglo — Popugai Poliglo translation agent for RomeoFlexVision.

Phase 1: ru ↔ he ↔ en (any pair, auto-detect source).

Quick start:
    from poliglo import PoligloAgent, ChatMessage

    agent = PoligloAgent()

    # Direct translation
    text_he = agent.translate("оптическая инспекция", target_lang="he", source_lang="ru")

    # Agent-to-agent chat
    reply = agent.handle_message(ChatMessage(sender="supervisor", text="...", target_lang="en"))
    print(reply.translated)

    # Operator ↔ Nemotron bridge
    from poliglo import OperatorBridge
    bridge = OperatorBridge(agent)
    session = bridge.create_session(operator_id="op-42", lang="ru")
    en_text = bridge.operator_to_nemotron(session, "Линия 3 остановлена")
"""

from .agent import ChatMessage, ChatReply, PoligloAgent
from .alert_translator import AlertResult, AlertTranslator, Severity
from .glossary import GLOSSARY
from .glossary_extractor import GlossaryExtractor, GlossarySuggestion
from .operator_bridge import OperatorBridge, OperatorSession

__all__ = [
    "PoligloAgent", "ChatMessage", "ChatReply", "GLOSSARY",
    "OperatorBridge", "OperatorSession",
    "AlertTranslator", "AlertResult", "Severity",
    "GlossaryExtractor", "GlossarySuggestion",
]
