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
"""

from .agent import ChatMessage, ChatReply, PoligloAgent
from .glossary import GLOSSARY

__all__ = ["PoligloAgent", "ChatMessage", "ChatReply", "GLOSSARY"]
