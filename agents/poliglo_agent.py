"""
Backward-compatible re-export.
The agent now lives in the `poliglo` package:
    from poliglo import PoligloAgent, ChatMessage, ChatReply
"""
from poliglo import ChatMessage, ChatReply, PoligloAgent  # noqa: F401

__all__ = ["PoligloAgent", "ChatMessage", "ChatReply"]


# ── Smoke-test (python agents/poliglo_agent.py) ─────────────────────────────
if __name__ == "__main__":
    agent = PoligloAgent()

    print("=== Direct translation ===")

    print("\n→ he → ru:")
    print(agent.translate(
        "התוצאות של בדיקת Robo QC מראות 3 פגמים מסוג predictive maintenance.",
        target_lang="ru", source_lang="he",
    ))

    print("\n→ ru → he:")
    print(agent.translate(
        "Результаты оптической инспекции показывают 3 дефекта, "
        "требующих предиктивного обслуживания.",
        target_lang="he", source_lang="ru",
    ))

    print("\n→ en → ru:")
    print(agent.translate(
        "The optical inspection unit requires predictive maintenance.",
        target_lang="ru",
    ))

    print("\n=== Agent-to-agent chat ===")

    msg = ChatMessage(
        sender="nemotron",
        text="Нужна немедленная калибровка датчика на производственной линии.",
        target_lang="he",
    )
    reply = agent.handle_message(msg)
    print(f"\nFrom: {msg.sender} → Poliglo → {reply.recipient}")
    print(f"  Original   [{reply.source_lang}]: {reply.original}")
    print(f"  Translated [{reply.target_lang}]: {reply.translated}")
    print(f"  OK: {reply.ok()}")

    print("\n=== Tool schemas (Nemotron) ===")
    for schema in PoligloAgent.tool_schemas():
        print(f"  • {schema['name']}")
