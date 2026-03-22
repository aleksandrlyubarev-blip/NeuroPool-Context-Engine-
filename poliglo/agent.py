"""
PoligloAgent — Popugai Poliglo translation agent.

Translates between Russian (ru), Hebrew (he), and English (en).
Can be used as a standalone translator or as a chat participant
in multi-agent pipelines (see handle_message / ChatMessage).
"""
import json
from dataclasses import dataclass, field
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from .glossary import GLOSSARY

SUPPORTED_LANGS = {"ru", "he", "en"}
AGENT_NAME = "poliglo"


# ── Message types for inter-agent chat ─────────────────────────────────────

@dataclass
class ChatMessage:
    """
    Envelope used by agents to send text to Poliglo.

    Fields:
        sender:      Name/id of the sending agent.
        text:        Text to translate.
        target_lang: Desired output language ("ru", "he", or "en").
                     Defaults to "en" so any Russian/Hebrew agent can
                     speak to an English-first orchestrator.
        source_lang: Explicit source language or "auto" (default).
        context:     Optional domain hint (e.g. "quality control report").
    """
    sender: str
    text: str
    target_lang: str = "en"
    source_lang: str = "auto"
    context: Optional[str] = None


@dataclass
class ChatReply:
    """
    Poliglo's response to a ChatMessage.

    Fields:
        recipient:   Echoes the sender from the incoming ChatMessage.
        original:    Original text unchanged.
        translated:  Translated text.
        source_lang: Detected or declared source language.
        target_lang: Target language that was used.
        error:       Non-None if translation failed.
    """
    recipient: str
    original: str
    translated: str
    source_lang: str
    target_lang: str
    error: Optional[str] = None

    def ok(self) -> bool:
        return self.error is None


# ── Main agent ──────────────────────────────────────────────────────────────

class PoligloAgent:
    """
    Translation agent for RomeoFlexVision.

    Phase 1 languages: ru ↔ he ↔ en (any pair, auto-detect).

    Two ways to use:
      1. Direct translation:
            agent.translate(text, target_lang="ru")

      2. Agent-to-agent chat:
            reply = agent.handle_message(ChatMessage(sender="nemotron", text="..."))
            print(reply.translated)
    """

    def __init__(self, region: str = "us-west-2"):
        self.translate_client = boto3.client("translate", region_name=region)
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        self.glossary = GLOSSARY
        self.name = AGENT_NAME

    # ── Chat interface (inter-agent) ────────────────────────────────────────

    def handle_message(self, message: ChatMessage) -> ChatReply:
        """
        Receive a ChatMessage from another agent and return a ChatReply.

        This is the main entry point for agent-to-agent communication.
        The caller does not need to know which translation backend is used.

        Example:
            msg = ChatMessage(sender="supervisor", text="Нужна оптическая инспекция", target_lang="he")
            reply = poliglo.handle_message(msg)
            if reply.ok():
                supervisor.receive(reply.translated)
        """
        if message.target_lang not in SUPPORTED_LANGS:
            return ChatReply(
                recipient=message.sender,
                original=message.text,
                translated=message.text,
                source_lang=message.source_lang,
                target_lang=message.target_lang,
                error=f"Unsupported target_lang '{message.target_lang}'. Use: {SUPPORTED_LANGS}",
            )

        try:
            translated = self.translate(
                text=message.text,
                target_lang=message.target_lang,
                source_lang=message.source_lang,
                context=message.context,
            )
            # Detect actual source language if "auto" was used
            detected = self._detect_lang(message.text) if message.source_lang == "auto" else message.source_lang
            return ChatReply(
                recipient=message.sender,
                original=message.text,
                translated=translated,
                source_lang=detected,
                target_lang=message.target_lang,
            )
        except Exception as exc:
            return ChatReply(
                recipient=message.sender,
                original=message.text,
                translated=message.text,
                source_lang=message.source_lang,
                target_lang=message.target_lang,
                error=str(exc),
            )

    # ── Direct translation ──────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        context: Optional[str] = None,
    ) -> str:
        """
        Translate text to target_lang.

        Args:
            text:        Input text (max ~10 000 chars).
            target_lang: "ru", "he", or "en".
            source_lang: "ru", "he", "en", or "auto" (default).
            context:     Optional domain hint for LLM refinement.

        Returns:
            Translated string. RTL preserved for Hebrew.
        """
        if not text or not text.strip():
            return text

        if target_lang not in SUPPORTED_LANGS:
            raise ValueError(f"target_lang must be one of {SUPPORTED_LANGS}")

        if source_lang != "auto" and source_lang == target_lang:
            return text

        try:
            response = self.translate_client.translate_text(
                Text=text,
                SourceLanguageCode=source_lang if source_lang != "auto" else "auto",
                TargetLanguageCode=target_lang,
            )
            translated = response["TranslatedText"]
        except ClientError:
            return self._llm_translate(text, source_lang, target_lang, context)

        # LLM refinement for he/ru targets and long texts
        if target_lang in {"he", "ru"} or len(text) > 500:
            translated = self._llm_technical_refine(translated, source_lang, target_lang, context)

        return translated

    # ── Nemotron tool schemas ───────────────────────────────────────────────

    @staticmethod
    def tool_schemas() -> list:
        """Nemotron-compatible tool schemas (translate_to_russian / hebrew / english)."""
        directions = [
            ("translate_to_russian", "ru", "Translate text to Russian"),
            ("translate_to_hebrew",  "he", "Translate text to Hebrew"),
            ("translate_to_english", "en", "Translate text to English"),
        ]
        schemas = []
        for name, target, description in directions:
            schemas.append({
                "name": name,
                "description": (
                    f"{description}. "
                    "Preserves RomeoFlexVision industrial terminology. "
                    "Supports ru / he / en input with auto-detection."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to translate (max ~10 000 chars)",
                        },
                        "source_lang": {
                            "type": "string",
                            "enum": ["auto", "ru", "he", "en"],
                            "description": "Source language (default: auto-detect)",
                            "default": "auto",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional domain context hint",
                        },
                    },
                    "required": ["text"],
                },
                "_target_lang": target,
            })
        return schemas

    # ── Internal helpers ────────────────────────────────────────────────────

    def _detect_lang(self, text: str) -> str:
        """Best-effort language detection via Amazon Translate (returns detected code)."""
        try:
            response = self.translate_client.translate_text(
                Text=text[:500],
                SourceLanguageCode="auto",
                TargetLanguageCode="en",
            )
            return response.get("SourceLanguageCode", "auto")
        except Exception:
            return "auto"

    def _llm_technical_refine(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[str],
    ) -> str:
        prompt = (
            "Ты — технический переводчик Industry 4.0 для израильских и "
            "русскоязычных заводов.\n"
            "Используй глоссарий ниже. "
            "Сохраняй точность терминов, естественность и RTL.\n\n"
            f"Глоссарий:\n{json.dumps(self.glossary, ensure_ascii=False, indent=2)}\n\n"
            f"Оригинал: {text}\n"
            f"Контекст: {context or 'Нет'}\n\n"
            f"Переведи/улучши на {target.upper()}. "
            "Ответь ТОЛЬКО переводом, без объяснений."
        )
        return self._call_claude(prompt)

    def _llm_translate(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[str],
    ) -> str:
        prompt = (
            f"Переведи текст с {source} на {target}.\n"
            f"Глоссарий RomeoFlexVision:\n"
            f"{json.dumps(self.glossary, ensure_ascii=False)}\n\n"
            f"Текст: {text}\n"
            f"Контекст: {context or 'Нет'}\n\n"
            "Ответь ТОЛЬКО переводом."
        )
        return self._call_claude(prompt)

    def _call_claude(self, prompt: str) -> str:
        try:
            response = self.bedrock_client.converse(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 2000, "temperature": 0.1},
            )
            return response["output"]["message"]["content"][0]["text"].strip()
        except Exception as e:
            return f"[ОШИБКА ПЕРЕВОДА: {e}]"
