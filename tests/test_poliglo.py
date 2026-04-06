"""
Unit tests for the poliglo package.

All AWS calls (boto3) are mocked — no real credentials required.

Test coverage:
  - PoligloAgent.translate / handle_message
  - OperatorBridge (create_session, operator_to_nemotron, nemotron_to_operator, relay)
  - AlertTranslator (translate_alert, translate_batch, severity detection)
  - GlossaryExtractor (extract, _parse_response, _chunk, apply_to_glossary)
"""

import json
import textwrap
import uuid
from unittest.mock import MagicMock, patch, mock_open

import pytest

from poliglo.agent import ChatMessage, ChatReply, PoligloAgent, SUPPORTED_LANGS
from poliglo.alert_translator import AlertTranslator, Severity, _SEVERITY_PREFIX
from poliglo.glossary_extractor import GlossaryExtractor, GlossarySuggestion
from poliglo.operator_bridge import OperatorBridge, OperatorSession


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _translate_response(translated: str, source: str = "ru") -> dict:
    """Helper: fake Amazon Translate response."""
    return {"TranslatedText": translated, "SourceLanguageCode": source}


def _mock_agent(translate_side_effect=None, bedrock_text="refined") -> PoligloAgent:
    """
    Return a PoligloAgent with both AWS clients mocked.

    translate_side_effect: what translate_client.translate_text returns.
                           Defaults to echoing TargetLanguageCode + text.
    """
    with patch("boto3.client"):
        agent = PoligloAgent()

    def default_translate(**kwargs):
        return _translate_response(
            f"[{kwargs['TargetLanguageCode']}] {kwargs['Text']}",
            source=kwargs.get("SourceLanguageCode", "auto"),
        )

    agent.translate_client = MagicMock()
    agent.translate_client.translate_text.side_effect = (
        translate_side_effect or default_translate
    )
    agent.bedrock_client = MagicMock()
    agent.bedrock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": bedrock_text}]}}
    }
    return agent


# ── PoligloAgent tests ────────────────────────────────────────────────────────

class TestPoligloAgent:

    def test_translate_same_lang_returns_original(self):
        agent = _mock_agent()
        result = agent.translate("hello", target_lang="en", source_lang="en")
        assert result == "hello"
        agent.translate_client.translate_text.assert_not_called()

    def test_translate_empty_returns_empty(self):
        agent = _mock_agent()
        assert agent.translate("", target_lang="ru") == ""
        assert agent.translate("   ", target_lang="ru") == "   "

    def test_translate_invalid_target_raises(self):
        agent = _mock_agent()
        with pytest.raises(ValueError, match="target_lang"):
            agent.translate("text", target_lang="fr")

    def test_translate_calls_aws(self):
        agent = _mock_agent()
        result = agent.translate("оптическая инспекция", target_lang="en", source_lang="ru")
        agent.translate_client.translate_text.assert_called_once()
        # Default mock prefixes target lang
        assert "[en]" in result

    def test_translate_aws_failure_falls_back_to_llm(self):
        from botocore.exceptions import ClientError
        agent = _mock_agent()
        agent.translate_client.translate_text.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailableException", "Message": "down"}}, "TranslateText"
        )
        agent.bedrock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "fallback translation"}]}}
        }
        result = agent.translate("test", target_lang="en", source_lang="ru")
        assert result == "fallback translation"

    def test_handle_message_returns_chat_reply(self):
        agent = _mock_agent()
        msg = ChatMessage(sender="nemotron", text="Линия 3 стоит", target_lang="en", source_lang="ru")
        reply = agent.handle_message(msg)
        assert isinstance(reply, ChatReply)
        assert reply.recipient == "nemotron"
        assert reply.ok()

    def test_handle_message_unsupported_lang_returns_error(self):
        agent = _mock_agent()
        msg = ChatMessage(sender="bot", text="hola", target_lang="es")
        reply = agent.handle_message(msg)
        assert not reply.ok()
        assert "es" in reply.error
        assert reply.translated == "hola"

    def test_tool_schemas_returns_three_directions(self):
        schemas = PoligloAgent.tool_schemas()
        assert len(schemas) == 3
        targets = {s["_target_lang"] for s in schemas}
        assert targets == {"ru", "he", "en"}
        for s in schemas:
            assert "text" in s["parameters"]["properties"]


# ── OperatorBridge tests ──────────────────────────────────────────────────────

class TestOperatorBridge:

    def test_create_session_ru(self):
        bridge = OperatorBridge(_mock_agent())
        session = bridge.create_session("op-1", "ru")
        assert session.operator_id == "op-1"
        assert session.lang == "ru"
        assert session.session_id  # UUID assigned

    def test_create_session_he(self):
        bridge = OperatorBridge(_mock_agent())
        session = bridge.create_session("op-2", "he")
        assert session.lang == "he"

    def test_create_session_invalid_lang_raises(self):
        bridge = OperatorBridge(_mock_agent())
        with pytest.raises(ValueError, match="ru.*he"):
            bridge.create_session("op-3", "en")

    def test_operator_to_nemotron_translates_to_english(self):
        agent = _mock_agent()
        bridge = OperatorBridge(agent)
        session = bridge.create_session("op-1", "ru")
        result = bridge.operator_to_nemotron(session, "Линия 3 стоит")
        # Mock prefixes target lang
        assert "[en]" in result
        call_kwargs = agent.translate_client.translate_text.call_args[1]
        assert call_kwargs["TargetLanguageCode"] == "en"

    def test_nemotron_to_operator_translates_to_ru(self):
        agent = _mock_agent()
        bridge = OperatorBridge(agent)
        session = bridge.create_session("op-1", "ru")
        bridge.nemotron_to_operator(session, "Line 3 restarting")
        # Verify AWS translate was called targeting the operator's language
        call_kwargs = agent.translate_client.translate_text.call_args[1]
        assert call_kwargs["TargetLanguageCode"] == "ru"

    def test_nemotron_to_operator_translates_to_he(self):
        agent = _mock_agent()
        bridge = OperatorBridge(agent)
        session = bridge.create_session("op-99", "he")
        bridge.nemotron_to_operator(session, "All systems nominal")
        call_kwargs = agent.translate_client.translate_text.call_args[1]
        assert call_kwargs["TargetLanguageCode"] == "he"

    def test_relay_full_roundtrip(self):
        agent = _mock_agent()
        bridge = OperatorBridge(agent)
        session = bridge.create_session("op-1", "ru")

        nemotron_fn = MagicMock(return_value="Line 3 is back online")
        bridge.relay(session, "Линия 3 стоит", nemotron_fn)

        # nemotron_fn must be called once with the English translation
        nemotron_fn.assert_called_once()
        en_sent = nemotron_fn.call_args[0][0]
        assert "[en]" in en_sent

        # Final AWS translate call must target the operator's language
        calls = agent.translate_client.translate_text.call_args_list
        last_target = calls[-1][1]["TargetLanguageCode"]
        assert last_target == "ru"

    def test_operator_to_nemotron_falls_back_on_error(self):
        """On translation failure the original text is returned (pipeline not blocked)."""
        from botocore.exceptions import ClientError
        agent = _mock_agent()
        agent.translate_client.translate_text.side_effect = ClientError(
            {"Error": {"Code": "Throttling", "Message": "slow down"}}, "TranslateText"
        )
        agent.bedrock_client.converse.side_effect = Exception("bedrock down")
        bridge = OperatorBridge(agent)
        session = bridge.create_session("op-1", "ru")
        original = "Линия 3 стоит"
        result = bridge.operator_to_nemotron(session, original)
        # Falls back to original because handle_message catches exceptions
        assert original in result or "[ОШИБКА" in result


# ── AlertTranslator tests ─────────────────────────────────────────────────────

class TestAlertTranslator:

    def test_translate_alert_critical_ru(self):
        agent = _mock_agent()
        tr = AlertTranslator(agent)
        result = tr.translate_alert("Defect threshold exceeded on line 3", lang="ru")
        assert result.ok()
        assert result.severity == Severity.CRITICAL
        assert "КРИТИЧНО" in result.translated
        assert "⛔" in result.translated

    def test_translate_alert_warning_he(self):
        agent = _mock_agent()
        tr = AlertTranslator(agent)
        result = tr.translate_alert("Warning: sensor delay detected", lang="he")
        assert result.ok()
        assert result.severity == Severity.WARNING
        assert "אזהרה" in result.translated

    def test_translate_alert_ok_severity(self):
        agent = _mock_agent()
        tr = AlertTranslator(agent)
        result = tr.translate_alert("System restored to normal", lang="ru")
        assert result.severity == Severity.OK
        assert "ОК" in result.translated

    def test_translate_alert_info_fallback(self):
        agent = _mock_agent()
        tr = AlertTranslator(agent)
        result = tr.translate_alert("Scheduled maintenance window", lang="ru")
        assert result.severity == Severity.INFO
        assert "ИНФО" in result.translated

    def test_translate_alert_explicit_severity_overrides_detection(self):
        agent = _mock_agent()
        tr = AlertTranslator(agent)
        # Text says "ok" but we force CRITICAL
        result = tr.translate_alert("System restored", lang="ru", severity=Severity.CRITICAL)
        assert result.severity == Severity.CRITICAL
        assert "КРИТИЧНО" in result.translated

    def test_translate_alert_error_returns_original(self):
        agent = _mock_agent()
        agent.translate_client.translate_text.side_effect = Exception("AWS gone")
        agent.bedrock_client.converse.side_effect = Exception("Bedrock gone")
        tr = AlertTranslator(agent)
        original = "Critical fault on motor"
        result = tr.translate_alert(original, lang="ru")
        assert not result.ok()
        assert result.translated == original
        assert result.error is not None

    def test_translate_batch_returns_same_length(self):
        agent = _mock_agent()
        tr = AlertTranslator(agent)
        alerts = [
            "Defect threshold exceeded",
            "Sensor delay detected",
            "System restored",
        ]
        results = tr.translate_batch(alerts, lang="ru")
        assert len(results) == 3
        assert all(isinstance(r, type(results[0])) for r in results)

    def test_translate_batch_he(self):
        agent = _mock_agent()
        tr = AlertTranslator(agent)
        results = tr.translate_batch(["Line stopped", "Line restored"], lang="he")
        assert results[0].lang == "he"
        assert results[1].lang == "he"

    @pytest.mark.parametrize("text,expected_severity", [
        ("Machine failure on conveyor", Severity.CRITICAL),
        ("Emergency stop triggered",   Severity.CRITICAL),
        ("Defect rate exceeded limit", Severity.CRITICAL),
        ("Warning: high temperature",  Severity.WARNING),
        ("Slight delay in throughput", Severity.WARNING),
        ("System back online",         Severity.OK),
        ("All clear",                  Severity.INFO),   # no keyword → INFO
    ])
    def test_detect_severity(self, text, expected_severity):
        assert AlertTranslator._detect_severity(text) == expected_severity

    def test_severity_prefixes_defined_for_ru_he_en(self):
        for lang in ("ru", "he", "en"):
            assert lang in _SEVERITY_PREFIX
            for sev in Severity:
                assert sev in _SEVERITY_PREFIX[lang]


# ── GlossaryExtractor tests ───────────────────────────────────────────────────

_SAMPLE_JSON = json.dumps([
    {
        "term_en": "throughput",
        "ru": "пропускная способность",
        "he": "תפוקה",
        "context": "The throughput of the line was measured.",
        "confidence": "high",
    },
    {
        "term_en": "downtime",
        "ru": "простой",
        "he": "זמן השבתה",
        "context": "Unplanned downtime costs were significant.",
        "confidence": "medium",
    },
])


class TestGlossaryExtractor:

    def test_extract_returns_suggestions(self):
        agent = _mock_agent(bedrock_text=_SAMPLE_JSON)
        # translate_text returns JSON so LLM path is hit
        from botocore.exceptions import ClientError
        agent.translate_client.translate_text.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailableException", "Message": "n/a"}}, "TranslateText"
        )
        extractor = GlossaryExtractor(agent, glossary={})
        suggestions = extractor.extract("The throughput and downtime of line 3 were measured.")
        assert len(suggestions) == 2
        terms = {s.term_en for s in suggestions}
        assert "throughput" in terms
        assert "downtime" in terms

    def test_extract_skips_existing_glossary_terms(self):
        agent = _mock_agent(bedrock_text=_SAMPLE_JSON)
        from botocore.exceptions import ClientError
        agent.translate_client.translate_text.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailableException", "Message": "n/a"}}, "TranslateText"
        )
        # throughput is already in the glossary
        glossary = {"throughput": {"ru": "пропускная способность", "he": "תפוקה", "en": "throughput"}}
        extractor = GlossaryExtractor(agent, glossary=glossary)
        suggestions = extractor.extract("throughput and downtime")
        throughput_sug = next((s for s in suggestions if s.term_en == "throughput"), None)
        assert throughput_sug is not None
        assert throughput_sug.already_in_glossary is True

    def test_extract_sorted_by_confidence_desc(self):
        agent = _mock_agent(bedrock_text=_SAMPLE_JSON)
        from botocore.exceptions import ClientError
        agent.translate_client.translate_text.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailableException", "Message": "n/a"}}, "TranslateText"
        )
        extractor = GlossaryExtractor(agent, glossary={})
        suggestions = extractor.extract("throughput downtime")
        # high before medium
        assert suggestions[0].confidence == "high"

    def test_parse_response_valid_json(self):
        extractor = GlossaryExtractor(_mock_agent(), glossary={})
        results = extractor._parse_response(_SAMPLE_JSON)
        assert len(results) == 2
        assert results[0].term_en == "throughput"
        assert results[0].translations["ru"] == "пропускная способность"

    def test_parse_response_strips_markdown_fence(self):
        extractor = GlossaryExtractor(_mock_agent(), glossary={})
        fenced = f"```json\n{_SAMPLE_JSON}\n```"
        results = extractor._parse_response(fenced)
        assert len(results) == 2

    def test_parse_response_invalid_json_returns_empty(self):
        extractor = GlossaryExtractor(_mock_agent(), glossary={})
        results = extractor._parse_response("not json at all")
        assert results == []

    def test_chunk_short_text_single_chunk(self):
        chunks = GlossaryExtractor._chunk("short text")
        assert chunks == ["short text"]

    def test_chunk_long_text_multiple_chunks(self):
        long_text = "word " * 1000   # ~5000 chars
        chunks = GlossaryExtractor._chunk(long_text)
        assert len(chunks) > 1
        # Each chunk should be at most _MAX_CHUNK chars
        from poliglo.glossary_extractor import _MAX_CHUNK
        for chunk in chunks:
            assert len(chunk) <= _MAX_CHUNK

    def test_chunk_overlap(self):
        """Adjacent chunks share some content (overlap)."""
        from poliglo.glossary_extractor import _MAX_CHUNK, _OVERLAP
        long_text = "X" * (_MAX_CHUNK + _OVERLAP + 100)
        chunks = GlossaryExtractor._chunk(long_text)
        assert len(chunks) >= 2
        # End of first chunk should appear at start of second
        overlap_from_first = chunks[0][-_OVERLAP:]
        assert chunks[1].startswith(overlap_from_first)

    def test_apply_to_glossary_writes_entries(self, tmp_path):
        glossary_py = tmp_path / "glossary.py"
        glossary_py.write_text(
            textwrap.dedent("""\
                GLOSSARY: dict = {
                    "quality control": {"ru": "контроль качества", "he": "בקרת איכות", "en": "quality control"},
                }
            """),
            encoding="utf-8",
        )
        extractor = GlossaryExtractor(_mock_agent(), glossary={})
        suggestions = [
            GlossarySuggestion(
                term_en="throughput",
                translations={"ru": "пропускная способность", "he": "תפוקה"},
                confidence="high",
            )
        ]
        n = extractor.apply_to_glossary(suggestions, glossary_path=glossary_py)
        assert n == 1
        content = glossary_py.read_text(encoding="utf-8")
        assert '"throughput"' in content
        assert "пропускная способность" in content
        assert "תפוקה" in content

    def test_apply_to_glossary_skips_already_in_glossary(self, tmp_path):
        glossary_py = tmp_path / "glossary.py"
        glossary_py.write_text("GLOSSARY: dict = {}\n", encoding="utf-8")
        extractor = GlossaryExtractor(_mock_agent(), glossary={})
        suggestions = [
            GlossarySuggestion(
                term_en="throughput",
                translations={"ru": "пропускная способность", "he": "תפוקה"},
                already_in_glossary=True,
            )
        ]
        n = extractor.apply_to_glossary(suggestions, glossary_path=glossary_py)
        assert n == 0

    def test_suggestion_to_glossary_entry(self):
        sug = GlossarySuggestion(
            term_en="downtime",
            translations={"ru": "простой", "he": "זמן השבתה"},
        )
        entry = sug.to_glossary_entry()
        assert entry == {
            "downtime": {"ru": "простой", "he": "זמן השבתה", "en": "downtime"}
        }
