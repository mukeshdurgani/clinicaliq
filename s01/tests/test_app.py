"""
s01/tests/test_app.py
----------------------
Tests for app.py: Streamlit UI + Human-in-the-Loop (ClinicalIQ).

Ported from WealthDesk's s13/tests/test_s13.py onto ClinicalIQ's supervisor
graph (documents_agent / services_agent instead of documents_agent / rates_agent).

Run with:
    pytest s01/tests/ -v

All tests are pure Python -- no Streamlit context needed.
The app helper functions are imported directly from app.py.

Test groups:
  TestBuildInputState    -- build_input_state() returns correct graph input dict
  TestGetThreadConfig    -- get_thread_config() returns correct LangGraph config
  TestComplianceBadge    -- compliance_badge() returns correct display text
  TestNeedsHumanReview   -- needs_human_review() detects REVISED status (HITL trigger)
  TestFormatRouteLabel   -- format_route_label() formats routing info correctly
  TestAgentGraph         -- build_graph() compiles; supervisor/compliance nodes present
"""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_DIR = Path(__file__).parent.parent

# Load app.py without triggering Streamlit
_spec = importlib.util.spec_from_file_location("app", APP_DIR / "app.py")
_app  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app)

build_input_state  = _app.build_input_state
get_thread_config  = _app.get_thread_config
compliance_badge   = _app.compliance_badge
needs_human_review = _app.needs_human_review
format_route_label = _app.format_route_label

from clinicaliq.agent import build_graph  # noqa: E402
import clinicaliq.nodes as _nodes         # noqa: E402


# ---------------------------------------------------------------------------
# TestBuildInputState
# ---------------------------------------------------------------------------

class TestBuildInputState:
    def test_has_customer_message(self):
        state = build_input_state("What should I bring for a blood test?")
        assert state["customer_message"] == "What should I bring for a blood test?"

    def test_has_empty_response(self):
        assert build_input_state("test")["response"] == ""

    def test_has_empty_specialist(self):
        assert build_input_state("test")["specialist"] == ""

    def test_has_empty_retrieved_docs(self):
        assert build_input_state("test")["retrieved_docs"] == []

    def test_has_empty_compliance_status(self):
        assert build_input_state("test")["compliance_status"] == ""

    def test_all_required_keys_present(self):
        state    = build_input_state("test")
        required = {"customer_message", "response", "specialist", "retrieved_docs", "compliance_status"}
        assert required.issubset(set(state.keys()))

    def test_different_messages_differ(self):
        assert build_input_state("appointments")["customer_message"] != build_input_state("test prep")["customer_message"]

    def test_empty_message_accepted(self):
        assert build_input_state("")["customer_message"] == ""


# ---------------------------------------------------------------------------
# TestGetThreadConfig
# ---------------------------------------------------------------------------

class TestGetThreadConfig:
    def test_returns_dict(self):
        assert isinstance(get_thread_config("abc"), dict)

    def test_has_configurable_key(self):
        assert "configurable" in get_thread_config("abc")

    def test_configurable_has_thread_id(self):
        assert get_thread_config("my-thread")["configurable"]["thread_id"] == "my-thread"

    def test_different_ids_produce_different_configs(self):
        c1 = get_thread_config("t1")
        c2 = get_thread_config("t2")
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]


# ---------------------------------------------------------------------------
# TestComplianceBadge
# ---------------------------------------------------------------------------

class TestComplianceBadge:
    def test_pass_returns_checkmark(self):
        assert "✅" in compliance_badge("PASS")

    def test_revised_returns_warning(self):
        assert "⚠️" in compliance_badge("REVISED")

    def test_fail_returns_cross(self):
        assert "❌" in compliance_badge("FAIL: banned phrase: 'ibuprofen'")

    def test_empty_status_returns_empty_string(self):
        assert compliance_badge("") == ""

    def test_unknown_status_returns_empty_string(self):
        assert compliance_badge("UNKNOWN") == ""

    def test_pass_text(self):
        assert "Compliant" in compliance_badge("PASS")

    def test_revised_text(self):
        assert "Revised" in compliance_badge("REVISED")


# ---------------------------------------------------------------------------
# TestNeedsHumanReview
# ---------------------------------------------------------------------------

class TestNeedsHumanReview:
    def test_revised_returns_true(self):
        assert needs_human_review({"compliance_status": "REVISED"}) is True

    def test_pass_returns_false(self):
        assert needs_human_review({"compliance_status": "PASS"}) is False

    def test_fail_returns_false(self):
        assert needs_human_review({"compliance_status": "FAIL: banned phrase"}) is False

    def test_empty_returns_false(self):
        assert needs_human_review({"compliance_status": ""}) is False

    def test_missing_key_returns_false(self):
        assert needs_human_review({}) is False

    def test_escalated_specialist_is_not_hitl(self):
        assert needs_human_review({"compliance_status": "PASS", "specialist": "escalated"}) is False


# ---------------------------------------------------------------------------
# TestFormatRouteLabel
# ---------------------------------------------------------------------------

class TestFormatRouteLabel:
    def test_includes_specialist(self):
        result = {"specialist": "services_agent", "compliance_status": "PASS"}
        assert "services_agent" in format_route_label(result)

    def test_includes_badge_for_pass(self):
        result = {"specialist": "services_agent", "compliance_status": "PASS"}
        assert "✅" in format_route_label(result)

    def test_no_badge_for_empty_status(self):
        result = {"specialist": "escalated", "compliance_status": ""}
        label  = format_route_label(result)
        assert "✅" not in label and "⚠️" not in label and "❌" not in label

    def test_dash_for_missing_keys(self):
        assert "—" in format_route_label({})

    def test_revised_badge_shown(self):
        result = {"specialist": "documents_agent", "compliance_status": "REVISED"}
        assert "⚠️" in format_route_label(result)

    def test_documents_agent_in_label(self):
        result = {"specialist": "documents_agent", "compliance_status": "PASS"}
        assert "documents_agent" in format_route_label(result)


# ---------------------------------------------------------------------------
# TestAgentGraph
# ---------------------------------------------------------------------------

class TestAgentGraph:
    def test_build_graph_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert build_graph(checkpointer=MemorySaver()) is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "classify" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_compliance_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        nodes = build_graph(checkpointer=MemorySaver()).get_graph().nodes
        assert any("compliance" in n for n in nodes)

    def test_graph_has_documents_and_services_nodes(self):
        from langgraph.checkpoint.memory import MemorySaver
        nodes = build_graph(checkpointer=MemorySaver()).get_graph().nodes
        assert any("call_documents_agent" in n for n in nodes)
        assert any("call_services_agent"  in n for n in nodes)

    def test_graph_invocable_returns_response(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_documents_agent") as mock_da, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="POLICY")
            mock_da.invoke.return_value  = {
                "response":       "Please bring a valid ID and arrive 15 minutes early.",
                "history":        [],
                "retrieved_docs": [],
                "specialist":     "documents_agent",
            }
            mock_ca.invoke.return_value  = {
                "response":          "Please bring a valid ID and arrive 15 minutes early.",
                "compliance_status": "PASS",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                build_input_state("What should I bring for my appointment?"),
                config=get_thread_config("test-app-graph"),
            )
        assert "response" in result
        assert result["compliance_status"] == "PASS"

    def test_revised_result_triggers_hitl(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_services_agent") as mock_sa, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="SERVICES")
            mock_sa.invoke.return_value  = {
                "response":       "Taking ibuprofen twice daily will guarantee recovery.",
                "history":        [],
                "retrieved_docs": [],
                "specialist":     "services_agent",
            }
            mock_ca.invoke.return_value  = {
                "response":          "Please speak with our nurse about medication options.",
                "compliance_status": "REVISED",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            # NOTE: deliberately not "What medicine should I take for my fever?" --
            # that phrase is now caught by nodes.py's deterministic ESCALATE
            # pre-filter (see ESCALATE_KEYWORD_PATTERNS in config.py) and routes
            # to the static escalate() response before ever reaching the mocked
            # classifier/services/compliance agents below. Use a SERVICES-shaped
            # query the pre-filter doesn't match, so this test still exercises
            # the specialist -> compliance -> HITL path it's named for.
            result = graph.invoke(
                build_input_state("What pain relief products are available at the pharmacy counter?"),
                config=get_thread_config("test-app-hitl"),
            )
        assert needs_human_review(result) is True
