
import json
import re
from hashlib import sha1
from typing import Any, Dict, Optional, Tuple, Literal, Sequence, List
from autogen_agentchat.messages import TextMessage, BaseChatMessage, BaseAgentEvent
# ======================================================================================
# Utility: render structured tool dicts and payload extraction
# ======================================================================================

def normalize_tool_dict_to_text(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict) or not d:
        return "No details found."
    lines: List[str] = []

    def add(name: str, value: Any, label: Optional[str] = None):
        if value is None:
            return
        val = value
        if isinstance(value, (dict, list)):
            val = json.dumps(value, ensure_ascii=False, indent=2)
        lines.append(f"{label or name}: {val}")

    add("Feature ID", d.get("feature_id"))
    add("Title", d.get("feature_details"))
    add("Description", d.get("description"))
    ac = d.get("acceptance_criteria")
    if isinstance(ac, list) and ac:
        lines.append("Acceptance criteria:")
        for i, item in enumerate(ac, 1):
            lines.append(f"  {i}. {item}")
    add("Linked tests", d.get("linked_test_keys"))
    add("Linked test summaries", d.get("linked_test_summaries"))
    return "\n".join(lines) if lines else "No details found."


def extract_payload_from_message(msg: Any) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    try:
        if isinstance(msg, TextMessage):
            c = getattr(msg, "content", None)
            if isinstance(c, str):
                return c, None, None
            if isinstance(c, (dict, list)):
                return normalize_tool_dict_to_text(c) if isinstance(c, dict) else None, c, "application/json"
            if isinstance(c, (bytes, bytearray)):
                try:
                    return c.decode("utf-8", errors="replace"), None, "text/plain"
                except Exception:
                    return None, c, "application/octet-stream"
            return str(c) if c is not None else None, None, None
    except Exception:
        pass

    if isinstance(msg, (dict, list)):
        text = normalize_tool_dict_to_text(msg) if isinstance(msg, dict) else None
        return text, msg, "application/json"

    if isinstance(msg, (bytes, bytearray)):
        try:
            return msg.decode("utf-8", errors="replace"), None, "text/plain"
        except Exception:
            return None, msg, "application/octet-stream"

    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c, None, None
    if isinstance(c, (dict, list)):
        return normalize_tool_dict_to_text(c) if isinstance(c, dict) else None, c, "application/json"
    if c is not None:
        return str(c), None, None

    return str(msg), None, None


# ======================================================================================
# Runtime selector with helpers
# ======================================================================================

def make_selector_and_helpers(
    ORCHESTRATOR_NAME: str,
    CRITIC_NAME: str,
    PO_NAME: str,
    QA_NAME: str,
    TM_NAME: str
):
    MAIN_SOURCES_SET = {PO_NAME, QA_NAME, TM_NAME}

    def _text_of(msg: Any) -> str:
        try:
            txt, _, _ = extract_payload_from_message(msg)
            return (txt or "")
        except Exception:
            return str(getattr(msg, "content", "") or "")

    def latest_orchestrator_plan_text(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str:
        for m in reversed(messages):
            if getattr(m, "source", None) == ORCHESTRATOR_NAME:
                return (_text_of(m)).lower()
        return ""

    def latest_user_text(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str:
        for m in reversed(messages):
            if getattr(m, "source", None) == "user":
                return (_text_of(m)).lower()
        return ""

    def agent_from_plan(plan_text: str) -> Optional[str]:
        alias_map = {
            PO_NAME: [PO_NAME.lower(), "product owner agent", "product_owner_agent", "po", "po_agent"],
            QA_NAME: [QA_NAME.lower(), "qa manager agent", "qa assistant", "quality manager agent", "qa_agent", "qa"],
            TM_NAME: [TM_NAME.lower(), "test manager agent", "test_manager_agent", "tm", "test manager assistant", "test manager"],
        }
        hits: list[tuple[int, str]] = []
        for agent, tokens in alias_map.items():
            for tok in tokens:
                pattern = r"\b{}\b".format(re.escape(tok))
                m = re.search(pattern, plan_text, flags=re.IGNORECASE)
                if m:
                    hits.append((m.start(), agent))
                    break
        hits.sort(key=lambda x: x[0])
        return hits[0][1] if hits else None

    def is_jira_intent(text: str) -> bool:
        text = (text or "").lower()
        jira_terms = ["jira", "story", "stories", "epic", "epics", "raid", "raid item", "backlog", "acceptance criteria", "issue", "ticket"]
        return any(k in text for k in jira_terms)

    def is_qa_intent(text: str) -> bool:
        """
        Detect QA-specific intent (coverage metrics ONLY).
        
        CRITICAL: Keep keywords VERY SPECIFIC to avoid hijacking PO tasks.
        - Use multi-word phrases instead of single generic words
        - Avoid overlap with JIRA/story/epic keywords
        """
        text = (text or "").lower()
        
        # VERY SPECIFIC coverage-related keywords (multi-word phrases preferred)
        qa_terms = [
            "coverage metric",
            "coverage metrics",
            "coverage report",
            "coverage analysis",
            "test coverage",
            "acceptance criteria coverage",
            "ac coverage",
            "defect rate",
            "bug triage",
            "quality metrics",
            "qa metrics",
        ]
        
        return any(k in text for k in qa_terms)

    def is_tm_intent(text: str) -> bool:
        text = (text or "").lower()
        tm_terms = ["test case", "test cases", "test plan", "execution", "schedule", "scheduling"]
        return any(k in text for k in tm_terms)

    def map_intent_to_agent(text: str) -> Optional[str]:
        """
        Map user intent to agent based on keyword analysis.
        
        Priority order (CRITICAL):
        1. Product Owner - JIRA operations (stories, epics, RAID, fetch details) - HIGHEST PRIORITY
        2. Test Manager - specific test case management (only if NO Jira context)
        3. QA Agent - coverage metrics ONLY when no JIRA story/epic operations
        
        This order ensures epic/story operations always go to PO first, even if "test" keywords present.
        """
        text = (text or "").lower()
        
        # Product Owner: JIRA operations (stories, epics, RAID, fetch details)
        # Check FIRST to prevent Test Manager from hijacking epic detail requests
        # Examples: "fetch epic PROJ-123", "create story for epic", "test plan for PROJ-123"
        if is_jira_intent(text):
            return PO_NAME
        
        # Test Manager: Test case management (only if NO Jira epic/story context)
        if is_tm_intent(text):
            return TM_NAME
        
        # QA Agent: Coverage metrics ONLY (least priority)
        # Only gets selected if no JIRA operations detected
        if is_qa_intent(text):
            return QA_NAME
        
        return None
    def _last_main_agent(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> Optional[str]:
        for m in reversed(messages):
            s = getattr(m, "source", None)
            if s in MAIN_SOURCES_SET:
                return s
        return None

    def _looks_like_question(text: str) -> bool:
        t = (text or "").strip().lower()
        if "?" in t:
            return True
        starters = ("please provide", "can you", "could you", "share", "what is", "which ", "give me")
        return t.startswith(starters)

    def _is_qa_coverage_output(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> bool:
        last_qa = None
        for m in reversed(messages):
            if getattr(m, "source", None) == QA_NAME:
                last_qa = m
                break
        if not last_qa:
            return False
        text = _text_of(last_qa).lower()
        coverage_signals = [
            "coverage",
            "acceptance criteria coverage",
            "posted",
            "published",
            "confluence",
            "post_jira_rows_to_confluence",
        ]
        is_coverage = any(k in text for k in coverage_signals)
        if _looks_like_question(text):
            return False
        return is_coverage

    def pick_final_output_from_last_main(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> Dict[str, Optional[str]]:
        # Prefer last main agent output
        for m in reversed(messages):
            src = getattr(m, "source", None)
            if src in MAIN_SOURCES_SET and getattr(m, "content", None) is not None:
                content, _, mime = extract_payload_from_message(m)
                return {"content": content, "source": src, "mime_type": mime}
        # Otherwise, last non-critic, non-user content
        for m in reversed(messages):
            src = getattr(m, "source", None)
            if src and src != CRITIC_NAME and src != "user" and getattr(m, "content", None) is not None:
                content, _, mime = extract_payload_from_message(m)
                return {"content": content, "source": src, "mime_type": mime}
        # Fallback to last message
        last = messages[-1] if messages else None
        content, _, mime = extract_payload_from_message(last) if last else (None, None, None)
        src = getattr(last, "source", None) if last else None
        return {"content": content, "source": src, "mime_type": mime}

    def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> Optional[str]:
        if not messages:
            return ORCHESTRATOR_NAME

        last = messages[-1]
        last_src = getattr(last, "source", None)
        last_text = (_text_of(last) or "").lower()

        if last_src == "user":
            return ORCHESTRATOR_NAME

        if last_src == ORCHESTRATOR_NAME:
            plan_text = latest_orchestrator_plan_text(messages)
            assigned = agent_from_plan(plan_text)
            if assigned in MAIN_SOURCES_SET:
                return assigned

            user_text = latest_user_text(messages)
            delegate = map_intent_to_agent(user_text) or map_intent_to_agent(plan_text)
            return delegate or ORCHESTRATOR_NAME

        def _eq(a: Optional[str], b: Optional[str]) -> bool:
            return (a or "").strip().lower() == (b or "").strip().lower()

        # Optionally bypass critic based on heuristic
        if _eq(last_src, QA_NAME) and _is_qa_coverage_output(messages):
            return CRITIC_NAME

        if last_src in MAIN_SOURCES_SET:
            return CRITIC_NAME

        if last_src == CRITIC_NAME:
            if "approve" in last_text:
                return None
            prev_worker = _last_main_agent(messages)
            return prev_worker or ORCHESTRATOR_NAME

        return ORCHESTRATOR_NAME

    return selector_func, pick_final_output_from_last_main, MAIN_SOURCES_SET
