
# src/agents/product_owner_agent.py
from typing import Optional, List
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from src.utils.llm_config_utils.agent_config import az_model_client

from src.agents.prompts import CRITIC_PROMPT ,ORCHESTRATOR_PROMPT,PO_PROMPT,QA_PROMPT


from src.connectors.jira_tools import (
    # create_userstory,
    create_raidstory,
    fetch_epic_details,
    fetch_project_epics,
    post_userstory,
    post_raidstory,
    extract_userstories_from_epic
)

# --------------------
# Tools (as you have)
# --------------------
fetch_project_all_epics_tool = FunctionTool(
    func=fetch_project_epics.tool, 
    name="fetch_project_epics",
    description="""List epics for a given Jira project key. Example Input: { "project_key": "ABCDE" } ..."""
)
fetch_epic_details_tool = FunctionTool(
    func=fetch_epic_details.tool,
    name="FETCH_EPIC_DETAILS",
    description="""
    Returns a dictionary containing all the details of an epic given an Epic Id.
    """
)

post_userstory_tool = FunctionTool(
    func=post_userstory.tool,
    name="POST_USERSTORY",
    description="""
    Post ONE user story to Jira under an epic.
    """
)


# --------------------
# Base prompt
# --------------------

# --------------------
# Base prompt (unchanged)
# --------------------
BASE_SYSTEM_MESSAGE = PO_PROMPT
 
def _normalize_list_upper(values: Optional[List[str]]) -> List[str]:
    return sorted({str(v).strip().upper() for v in (values or []) if v})

def make_scope_header(
    application_id: Optional[str],
    jira_projects: Optional[List[str]],
    jira_issues: Optional[List[str]],
    aha_ids: Optional[List[str]],
    confluence_spacekeys: Optional[List[str]] = None,   # <-- new
) -> str:
    jp_list = _normalize_list_upper(jira_projects)
    ji_list = _normalize_list_upper(jira_issues)
    ah_list = _normalize_list_upper(aha_ids)
    cf_list = _normalize_list_upper(confluence_spacekeys)

    jp = ", ".join(jp_list) if jp_list else "(none)"
    ji = ", ".join(ji_list) if ji_list else "(none)"
    ah = ", ".join(ah_list) if ah_list else "(none)"
    cf = ", ".join(cf_list) if cf_list else "(none)"
    app = application_id or "(unknown)"

    return (
        "### APPLICATION SCOPE (STRICT)\n"
        f"application_id: {app}\n\n"
        "JIRA:\n"
        f"  - Allowed project keys: {jp}\n"
        f"  - Allowed issue keys:   {ji}\n\n"
        "AHA:\n"
        f"  - Allowed IDs: {ah}\n\n"
        "CONFLUENCE:\n"
        f"  - Allowed space keys: {cf}\n\n"
        "BOUNDARY RULES:\n"
        "- Do NOT exceed the defined scope.\n"
        f"- Operate ONLY within the {jp},{ah},{cf} project keys, reject all others request apart from this scope.\n"
        "- Do NOT ask for project key; it is already known from context.\n"
        "- Treat Jira keys case-insensitively; normalize to UPPERCASE when reasoning or calling tools.\n"
        "- For any Jira/Aha READ, prefer tools; if tools are unavailable/fail, do NOT fabricate—say the data is unavailable.\n"
        "### END SCOPE"
    )

def build_product_owner_agent(
    application_id: Optional[str] = None,
    jira_projects: Optional[List[str]] = None,
    jira_issues: Optional[List[str]] = None,
    aha_ids: Optional[List[str]] = None,
    confluence_spacekeys: Optional[List[str]] = None,   # <-- new
) -> AssistantAgent:
    jira_projects = _normalize_list_upper(jira_projects)
    jira_issues = _normalize_list_upper(jira_issues)
    aha_ids = _normalize_list_upper(aha_ids)
    confluence_spacekeys = _normalize_list_upper(confluence_spacekeys)

    if not jira_projects:
        missing_scope_note = (
            "\n\n[NOTE FOR AGENT]\n"
            "No Jira project key is available in scope. "
            "Before using Jira tools, ask ONE concise question to get the project key from the user.\n"
        )
    else:
        missing_scope_note = ""

    scope_header = make_scope_header(application_id, jira_projects, jira_issues, aha_ids, confluence_spacekeys)
    final_system_message = f"{scope_header}\n\n{BASE_SYSTEM_MESSAGE}{missing_scope_note}"

    return AssistantAgent(
        name="PRODUCT_OWNER_ASSISTANT",
        model_client=az_model_client,
        system_message=final_system_message,
        tools=[
            fetch_project_all_epics_tool,
            # create_userstory_tool,
            create_userstory_tool_multiple,
            fetch_epic_details_tool,
            post_userstory_tool,
            create_jira_raidstory_tool,
            post_productRAID_tool,
        ],
        reflect_on_tool_use=True,
        max_tool_iterations=3,
    
    )