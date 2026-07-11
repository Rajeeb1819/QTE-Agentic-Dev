# Z_C_Autogen - High Level Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MENDIX UI (Frontend)                                 │
│                   (Chat Interface - User Input/Output)                       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             │ HTTP POST /chat
                             │ ChatIn: {content, application_id}
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (demo.py)                                  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. INPUT VALIDATION                                                 │    │
│  │   ├─ Check if input is empty/whitespace                            │    │
│  │   └─ Return error if invalid                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. LOAD CONVERSATION CONTEXT (STM)                                 │    │
│  │   ├─ Get short-term memory from MSSQL                             │    │
│  │   ├─ Prepare last 20 messages (5 recent + 15 historical)         │    │
│  │   └─ Log memory sample                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. STORE USER MESSAGE                                               │    │
│  │   └─ Persist user input to MSSQL history                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. FETCH SCOPE (JIRA, AHA, CONFLUENCE)                             │    │
│  │   ├─ Get application scope from ScopeService                       │    │
│  │   ├─ Extract JIRA projects, Aha IDs, Confluence spaces            │    │
│  │   └─ Fallback to empty scope if error                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 5. BUILD/CACHE AGENTS                                               │    │
│  │   ├─ Check agent cache (hash-based on scope)                       │    │
│  │   ├─ If scope changed → rebuild agents                            │    │
│  │   └─ If scope same → reuse cached agents                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                 │
└─────────────────────────────┼──────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                            │
        ▼                                            ▼
   ┌────────────────────────────┐          ┌──────────────────────────┐
   │    AGENT TEAM SETUP        │          │   AGENT CACHE LAYER      │
   └────────────────────────────┘          └──────────────────────────┘
        │                                         │
        ├─ Orchestrator Agent                     │
        │  └─ Routes tasks to appropriate team   │
        │                                         │
        ├─ Product Owner Agent                    │
        │  └─ Creates epics, user stories         │
        │  └─ Business requirement analysis       │
        │                                         │
        ├─ QA Agent                               │
        │  └─ Test case generation                │
        │  └─ Quality assessment                  │
        │                                         │
        ├─ Test Manager Agent                     │
        │  └─ Test planning & execution           │
        │  └─ Coverage metrics                    │
        │                                         │
        └─ Critic Agent                           │
           └─ Reviews all decisions               │
           └─ Quality control                     │
                   │
        ┌──────────┴───────────┐
        │                      │
        ▼                      ▼
    ┌───────────────────────────────────┐
    │  BUILD CONTEXT BLOCK              │
    │  (Hybrid approach)                │
    │                                   │
    │  [Recent Messages (5)]            │
    │  ├─ Full content as-is           │
    │  └─ Last 5 turns verbatim        │
    │                                   │
    │  [Historical Summary (15)]        │
    │  ├─ Messages 6-20 from end       │
    │  └─ LLM-summarized (max 500w)    │
    │                                   │
    │  [Current Scope]                 │
    │  ├─ JIRA Projects                │
    │  ├─ Aha IDs                      │
    │  └─ Confluence Spaces            │
    │                                   │
    │  [Guidance]                      │
    │  └─ System instructions          │
    └───────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────┐
    │  TEAM EXECUTION (SelectorGroupChat)         │
    │                                             │
    │  1. Wrap user input with context           │
    │  2. Run team.run(task=user_msg)            │
    │  3. Agents respond iteratively              │
    │     ├─ Max turns: 3 (configurable)         │
    │     ├─ Termination: "APPROVE" or max turns│
    │     └─ No repeated speakers                │
    │  4. Collect all messages                    │
    │  5. Select final output                     │
    └─────────────────────────────────────────────┘
        │
        ├─ Agent-to-Agent Communication
        │  └─ Selector func determines next speaker
        │
        ├─ Tool Calls (Optional)
        │  ├─ JIRA tools (create/fetch stories, epics)
        │  ├─ Aha tools (fetch features)
        │  └─ Confluence tools (validate pages)
        │
        └─ Error Handling
           ├─ LLM API errors (401, 403, 429, 503)
           ├─ Backoff & retry logic
           └─ Classified error responses
                │
                ▼
    ┌────────────────────────────────────┐
    │ ERROR CLASSIFICATION & HANDLING    │
    │                                    │
    │ • 401 Unauthorized                │
    │ • 403 Forbidden                   │
    │ • 429 Rate Limit (reduced retries)│
    │ • 503 Model Unavailable           │
    │ • 500-504 Server Errors           │
    │ • Generic Unexpected Errors       │
    └────────────────────────────────────┘
                │
                ▼
    ┌────────────────────────────────────┐
    │ PERSISTENCE LAYER                  │
    │                                    │
    │ 1. Save team state (if no error)  │
    │ 2. Persist messages to full       │
    │    history (skip duplicates)      │
    │ 3. Store final message to STM     │
    │    (short-term memory)            │
    └────────────────────────────────────┘
                │
                ▼
    ┌────────────────────────────────────┐
    │ BUILD RESPONSE (ChatOut)           │
    │                                    │
    │ • Message ID & source             │
    │ • Token usage (per-agent breakdown)│
    │ • Critic review (if available)    │
    │ • Metadata (errors, timestamps)   │
    │ • Full conversation data          │
    │ • Mime type & type info           │
    └────────────────────────────────────┘
                │
                ▼
    ┌────────────────────────────────────┐
    │ RETURN TO MENDIX UI                │
    │                                    │
    │ ChatOut JSON with:                │
    │ • content (agent response)        │
    │ • final_output                    │
    │ • critic_review                   │
    │ • models_usage (tokens)           │
    │ • metadata (errors, timing)       │
    └────────────────────────────────────┘
                │
                ▼
            MENDIX UI
         (Display Response)
```

---

## Database Architecture (MSSQL)

```
┌──────────────────────────────────────────────────────────────────┐
│                     MSSQL Database                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ CHAT_HISTORY (Full History)                             │    │
│  │ ├─ id (PK)                                              │    │
│  │ ├─ application_id (FK)                                  │    │
│  │ ├─ role (PO, QA, TM, Orchestrator, Critic, system)   │    │
│  │ ├─ content (message body)                              │    │
│  │ ├─ created_at (timestamp)                              │    │
│  │ └─ metadata (JSON - optional)                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ SHORT_TERM_MEMORY (STM - Last N Messages)              │    │
│  │ ├─ id (PK)                                              │    │
│  │ ├─ application_id (FK)                                  │    │
│  │ ├─ role (agent name)                                    │    │
│  │ ├─ content (message)                                    │    │
│  │ ├─ created_at (timestamp)                              │    │
│  │ └─ [Soft delete or aging out old messages]            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ SCOPE (Application Configuration)                       │    │
│  │ ├─ application_id (PK)                                  │    │
│  │ ├─ jira_projectkey                                      │    │
│  │ ├─ aha_id                                               │    │
│  │ ├─ confluence_spacekey                                  │    │
│  │ └─ created_at / updated_at                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ TEAM_STATE (Persisted Team State)                       │    │
│  │ ├─ application_id (PK)                                  │    │
│  │ ├─ team_state_json (serialized team state)             │    │
│  │ └─ last_updated (timestamp)                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ EVAL_SCORES (Ragas Observability)                       │    │
│  │ ├─ run_id (FK)                                          │    │
│  │ ├─ metric_name (context_precision, answer_relevancy)   │    │
│  │ ├─ metric_value (float)                                │    │
│  │ └─ created_at                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## External Integrations

```
┌─────────────────────────────────────────────────────────────────┐
│                   External Systems                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │   JIRA API       │     │   Aha! API       │                  │
│  ├──────────────────┤     ├──────────────────┤                  │
│  │ Create Epics     │     │ Fetch Features   │                  │
│  │ Create Stories   │     │ Feature Details  │                  │
│  │ Extract Stories  │     │ Map to Jira      │                  │
│  │ Fetch Epics      │     └──────────────────┘                  │
│  │ Fetch Issues     │                                            │
│  └──────────────────┘     ┌──────────────────┐                  │
│         │                 │ Confluence API   │                  │
│         │                 ├──────────────────┤                  │
│         │                 │ Validate Pages   │                  │
│         │                 │ Fetch Content    │                  │
│         │                 └──────────────────┘                  │
│         │                                                        │
│         │                 ┌──────────────────┐                  │
│         │                 │ Azure OpenAI     │                  │
│         └──────────────────┼──────────────────┤                  │
│                            │ LLM Completions  │                  │
│                            │ Embeddings       │                  │
│                            │ (Backoff & Retry)│                  │
│                            └──────────────────┘                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Example: User Query → Agent Response

```
User Input: "list all epics"
    │
    ▼
[VALIDATION] ✓ Not empty
    │
    ▼
[LOAD STM] Get last 20 messages from MSSQL
    │
    ▼
[STORE USER MSG] Save "list all epics" to MSSQL history
    │
    ▼
[FETCH SCOPE] Get JIRA project key from scope
    │
    ▼
[BUILD AGENTS] Load/create PO, QA, TM, Orchestrator, Critic
    │
    ▼
[BUILD CONTEXT] 
    ├─ Recent 5 messages (full)
    ├─ Historical 15 messages (summarized by LLM)
    ├─ Current scope (JIRA project)
    └─ System guidance
    │
    ▼
[WRAP PROMPT]
    context_block + user_input
    │
    ▼
[TEAM.RUN]
    1. Orchestrator decides: "This is a list request → PO Agent"
    2. PO Agent calls JIRA tool: fetch_project_epics()
    3. PO Agent returns: "Found 5 epics: [Epic1, Epic2, ...]"
    4. Critic Agent reviews response
    5. Final output selected
    │
    ▼
[SAVE STATE] Persist team state
    │
    ▼
[PERSIST MESSAGES] Save all agent messages to full history
    │
    ▼
[PERSIST FINAL] Save final message to STM
    │
    ▼
    [RESPONSE] Return ChatOut to Mendix UI with:
    {
        "content": "Found 5 epics: [Epic1, Epic2, ...]",
        "final_output": "Found 5 epics: [Epic1, Epic2, ...]",
        "critic_review": "Response is accurate and complete",
        "models_usage": {
            "prompt_tokens": 2500,
            "completion_tokens": 150
        },
        "data": [full conversation messages]
    }
    │
    ▼
Mendix UI displays response in chat
```

---

## Configuration & Environment Variables

```
# Context Management
CHAT_MAX_RECENT_MESSAGES=5              # Last N messages in full
CHAT_MAX_HISTORICAL_MESSAGES=15         # Messages 6-20 for summarization
CHAT_MAX_CONTEXT_TOKENS=3500            # Total token budget

# Team Execution
CHAT_MAX_TURNS_PER_RUN=3                # Max iterations per request
CHAT_MAX_RETRIES_RATE_LIMIT=1           # Retries for rate limit errors
CHAT_ENABLE_RATE_LIMITER=0              # Optional token bucket limiter

# Logging
LOG_LEVEL=INFO
CHAT_LOG_MEMORY=1                       # Log STM sample
CHAT_LOG_NO_TRUNCATE=0                  # Don't truncate long logs
CHAT_MEMORY_LOG_MAX_CHARS=1000

# External APIs
JIRA_DOMAIN=<domain>
JIRA_EMAIL=<email>
JIRA_API_TOKEN=<token>

CONF_BASE=<domain>
AHA_API_KEY=<key>

AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=<endpoint>
```

---
## Key Features & Optimizations

### 1. **Hybrid Context Strategy**
- Recent messages: Full verbatim content
- Historical messages: LLM-summarized for efficiency
- Scope emphasis: Always placed first to override historical context

### 2. **Agent Caching**
- Hash-based scope detection
- Agents reused if scope unchanged
- Automatic rebuild if scope changes

### 3. **Error Handling**
- Classified LLM errors (401, 403, 429, 503, 500-504)
- Reduced retries for rate limit errors
- User-friendly error messages

### 4. **Concurrent Request Support**
- No semaphores (full parallelism)
- Per-app request tracking
- Optional token bucket rate limiter

### 5. **Token Tracking**
- Per-agent token breakdown
- Total prompt/completion tokens
- Cost calculation capability

### 6. **Input Validation**
- Empty input detection
- Immediate validation response
- No unnecessary processing

### 7. **Observability (Ragas)**
- Trace collection per run
- Metrics storage
- Context preservation for analysis

---

## File Structure

```
Z_C_Autogen/
├── demo.py                          # Main FastAPI app
├── ARCHITECTURE.md                  # This file
│
├── src/
│   ├── agents/
│   │   ├── orchestrator_agent1.py   # Route/delegation agent
│   │   ├── product_owner_agent.py   # Business logic agent
│   │   ├── qa_agent.py              # QA agent
│   │   ├── test_manager_agent.py    # Test management agent
│   │   ├── critic_agent.py          # Quality review agent
│   │   ├── prompts.py               # Agent prompts
│   │   └── team_selectorchat.py     # Team setup
│   │
│   ├── connectors/
│   │   ├── jira_tools/
│   │   │   ├── fetch_project_epics.py
│   │   │   ├── fetch_epic_details.py
│   │   │   ├── extract_userstories_from_epic.py
│   │   │   ├── create_userstory.py
│   │   │   ├── post_userstory.py
│   │   │   └── post_raidstory.py    # RAID story posting
│   │   │
│   │   └── aha_tools/
│   │       ├── fetch_aha_feature_details.py
│   │       └── fetch_project_aha_features.py
│   │
│   ├── db_model_mssql/
│   │   ├── chat_service.py          # Chat CRUD operations
│   │   ├── scope_service3.py        # Scope management
│   │   ├── teams_registry.py        # Team state persistence
│   │   ├── session1.py              # DB session management
│   │   └── model.py                 # ORM models
│   │
│   ├── memory/
│   │   ├── history.py               # History management
│   │   └── memory_manager.py        # Memory operations
│   │
│   ├── utils/
│   │   ├── llm_config_utils/
│   │   │   ├── agent_config.py      # Model client setup
│   │   │   ├── llm_initiate.py      # LLM initialization
│   │   │   └── ragas_utils.py       # Ragas configuration
│   │   │
│   │   ├── selector_func.py         # Agent selector logic
│   │   └── ...
│   │
│   └── Ragas_observability/
│       ├── trace.py                 # Trace management
│       ├── metrics_store.py         # Metrics persistence
│       └── contexts_from_messages.py # Context extraction
│
└── requirements.txt
```

---

## Summary

The system follows a **conversational AI orchestration** pattern:

1. **Request** → User input via Mendix
2. **Validation** → Quick check for empty/invalid input
3. **Context Loading** → Fetch conversation history from MSSQL
4. **Scope Resolution** → Get project scope (JIRA/Aha/Confluence)
5. **Agent Assembly** → Load or build agents with caching
6. **Context Building** → Create hybrid context (recent + summarized)
7. **Team Execution** → Agents collaborate, call external APIs
8. **Persistence** → Save state and messages to MSSQL
9. **Response** → Return structured ChatOut to Mendix UI

**Key Innovation**: Hybrid context strategy balances detail (recent messages) with efficiency (summarized history), ensuring agents always work with the most relevant and current scope information.
