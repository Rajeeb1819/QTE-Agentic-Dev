# Chat Endpoint (/chat) - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture & Flow Diagram](#architecture--flow-diagram)
3. [Request/Response Model](#requestresponse-model)
4. [Step-by-Step Execution Flow](#step-by-step-execution-flow)
5. [Key Components](#key-components)
6. [Context Building Strategy](#context-building-strategy)
7. [Error Handling](#error-handling)
8. [Multi-User Concurrency](#multi-user-concurrency)
9. [Database Interactions](#database-interactions)
10. [Token Usage Tracking](#token-usage-tracking)
11. [Configuration Parameters](#configuration-parameters)
12. [Debugging & Monitoring](#debugging--monitoring)
13. [Common Issues & Solutions](#common-issues--solutions)

---

## Overview

The `/chat` endpoint is the **primary entry point** for all conversational AI interactions. It orchestrates a team of specialized agents (Product Owner, QA, Test Manager, Orchestrator, Critic) to handle complex user requests with multi-turn conversations.

**Key Features:**
- ✅ Multi-user concurrent support (no semaphores/throttling)
- ✅ Hybrid context strategy (recent messages in full + historical messages summarized)
- ✅ Automatic message deduplication
- ✅ Team state persistence and resumption
- ✅ Comprehensive error classification and handling
- ✅ Per-agent token usage tracking
- ✅ Rate limiting (optional)

**Technologies:**
- **Framework:** FastAPI (async)
- **LLM:** Azure OpenAI (gpt-4)
- **Agent Orchestration:** AutoGen SelectorGroupChat
- **Database:** MSSQL (chat history + short-term memory)
- **Backoff Strategy:** Exponential backoff with jitter for LLM errors

---

## Architecture & Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    User POST /chat Endpoint                         │
│                      Content + Application_ID                        │
└────────────────────────┬────────────────────────────────────────────┘
                         │
         ┌───────────────▼───────────────────┐
         │ 1. VALIDATE INPUT                 │
         │    ├─ Not empty?                  │
         │    └─ Application_id present?     │
         └───────────────┬───────────────────┘
                         │
         ┌───────────────▼───────────────────────────────┐
         │ 2. LOAD SHORT-TERM MEMORY (STM) FROM DB       │
         │    ├─ Last 20 messages from app_short_memory  │
         │    ├─ Ordered: oldest → newest                │
         │    └─ Filter by application_id                │
         └───────────────┬───────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────────┐
         │ 3. STORE USER MESSAGE                                │
         │    ├─ To app_chat_history (full audit)               │
         │    └─ To app_short_memory (with auto-pruning to 20)  │
         └───────────────┬──────────────────────────────────────┘
                         │
         ┌───────────────▼───────────────────────────────────┐
         │ 4. LOAD SCOPE (Application Boundaries)            │
         │    ├─ Get JIRA projects allowed                   │
         │    ├─ Get AHA IDs allowed                         │
         │    ├─ Get Confluence spaces allowed               │
         │    └─ Inject as system constraint for agents      │
         └───────────────┬───────────────────────────────────┘
                         │
         ┌───────────────▼────────────────────────────────────────┐
         │ 5. BUILD/REUSE AGENTS (Cached by app_id + scope_hash)  │
         │    ├─ Orchestrator Agent                               │
         │    ├─ Product Owner Agent                              │
         │    ├─ QA Agent                                         │
         │    ├─ Test Manager Agent                               │
         │    └─ Critic Agent (final review)                      │
         └───────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────┐
         │ 6. BUILD HYBRID CONTEXT BLOCK                    │
         │    ├─ Last 5 messages: Full content (as-is)      │
         │    ├─ Messages 6-20: Summarized by LLM           │
         │    ├─ Current scope (always included)            │
         │    ├─ Guidance for agent behavior                │
         │    └─ Token budget: ~3500 tokens max             │
         └───────────────┬──────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────────┐
         │ 7. BUILD TEAM (SelectorGroupChat)                     │
         │    ├─ Add all 5 agents as participants                │
         │    ├─ Set selector function (who speaks next?)        │
         │    ├─ Termination: "APPROVE" OR max {3} turns         │
         │    └─ Backoff-enabled model client (auto-retry)       │
         └───────────────┬──────────────────────────────────────┘
                         │
         ┌───────────────▼────────────────────────────────────┐
         │ 8. WRAP USER MESSAGE                               │
         │    ├─ Include context block                        │
         │    ├─ Include current request                      │
         │    └─ Inject as TextMessage("user")               │
         └───────────────┬────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────────┐
         │ 9. RUN TEAM (team.run() with error handling)          │
         │    ├─ Agents collaborate (orchestrator selects next)  │
         │    ├─ Auto-retry on 429, 500, 502, 503, 504          │
         │    ├─ Handle 401/403 (non-retryable)                 │
         │    └─ Max 3 turns or "APPROVE" to stop               │
         └───────────────┬──────────────────────────────────────┘
                         │
         ┌───────────────▼────────────────────────────────────┐
         │ 10. SAVE TEAM STATE (if no error)                 │
         │     └─ To app_team_state table (JSON)             │
         └───────────────┬────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────┐
         │ 11. PERSIST TO FULL HISTORY (if no error)       │
         │     ├─ All non-user messages                     │
         │     └─ Skip duplicates (via fingerprint)         │
         └───────────────┬──────────────────────────────────┘
                         │
         ┌───────────────▼─────────────────────────────────┐
         │ 12. EXTRACT FINAL OUTPUT                        │
         │     ├─ Pick last main agent response            │
         │     ├─ Extract critic review (if available)     │
         │     └─ Normalize content                        │
         └───────────────┬─────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────┐
         │ 13. PERSIST FINAL TO STM (only)                 │
         │     ├─ Store final response                      │
         │     └─ Auto-pruned to 20 rows                   │
         └───────────────┬──────────────────────────────────┘
                         │
         ┌───────────────▼───────────────────────────────┐
         │ 14. AGGREGATE TOKEN USAGE                     │
         │     ├─ Total prompt + completion tokens       │
         │     ├─ Per-agent breakdown                    │
         │     └─ Log token consumption                  │
         └───────────────┬───────────────────────────────┘
                         │
                                  ┌───────────────▼──────────────────────────┐
         │ 15. BUILD RESPONSE (ChatOut)              │
         │     ├─ Final output                      │
         │     ├─ All conversation messages (data) │
         │     ├─ Token usage metadata              │
         │     ├─ Critic review                     │
         │     └─ Timestamps                        │
         └───────────────┬──────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────┐
         │ Response Sent to User                     │
         │ HTTP 200 + ChatOut JSON                   │
         └──────────────────────────────────────────┘
```

---

## Request/Response Model

### Request: ChatIn

```python
class ChatIn(BaseModel):
    content: str              # User's message (required, non-empty)
    application_id: str       # App identifier (required)
```

**Example Request:**
```json
{
  "content": "Create user stories for the authentication module",
  "application_id": "app_12345"
}
```

### Response: ChatOut

```python
class ChatOut(BaseModel):
    id: Optional[str] = None              # Message ID from last agent
    source: str = "assistant"             # Who sent the final message
    models_usage: Optional[ModelsUsage] = None  # Total tokens used
    metadata: Dict[str, Any] = {}         # Additional context
    created_at: datetime                  # Timestamp
    content: Optional[str] = None         # Final output text
    data: Optional[Any] = None            # Full conversation messages
    mime_type: Optional[str] = None       # Content type (usually text/plain)
    type: Literal["TextMessage", "ToolResult"] = "TextMessage"
    final_output: Optional[str] = None    # Same as content (for clarity)
    critic_review: Optional[str] = None   # Critic agent's feedback
```

**Example Response:**
```json
{
  "id": "msg_xyz",
  "source": "po_agent",
  "models_usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 800
  },
  "metadata": {
    "error_type": null
  },
  "created_at": "2026-03-25T10:30:00Z",
  "content": "Here are 3 user stories for authentication: ...",
  "data": [
    {
      "source": "user",
      "content": "Create user stories...",
      "created_at": "2026-03-25T10:29:00Z",
      "role": "user"
    },
    {
      "source": "orchestrator",
      "content": "I'll delegate this to PO Agent...",
      "created_at": "2026-03-25T10:29:30Z",
      "role": "orchestrator"
    },
    {
      "source": "po_agent",
      "content": "Here are 3 user stories...",
      "created_at": "2026-03-25T10:30:00Z",
      "role": "po_agent"
    }
  ],
  "final_output": "Here are 3 user stories for authentication: ...",
  "critic_review": "The user stories are well-structured and testable."
}
```

---

## Step-by-Step Execution Flow

### Step 1: Input Validation (Line 1002)

**Code:**
```python
if not request.content or not request.content.strip():
    logger.warning(f"[VALIDATION_FAILED] Empty user input for app {request.application_id}")
    return ChatOut(
        id=None,
        source="system",
        content="Please enter a valid query. Your message cannot be empty.",
        # ...
    )
```

**Purpose:** Reject empty requests early to save resources.

**What Happens:**
- ✅ If content is empty or only whitespace → Return error ChatOut
- ✅ If content has text → Continue to step 2

**Response Code:** HTTP 200 (error message in ChatOut)

---
### Step 2: Rate Limiting (Optional) (Line 1005)

**Code:**
```python
if ENABLE_RATE_LIMITER:
    await RATE_LIMITER.acquire(1)
```

**Purpose:** Optional token bucket rate limiter (disabled by default).

**Configuration:**
- Enabled via: `ENABLE_RATE_LIMITER=1` (environment variable)
- Default: Disabled (full concurrent access)
- Rate: 1 token per second, capacity 10

**When Used:**
- High-traffic environments where you want to limit requests per user
- Setting: Not recommended for most deployments (loses concurrent advantage)

---

### Step 3: Request Tracking (Line 1008)

**Code:**
```python
async with ACTIVE_REQUESTS_LOCK:
    ACTIVE_REQUESTS[request.application_id] += 1

t0 = tnow()
run_id = str(uuid4())

logger.info(f"[CONCURRENT] Processing request {run_id} for app {request.application_id}")
```

**Purpose:** Track concurrent requests for monitoring.

**Tracking Information:**
- `run_id`: Unique request identifier (UUID)
- `t0`: Start time for performance metrics
- `ACTIVE_REQUESTS[app_id]`: Counter of concurrent users
- Logged to debug concurrent behavior

**Output Examples:**
```
[CONCURRENT] Processing request abc-123-def for app app_12345 (active: 3)
```

---

### Step 4: Load Short-Term Memory (STM) (Line 1016)

**Code:**
```python
short_memory = await get_short_memory_db(request.application_id)
_log_memory_sample(logger, run_id=run_id, app_id=request.application_id, short_memory=short_memory)

print(f"[STM_DEBUG] loaded_from_db={len(short_memory)} rows")

# Prepare up to last 20 messages
max_context_messages = MAX_RECENT_MESSAGES + MAX_HISTORICAL_MESSAGES  # 5 + 15 = 20
seeded_rows = short_memory[-max_context_messages:] if len(short_memory) > max_context_messages else short_memory
```

**Purpose:** Fetch conversation history for context injection.

**What Happens:**
1. Query `app_short_memory` table for this application_id
2. Retrieve up to 20 most recent messages (ordered: oldest → newest)
3. Select the last 20 messages from STM (if more exist)
4. Log debug information (if `LOG_MEMORY=1`)

**Database Query (Conceptual):**
```sql
SELECT id, role, content, created_at FROM app_short_memory
WHERE application_id = @app_id
ORDER BY created_at ASC
LIMIT 20
```

**Output:**
```
[STM_DEBUG] loaded_from_db=15 rows
[STM_DEBUG] to_seed=15 rows (last 20 messages max)
[STM_DEBUG] seeded_ids_roles=[(1, 'user'), (2, 'orchestrator'), (3, 'po_agent'), ...]
```

---

### Step 5: Build Fingerprints (Line 1039)

**Code:**
```python
prev_fps: set[str] = set()
for m in short_memory:
    role = m.get("role") or m.get("source") or "unknown"
    content = m.get("content") or ""
    prev_fps.add(_msg_fingerprint(role, content))
```

**Purpose:** Prevent message duplication when persisting.

**How Fingerprinting Works:**
```python
def _msg_fingerprint(source: str, content: Any) -> str:
    h = hashlib.sha256()
    h.update((source + "\n" + content).encode("utf-8"))
    return h.hexdigest()
```
**Example:**
- Input: source="po_agent", content="Created 3 user stories"
- Output: "a1b2c3d4..." (SHA256 hash)
- Purpose: Skip persisting duplicate messages when running team.run() again

---

### Step 6: Store User Message (Line 1047)

**Code:**
```python
await chat_service.store_user_message(request.application_id, request.content)
```

**Purpose:** Record the user's new message in both history and short-term memory.

**Operations:**
```
1. INSERT into app_chat_history (full immutable audit log)
2. INSERT into app_short_memory (for LLM context)
3. AUTO-PRUNE app_short_memory to keep only last 20 rows
```

**Database Effect:**
```sql
-- 1. Append to full history
INSERT INTO app_chat_history (application_id, role, content, created_at)
VALUES (@app_id, 'user', @content, GETUTCDATE())

-- 2. Insert to STM
INSERT INTO app_short_memory (application_id, role, content, created_at)
VALUES (@app_id, 'user', @content, GETUTCDATE())

-- 3. Prune to 20 rows
DELETE FROM app_short_memory
WHERE application_id = @app_id
  AND id NOT IN (
    SELECT TOP 20 id FROM app_short_memory
    WHERE application_id = @app_id
    ORDER BY created_at DESC
  )
```

**Handles Deadlocks:** Auto-retry with exponential backoff if SQL error 40001 occurs

---

### Step 7: Load Scope (Application Boundaries) (Line 1050)

**Code:**
```python
try:
    scope = await scope_service.get_scope(request.application_id)
    proj = (scope.jira_projectkey or "").strip().upper()
    scope_kwargs = dict(
        jira_projects=[proj] if proj else [],
        jira_issues=[],
        aha_ids=[(scope.aha_id or "").strip().upper()] if getattr(scope, "aha_id", None) else [],
        confluence_spacekeys=[(scope.confluence_spacekey or "").strip().upper()] if getattr(scope, "confluence_spacekey", None) else [],
    )
except Exception:
    scope_kwargs = dict(jira_projects=[], jira_issues=[], aha_ids=[], confluence_spacekeys=[])
```

**Purpose:** Define what resources each application can access (enforcement boundaries).

**Scope Context:**
- **JIRA Projects:** Which project keys the app can work with (e.g., ["PROJ", "PROJ2"])
- **AHA IDs:** Which Aha product IDs are in scope
- **Confluence Spaces:** Which document spaces can be referenced
- Used to inject constraints into agent prompts

**If Scope Not Found:**
- Fallback to empty scope (no restrictions, but agents will warn user)
- Application continues (graceful degradation)

**Example Scope:**
```python
{
    "jira_projects": ["PROJ"],
    "aha_ids": ["AHA001"],
    "confluence_spacekeys": ["FIN"]
}
```

---

### Step 8: Build/Reuse Agents (Line 1065)

**Code:**
```python
agents = get_or_build_agents(request.application_id, scope_kwargs)
po_agent, qa_agent, tm_agent, orch_agent, critic_agent = (
    agents["po"], agents["qa"], agents["tm"], agents["orch"], agents["critic"]
)
```
**Purpose:** Create or retrieve cached agent instances.

**Agent Caching Logic:**

```python
AGENT_CACHE: dict[str, tuple[str, dict]] = {}

def get_or_build_agents(app_id: str, scope_kwargs: dict) -> Dict[str, Any]:
    scope_h = _scope_hash(scope_kwargs)  # Hash of scope for change detection
    cached = AGENT_CACHE.get(app_id)
    
    if cached:
        old_h, agents = cached
        if old_h == scope_h:
            # ✅ Scope unchanged → Use cached agents
            logger.info(f"Using cached agents for app {app_id}")
            return agents
        else:
            # ❌ Scope changed → Rebuild agents
            logger.warning(f"Scope changed! Rebuilding agents for app {app_id}")
    
    # Build new agents
    agents = {
        "po": build_product_owner_agent(application_id=app_id, **scope_kwargs),
        "qa": build_qa_agent(application_id=app_id, **scope_kwargs),
        "tm": build_test_manager_agent(application_id=app_id, **scope_kwargs),
        "orch": build_orchestrator_agent(application_id=app_id, **scope_kwargs),
        "critic": build_critic_agent(application_id=app_id, **scope_kwargs),
    }
    AGENT_CACHE[app_id] = (scope_h, agents)
    return agents
```

**Performance Benefit:**
- First request for app_id: ~2 seconds (agent initialization)
- Subsequent requests: <100ms (cached lookup)
- Cache invalidated if scope changes → agents rebuilt automatically

**Agents in the Team:**
| Agent | Role | Decides |
|-------|------|---------|
| **Orchestrator** | Coordinator | Who speaks next based on task type |
| **Product Owner (PO)** | Creates artifacts | User stories, RAID logs |
| **QA** | Tests/validates | Test cases, coverage analysis |
| **Test Manager (TM)** | Plans testing | Test strategy, automation roadmap |
| **Critic** | Final review | Quality check before delivering to user |

---

### Step 9: Build Team Selector Functions (Line 1074)

**Code:**
```python
selector_func, pick_final_output_from_last_main, MAIN_SOURCES_SET = make_selector_and_helpers(
    ORCHESTRATOR_NAME=orch_agent.name,
    CRITIC_NAME=critic_agent.name,
    PO_NAME=po_agent.name,
    QA_NAME=qa_agent.name,
    TM_NAME=tm_agent.name
)
```

**Purpose:** Create functions that decide who speaks next and how to select final output.

**Functions Returned:**

1. **selector_func(messages):** Decides next agent
   - Input: List of all messages so far
   - Output: Next agent name to speak
   - Logic:
     - If last speaker is ORCHESTRATOR → delegate to specialized agent (PO/QA/TM)
     - If last speaker is specialized agent → back to ORCHESTRATOR to decide next step
     - If CRITIC hasn't spoken yet → ORCHESTRATOR routes to CRITIC for final review

2. **pick_final_output_from_last_main:** Selects final response
   - Picks last message from main agents (PO/QA/TM/Orchestrator)
   - Skips intermediate messages
   - Returns: `{"source": agent_name, "content": response_text}`

3. **MAIN_SOURCES_SET:** Set of main agent names
   - Used to filter which agents count as "final output"

---

### Step 10: Build Team with Termination (Line 1086)

**Code:**
```python
termination = TextMentionTermination("APPROVE") | MaxMessageTermination(MAX_TURNS_PER_RUN)
backoff_model_client = BackoffModelClient(az_model_client)
team = SelectorGroupChat(
    participants=[orch_agent, po_agent, qa_agent, tm_agent, critic_agent],
    model_client=backoff_model_client,
    selector_func=selector_func,
    allow_repeated_speaker=False,
    termination_condition=termination,
)
```
**Purpose:** Initialize the multi-agent team orchestration engine.

**Termination Conditions:**
- Stop if ANY agent writes "APPROVE" in their message → Task complete
- Stop if team reaches max turns (default: 3 turns) → Time limit reached
- Combined with `|` (OR) → Either condition stops execution

**Configuration:**
- `MAX_TURNS_PER_RUN`: Environment variable (default: 3)
  - Set to: `CHAT_MAX_TURNS_PER_RUN=5` for longer conversations
- `allow_repeated_speaker=False`: Same agent can't speak twice in a row

**Backoff Client:**
- Wraps Azure OpenAI client with exponential backoff retry logic
- Auto-retries on: 429, 500, 502, 503, 504
- Does NOT retry: 401, 403 (auth errors)

---

### Step 11: Build Hybrid Context Block (Line 1102)

**Code:**
```python
context_block = await _build_context_block(seeded_rows, scope_kwargs)
```

**Purpose:** Create intelligent context message combining recent + summarized historical context.

**Strategy:**

```
Input: seeded_rows (up to 20 messages)

Split into:
├─ Recent messages: Last 5 (keep as-is, full detail)
├─ Historical messages: Messages 6-20 (summarize with LLM)
└─ Scope + Guidance

Output: context_block (structured text ~3500 tokens max)
```

**Context Block Structure:**

```
[CURRENT SCOPE - Use These Values]
JIRA Projects: PROJ
Aha IDs: AHA001
Confluence Spaces: FIN
Note: Always use above IDs from current scope...

[Earlier Conversation Summary — 15 messages]
In previous turns, we discussed...
Key decisions: ...
Action items: ...

[Recent Messages — Last 5 turns]
1) USER: Create user stories
2) ORCHESTRATOR: I'll delegate to PO...
3) PO_AGENT: Here are 3 stories...
4) CRITIC: Well-structured...
5) ORCHESTRATOR: Proceeding...

[Guidance]
You are a coordinated team. Use context faithfully...
- Prefer concise, actionable answers
- Do NOT restate entire context
- Use project IDs from CURRENT SCOPE above
```

**Token Budget:**
```python
MAX_CONTEXT_TOKENS = 3500  # Total tokens available for context
MAX_RECENT_MESSAGES = 5    # Last 5 messages (full)
MAX_HISTORICAL_MESSAGES = 15  # Messages 6-20 (summarized)
```

**Summarization:**
```python
async def _summarize_historical_messages(messages: list[dict]) -> str:
    # Uses LLM to condense messages 6-20 into ~500 words
    # Preserves key decisions, action items, requirements
    return summary_text
```

---

### Step 12: Inject Context as System Message (Line 1113)

**Code:**
```python
context_msg = TextMessage(content=context_block, source="system")

if hasattr(team, "post_message"):
    await team.post_message(context_msg)
elif hasattr(team, "add_message"):
    team.add_message(context_msg)

print(f"[CTX] Injected context system message (~{_estimate_tokens(context_block)} tokens)")
```

**Purpose:** Pre-populate team buffer with context before running.

**Why Needed:**
- Ensures all agents see the same context
- Prevents each agent from independently reading STM
- Creates single source of truth for conversation history
- Reduces token usage (summarized, not full history)

---

### Step 13: Wrap User Message (Line 1141)

**Code:**
```python
wrapped_user_prompt = f"""
Below is the recent conversation context for this application. Please use it faithfully and continue the dialogue:

{context_block} 

Now here is the user's new message. Respond only to this new message, but using the above context whenever needed.

[User Input]
{request.content}
"""

user_msg = TextMessage(content=wrapped_user_prompt.strip(), source="user")
```
**Purpose:** Combine context + new user request into single prompt.

**Result:**
- Single TextMessage that orchestrator processes
- Context already in message → agents can reference it
- Clear separation between context and new request

---

### Step 14: Run Team with Error Handling (Line 1153)

**Code:**
```python
error_occurred = False
error_message_obj = None

try:
    result = await team.run(task=user_msg)
except Exception as team_error:
    error_occurred = True
    llm_error = _classify_llm_error(team_error)
    
    if llm_error:
        # Create error TextMessage with metadata
        error_message_obj = TextMessage(
            content=llm_error.message,
            source="system",
            metadata={
                "error": "True",
                "error_type": "llm_error",
                "error_code": llm_error.status_code,
                "recoverable": llm_error.status_code in {429, 500, 502, 503, 504},
            }
        )
    else:
        error_message_obj = TextMessage(
            content="An unexpected error occurred...",
            source="system",
            metadata={"error": "True", "error_type": "unexpected_error"}
        )
    
    result = MockResult(messages=[error_message_obj])
```

**Purpose:** Execute multi-agent conversation with robust error handling.

**What Happens Inside team.run():**
1. ORCHESTRATOR reads user message
2. ORCHESTRATOR delegates to appropriate agent (PO/QA/TM)
3. That agent responds
4. ORCHESTRATOR decides next step OR stops if "APPROVE" mentioned
5. Continue until termination condition met or error occurs

**Error Classification:**

| Error | Status | Retryable? | Action |
|-------|--------|-----------|--------|
| 401 Unauthorized | 401 | ❌ No | Fail immediately (auth issue) |
| 403 Forbidden | 403 | ❌ No | Fail immediately (permission) |
| 429 Rate Limited | 429 | ✅ Yes (limited) | Retry max 1 time (from `MAX_RETRIES_RATE_LIMIT`) |
| 500 Server Error | 500 | ✅ Yes | Retry 5 times with backoff |
| 502 Bad Gateway | 502 | ✅ Yes | Retry 5 times with backoff |
| 503 Service Unavailable | 503 | ✅ Yes | Retry 5 times with backoff |
| 504 Gateway Timeout | 504 | ✅ Yes | Retry 5 times with backoff |

**Backoff Strategy:**
```python
# Exponential backoff with jitter
wait = min(base * (2 ** (attempt - 1)) * jitter, max_wait)
# Attempt 1: 0.5s
# Attempt 2: 1.0s
# Attempt 3: 2.0s
# Attempt 4: 4.0s
# Attempt 5: 8.0s
```

---

### Step 15: Save Team State (Line 1214)

**Code:**
```python
if not error_occurred:
    await save_team_state_direct(request.application_id, team, team_state_service.save)
```

**Purpose:** Persist team workflow state for resumption on next request.

**Saved Information:**
```json
{
  "application_id": "app_12345",
  "state": {
    "current_agent": "orchestrator",
    "agents_available": ["orchestrator", "po_agent", "qa_agent", "tm_agent", "critic_agent"],
    "conversation": {
      "turn": 3,
      "messages": [
        {"role": "user", "content": "Create user stories"},
        {"role": "orchestrator", "content": "I'll delegate to PO..."},
        {"role": "po_agent", "content": "Here are 3 stories..."}
      ]
    },
    "po_agent": {
      "status": "active",
      "tasks_completed": 2,
      "last_output": "Posted 3 user stories"
    },
    "metrics": {
      "total_tokens": 2300,
      "agent_breakdown": {"orchestrator": 300, "po_agent": 1500, "qa_agent": 500}
    }
  },
  "updated_at": "2026-03-25T10:30:00Z"
}
```

**Storage:**
- Table: `app_team_state`
- One row per application_id
- JSON field with full state
- Used to resume workflows if interrupted

---

### Step 16: Persist to Full History (Line 1221)

**Code:**
```python
if not error_occurred:
    await persist_run_messages(
        application_id=request.application_id,
        messages=result.messages,
        chat_service=chat_service,
        skip_fingerprints=prev_fps
    )
```
**Purpose:** Record all non-user messages to permanent audit log.

**Implementation:**
```python
async def persist_run_messages(application_id, messages, chat_service, skip_fingerprints=None):
    skip_fingerprints = skip_fingerprints or set()
    for m in messages:
        source = getattr(m, "source", None) or "unknown"
        content = _normalize_content(getattr(m, "content", None))
        
        fp = _msg_fingerprint(source, content)
        if fp in skip_fingerprints:
            continue  # Skip duplicates
        
        if source == "user":
            continue  # Already stored in step 6
        else:
            # Store to app_chat_history (full immutable log)
            await chat_service.store_agent_message(
                application_id=application_id,
                role=source,
                content=content
            )
```

**What Gets Stored:**
- ✅ Orchestrator messages
- ✅ PO Agent messages
- ✅ QA Agent messages
- ✅ Test Manager messages
- ✅ Critic Agent messages
- ✅ System error messages
- ❌ User messages (already stored)
- ❌ Duplicates (fingerprint-based dedup)

**Database Result:**
```
app_chat_history table:
│ id │ application_id │ role            │ content                    │ created_at          │
├────┼────────────────┼─────────────────┼────────────────────────────┼─────────────────────┤
│ 1  │ app_12345      │ user            │ Create user stories...     │ 2026-03-25 10:29:00 │
│ 2  │ app_12345      │ orchestrator    │ I'll delegate to PO...     │ 2026-03-25 10:29:30 │
│ 3  │ app_12345      │ po_agent        │ Here are 3 stories...      │ 2026-03-25 10:30:00 │
│ 4  │ app_12345      │ critic_agent    │ Well-structured and...     │ 2026-03-25 10:30:30 │
```

---

### Step 17: Extract Final Output (Line 1228)

**Code:**
```python
final = pick_final_output_from_last_main(result.messages)

final_content = (final.get("content") or "").strip()
final_source = "assistant"
```

**Purpose:** Select the response to return to user.

**Selection Logic:**
```python
def pick_final_output_from_last_main(messages):
    # Iterate through messages in reverse (newest first)
    for m in reversed(messages):
        source = getattr(m, "source", None)
        
        # Is this from a main agent (not system/error)?
        if source in MAIN_SOURCES_SET:  # {orchestrator, po_agent, qa_agent, tm_agent}
            return {
                "source": source,
                "content": getattr(m, "content", None),
                "mime_type": getattr(m, "mime_type", None)
            }
    
    # Fallback: return empty
    return {"source": "system", "content": "No response generated", "mime_type": None}
```

**Example:**
```
Messages: [
  user: "Create stories",
  orchestrator: "I'll delegate",
  po_agent: "Here are 3 stories",       ← Last main agent
  critic: "Well-structured"             ← Critic is not "main"
]

Result: {source: "po_agent", content: "Here are 3 stories", ...}
```

---

### Step 18: Persist Final to STM (Line 1240)

**Code:**
```python
if final_content:
    await chat_service.store_final_message(
        application_id=request.application_id,
        role=final_source,
        content=final_content
    )
```

**Purpose:** Store final response to short-term memory for next conversation turn.

**Operation:**
```sql
-- Insert final response
INSERT INTO app_short_memory (application_id, role, content, created_at)
VALUES (@app_id, 'assistant', @final_content, GETUTCDATE())

-- Auto-prune to 20 rows
DELETE FROM app_short_memory
WHERE application_id = @app_id
  AND id NOT IN (
    SELECT TOP 20 id FROM app_short_memory
    WHERE application_id = @app_id
    ORDER BY created_at DESC
  )
```

**Result:**
- STM now contains: user message + orchestrator + po response + final response
- Next request will see these 4 messages in context
- Older messages pruned away (limit 20)

---

### Step 19: Extract Critic Review (Line 1249)

**Code:**
```python
final_critic_msg_obj = None
for m in reversed(result.messages):
    if getattr(m, "source", None) == critic_agent.name:
        final_critic_msg_obj = m
        break

critic_review_str = None
if final_critic_msg_obj is not None:
    critic_review_str, _, _ = extract_payload_from_message(final_critic_msg_obj)
```

**Purpose:** Extract critic agent's quality review for inclusion in response.

**What is Critic Review:**
- Last message from critic_agent (if any)
- Extracted using `extract_payload_from_message()` helper
- Returned in `ChatOut.critic_review` field
- Used by UI to show quality assessment alongside response

---

### Step 20: Aggregate Token Usage (Line 1259)

**Code:**
```python
usage = None
token_breakdown = {}

prompt_total, completion_total = 0, 0

for m in result.messages:
    source = getattr(m, "source", None)
    mu = getattr(m, "models_usage", None)
    
    if isinstance(mu, dict):
        prompt = int(mu.get("prompt_tokens", 0))
        completion = int(mu.get("completion_tokens", 0))
        
        if prompt or completion:
            if source not in token_breakdown:
                token_breakdown[source] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            token_breakdown[source]["prompt_tokens"] += prompt
            token_breakdown[source]["completion_tokens"] += completion
            token_breakdown[source]["total_tokens"] += (prompt + completion)
            
            prompt_total += prompt
            completion_total += completion

if prompt_total or completion_total:
    usage = ModelsUsage(prompt_tokens=prompt_total, completion_tokens=completion_total)
    
    logger.info({
        "event": "token_usage",
        "total": {"prompt_tokens": prompt_total, "completion_tokens": completion_total},
        "per_agent": token_breakdown
    })
```

**Purpose:** Calculate total and per-agent token consumption.

**Output Example:**
```
{
  "total": {
    "prompt_tokens": 1500,
    "completion_tokens": 800,
    "total_tokens": 2300
  },
  "per_agent": {
    "orchestrator": {"prompt_tokens": 300, "completion_tokens": 100, "total_tokens": 400},
    "po_agent": {"prompt_tokens": 900, "completion_tokens": 600, "total_tokens": 1500},
    "critic_agent": {"prompt_tokens": 300, "completion_tokens": 100, "total_tokens": 400}
  }
}
```

**Cost Calculation (Azure OpenAI):**
```
Cost = (prompt_tokens / 1000 * prompt_price) + (completion_tokens / 1000 * completion_price)

For GPT-4:
  - Prompt: $0.03 per 1K tokens
  - Completion: $0.06 per 1K tokens
  
Cost = (1500 / 1000 * 0.03) + (800 / 1000 * 0.06)
     = 0.045 + 0.048
     = $0.093 per request
```

---

### Step 21: Build ChatOut Response (Line 1340)

**Code:**
```python
created_at = datetime.now(timezone.utc)

convo_dicts: List[dict[str, Any]] = []
for m in result.messages:
    if isinstance(m, TextMessage):
        convo_dicts.append(msg_to_dict(m))
    else:
        convo_dicts.append({
            "id": getattr(m, "id", None),
            "source": getattr(m, "source", "unknown"),
            "metadata": getattr(m, "metadata", {}) or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content": getattr(m, "content", str(m)),
            "type": getattr(m, "type", "Message"),
        })

return ChatOut(
    id=msg_id,
    source=msg_source,
    models_usage=usage,
    metadata=msg_metadata,
    created_at=created_at,
    content=final.get("content"),
    final_output=final.get("content"),
    critic_review=critic_review_str,
    data=convo_dicts,
    mime_type=final.get("mime_type"),
    type=msg_type,
)
```

**Purpose:** Format and return response to client.

**Response Fields:**
- `id`: Message ID from final agent
- `source`: Agent name ("po_agent", "orchestrator", etc.)
- `models_usage`: Token breakdown (for billing/monitoring)
- `content`: Final response text
- `data`: Full conversation for UI display
- `critic_review`: Quality feedback
- `created_at`: Timestamp
- `final_output`: Same as content (for clarity)

---

### Step 22: Error Handling (Finally Block) (Line 1370)

**Code:**
```python
except LLMAPIError as e:
    logger.error(f"LLM API Error in chat endpoint: {e.status_code} - {e.message}")
    raise

except HTTPException:
    raise

except Exception as e:
    logger.error(f"Unhandled error in chat endpoint: {str(e)}")
    traceback.print_exc()
    
    # Classify error and build user-friendly message
    error_str = str(e).lower()
    
    if "429" in error_str:
        error_message = "⏳ Too many requests. Please wait a moment and try again."
    elif "401" in error_str:
        error_message = "🔐 Authentication issue detected. Please contact your administrator."
    elif "503" in error_str:
        error_message = "🔄 Service temporarily unavailable. Please try again shortly."
    else:
        error_message = "❌ Unexpected error occurred. Please try again or contact support."
    
    return ChatOut(
        id=None,
        source="system",
        content=error_message,
        metadata={"error": "True", "error_type": error_type},
        data=[{"id": None, "source": "system", "content": error_message, "type": "TextMessage"}],
    )

finally:
    # Decrement active request counter (guaranteed cleanup)
    async with ACTIVE_REQUESTS_LOCK:
        ACTIVE_REQUESTS[request.application_id] -= 1
        if ACTIVE_REQUESTS[request.application_id] <= 0:
            del ACTIVE_REQUESTS[request.application_id]
    
    logger.info(f"[CONCURRENT] Completed request {run_id} for app {request.application_id}")
```

**Purpose:** Catch and handle all possible errors gracefully.

**Cleanup:**
- Guaranteed to decrement `ACTIVE_REQUESTS` counter
- Guaranteed to log completion
- Ensures request tracking is accurate

---

## Key Components

### 1. Context Building Strategy

**Hybrid Approach (Recent + Summarized Historical):**

```
Scenario: 25 messages in STM (more than 20-message limit)

Step 1: Load last 20 messages
  Messages 1-25 → Select messages 6-25 (last 20)

Step 2: Split into recent + historical
  Recent (last 5):
    ├─ Message 21: USER: "Create user stories"
    ├─ Message 22: ORCH: "I'll delegate..."
    ├─ Message 23: PO: "Here are 3 stories"
    ├─ Message 24: CRITIC: "Well-structured..."
    └─ Message 25: ORCH: "Proceeding..."
  
  Historical (6-20):
    ├─ Messages 6-20: "In earlier turns, we..."
    └─ Summarized by LLM into ~500 words

Step 3: Assemble context block
  [CURRENT SCOPE] → Always included
  [Earlier Summary] → If historical messages exist
  [Recent Messages] → Always included (5 messages)
  [Guidance] → Behavioral instructions

Step 4: Token budget enforcement
  Total tokens: ~3500
  If exceeds → Truncate earliest messages
```

**Why This Approach:**

| Aspect | Full History | STM Only | Hybrid (Recent + Summarized) |
|--------|--------------|----------|------------------------------|
| **Recency** | ❌ Stale | ✅ Fresh | ✅ Fresh (recent in full) |
| **Context** | ✅ Complete | ❌ Limited | ✅ Complete (summarized) |
| **Tokens** | ❌ Expensive | ✅ Cheap | ✅ Balanced (~3500) |
| **Decisions** | ✅ All available | ❌ Lost | ✅ Key decisions preserved |

---

### 2. Multi-Agent Orchestration

**SelectorGroupChat Flow:**

```
User Message Arrives
    ↓
Orchestrator Receives Message
    ├─ Analyzes task type
    ├─ Decides: "This is a PO task"
    ├─ Selects: PO_AGENT
    └─ Delegates with context
        ↓
    PO Agent Processes
    ├─ Reads context (STM + scope)
    ├─ Performs work (JIRA API calls, etc.)
    ├─ Generates response
    └─ Returns to orchestrator
        ↓
    Orchestrator Decides Next Step
    ├─ Check: "Is task complete?"
    ├─ Option 1: "Yes → Route to CRITIC"
    ├─ Option 2: "No → Route to QA for validation"
    └─ Option 3: "Complex → Route to TM for planning"
        ↓
    Critic Agent Reviews
    ├─ Evaluates quality
    ├─ Checks compliance
    ├─ Approves or suggests changes
    └─ Final assessment
        ↓
    Response Sent to User
```

---

### 3. Database Integration

**Message Storage Pattern:**

```
Request Arrives with user_message

├─ Step 1: store_user_message()
│   ├─ INSERT into app_chat_history (full immutable)
│   ├─ INSERT into app_short_memory
│   └─ AUTO-PRUNE app_short_memory to 20 rows
│
├─ Step 2-8: Team processes request
│   └─ No DB writes (in-memory only)
│
├─ Step 9: persist_run_messages() [non-user messages]
│   └─ INSERT into app_chat_history (orchestrator, agents, etc.)
│
└─ Step 10: store_final_message()
    ├─ INSERT final response into app_short_memory
    └─ AUTO-PRUNE app_short_memory to 20 rows

Result:
✅ app_chat_history: Complete audit log (all messages)
✅ app_short_memory: Recent context (20 messages, auto-maintained)
```

---

## Context Building Strategy

See detailed section above in "Step 11: Build Hybrid Context Block"

Key Points:
- Last 5 messages: Full detail (for immediate context)
- Messages 6-20: LLM-summarized (for historical understanding)
- Scope injection: Current boundaries always visible
- Token limit: 3500 tokens maximu



## Error Handling

### Error Classification System

```python
def _classify_llm_error(e: Exception) -> Optional[LLMAPIError]:
    """Classify exception and return appropriate error type"""
    
    status = extract_status_code(e)
    error_msg = str(e).lower()
    
    # 401 - Auth Error
    if status == 401 or "unauthorized" in error_msg:
        return UnauthorizedError("Authentication failed...")  # ❌ NON-RETRYABLE
    
    # 403 - Permission Error
    if status == 403 or "forbidden" in error_msg:
        return ForbiddenError("Access denied...")  # ❌ NON-RETRYABLE
    
    # 429 - Rate Limit
    if status == 429 or "rate limit" in error_msg:
        return RateLimitError("Rate limited...", retry_after=30)  # ✅ RETRYABLE (limited)
    
    # 503 - Service Unavailable
    if status == 503:
        return ModelUnavailableError("Service unavailable...")  # ✅ RETRYABLE
    
    # 500/502/504 - Server Errors
    if status in {500, 502, 504}:
        return ModelUnavailableError(f"Server error HTTP {status}...")  # ✅ RETRYABLE
```

### Retry Strategy

**Exponential Backoff with Jitter:**

```python
async def call_with_backoff_async(create_call, *, max_retries=5, base=1.0, **kwargs):
    attempt = 0
    
    while True:
        try:
            return await create_call(**kwargs)
        
        except Exception as e:
            llm_error = _classify_llm_error(e)
            
            # Non-retryable errors (401, 403) → Fail immediately
            if llm_error and llm_error.status_code in {401, 403}:
                raise llm_error
            
            # Retryable errors with limited retries for 429
            if llm_error and llm_error.status_code == 429:
                max_retries = MAX_RETRIES_RATE_LIMIT  # Usually 1-2
            
            # Check if we should retry
            if status in RETRYABLE_STATUS and attempt < max_retries:
                attempt += 1
                
                # Calculate wait time
                wait = extract_retry_after_header(e)  # Check response header
                if wait is None:
                    # Exponential backoff: 2^(attempt-1) * base_time
                    wait = base * (2 ** (attempt - 1)) * random_jitter()
                    wait = min(wait, max_wait=60.0)
                
                logger.warning(f"Retry {attempt}/{max_retries}, waiting {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            
            # Exhausted retries → raise
            raise
```

**Retry Examples:**

| Scenario | Status | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 5 | Result |
|----------|--------|-----------|-----------|-----------|-----------|-----------|--------|
| Server Error | 500 | Retry @1s | Retry @2s | Retry @4s | Retry @8s | Retry @16s | ✅ Success on attempt 3 |
| Rate Limited | 429 | Retry @1s | Fail | N/A | N/A | N/A | ❌ Fail after 1 retry |
| Auth Failed | 401 | Fail immediately | N/A | N/A | N/A | N/A | ❌ Fail immediately |
| Bad Request | 400 | Fail immediately | N/A | N/A | N/A | N/A | ❌ Fail immediately |

---

## Multi-User Concurrency

### Request Tracking (No Throttling)

```python
# Global counter
ACTIVE_REQUESTS = defaultdict(int)  # app_id → count
ACTIVE_REQUESTS_LOCK = asyncio.Lock()

# When request arrives
async with ACTIVE_REQUESTS_LOCK:
    ACTIVE_REQUESTS[app_id] += 1
    active_count = ACTIVE_REQUESTS[app_id]

logger.info(f"Processing app {app_id} (active: {active_count})")

# Endpoint: /metrics/rate-limit
GET /metrics/rate-limit
Response:
{
  "active_requests": {
    "total": 25,
    "by_app": {
      "app_12345": 5,
      "app_67890": 3,
      "app_11111": 17
    }
  }
}
```

### True Concurrent Access

**No Semaphores:**
- ✅ Multiple users for same app_id → All run in parallel
- ✅ Multiple apps → All run in parallel
- ✅ Database connection pool handles concurrency
- ✅ Deadlock retry in repository handles race conditions

**Scaling:**
- Default: Unlimited concurrent requests
- Bottleneck: Azure OpenAI rate limits (not the endpoint)
- Database: Connection pool (pool_size=5, max_overflow=10 = 15 total connections)

---

## Database Interactions

### Tables Used

| Table | Purpose | Writes From | Reads From |
|-------|---------|-------------|-----------|
| `app_chat_history` | Full immutable audit log | store_user_message(), persist_run_messages() | /history endpoint |
| `app_short_memory` | Recent context (20 rows) | store_user_message(), store_final_message() | Step 4 (context loading) |
| `app_team_state` | Team workflow state | save_team_state_direct() | (optional) load_team_state() |

### Query Patterns

**Load STM:**
```sql
SELECT id, role, content, created_at FROM app_short_memory
WHERE application_id = @app_id
ORDER BY created_at ASC
```

**Store User Message:**
```sql
-- 1. Append to history
INSERT INTO app_chat_history (application_id, role, content, created_at)
VALUES (@app_id, 'user', @content, GETUTCDATE())

-- 2. Append to STM
INSERT INTO app_short_memory (application_id, role, content, created_at)
VALUES (@app_id, 'user', @content, GETUTCDATE())

-- 3. Prune to 20 rows
DELETE FROM app_short_memory
WHERE application_id = @app_id AND id NOT IN (
  SELECT TOP 20 id FROM app_short_memory
  WHERE application_id = @app_id
  ORDER BY created_at DESC
)
```

**Store Final Message:**
```sql
-- 1. Append to STM
INSERT INTO app_short_memory (application_id, role, content, created_at)
VALUES (@app_id, 'assistant', @final_content, GETUTCDATE())

-- 2. Prune to 20 rows (same as above)
```

---

## Token Usage Tracking

### Token Breakdown

**Per-Agent Consumption:**
```
Typical request:
├─ Orchestrator: 400 tokens (route selection logic)
├─ PO Agent: 1500 tokens (create user stories)
├─ QA Agent: 300 tokens (validation)
├─ Critic Agent: 400 tokens (quality review)
└─ Total: 2600 tokens

Cost calculation (GPT-4):
  Prompt: 1500 tokens × $0.03 / 1000 = $0.045
  Completion: 1100 tokens × $0.06 / 1000 = $0.066
  Total: $0.111 per request
```

### Monitoring

**Via Logs:**
```
[TOKEN_USAGE] Total: 2600 tokens (prompt: 1500, completion: 1100)
  └─ orchestrator: 400 tokens
  └─ po_agent: 1500 tokens
  └─ qa_agent: 300 tokens
  └─ critic_agent: 400 tokens
```

**Via Response:**
```json
{
  "models_usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 1100
  }
}
```

**Optimization Tips:**
```
1. Reduce MAX_RECENT_MESSAGES from 5 → 3 (saves ~500 tokens)
2. Reduce MAX_HISTORICAL_MESSAGES from 15 → 10 (saves ~200 tokens)
3. Reduce MAX_CONTEXT_TOKENS from 3500 → 2500 (hard limit)
4. Reduce MAX_TURNS_PER_RUN from 3 → 2 (less agent interaction)
```

---

## Configuration Parameters

All environment variables:

| Variable | Default | Purpose | Range |
|----------|---------|---------|-------|
| `CHAT_MAX_TURNS_PER_RUN` | 3 | Max agent interactions | 1-10 |
| `CHAT_MAX_RECENT_MESSAGES` | 5 | Recent context (full) | 1-10 |
| `CHAT_MAX_HISTORICAL_MESSAGES` | 15 | Historical context (summarized) | 5-30 |
| `CHAT_MAX_CONTEXT_TOKENS` | 3500 | Total context token budget | 1000-10000 |
| `CHAT_ENABLE_RATE_LIMITER` | 0 | Enable token bucket limiter | 0 or 1 |
| `CHAT_LOG_MEMORY` | 0 | Debug STM loading | 0 or 1 |
| `CHAT_MEMORY_LOG_MAX_CHARS` | 1000 | Max chars per STM debug log | 100-5000 |
| `CHAT_LOG_NO_TRUNCATE` | 0 | Disable log truncation | 0 or 1 |
| `MAX_RETRIES_RATE_LIMIT` | 1 | Max 429 retries | 1-3 |
| `LOG_LEVEL` | INFO | Logging level | DEBUG, INFO, WARNING, ERROR |

### Example .env Configuration

```bash
# Context strategy
CHAT_MAX_TURNS_PER_RUN=3
CHAT_MAX_RECENT_MESSAGES=5
CHAT_MAX_HISTORICAL_MESSAGES=15
CHAT_MAX_CONTEXT_TOKENS=3500

# Error handling
MAX_RETRIES_RATE_LIMIT=1
CHAT_ENABLE_RATE_LIMITER=0

# Debugging
CHAT_LOG_MEMORY=0
LOG_LEVEL=INFO
```

---

## Debugging & Monitoring

### Debug Endpoints

**1. Health Check:**
```bash
GET /health
Response: {"status": "ok"}
```

**2. Metrics - Rate Limiting & Concurrency:**
```bash
GET /metrics/rate-limit
Response:
{
  "active_requests": {
    "total": 12,
    "by_app": {"app_123": 5, "app_456": 7}
  },
  "rate_limiter": {
    "enabled": false,
    "tokens_available": null
  }
}
```

**3. Metrics - Token Usage Info:**
```bash
GET /metrics/tokens
Response:
{
  "azure_pricing_reference": {
    "gpt-4": {"prompt": 0.03, "completion": 0.06}
  },
  "optimization_tips": [...],
  "current_config": {...}
}
```

**4. History:**
```bash
GET /history/{application_id}
Response: [{"role": "user", "content": "...", "created_at": "..."}, ...]
```

### Debug Flags

**Enable Memory Logging:**
```bash
CHAT_LOG_MEMORY=1 python demo.py
```

Output:
```
{
  "event": "short_memory_loaded",
  "run_id": "abc-123",
  "application_id": "app_12345",
  "total": 15,
  "head": [
    {"role": "user", "content": "Create stories", "created_at": "2026-03-25T10:29:00Z"},
    ...
  ],
  "tail": [
    {"role": "critic", "content": "Well-structured", "created_at": "2026-03-25T10:30:00Z"}
  ]
}
```

**Print Debug Info:**
```python
# In code: Set LOG_LEVEL=DEBUG
logger.debug(f"[DEBUG] STM loaded: {len(short_memory)} rows")
```

### Common Debug Strings in Logs

| String | Meaning |
|--------|---------|
| `[CONCURRENT]` | Request lifecycle (start/end) |
| `[STM_DEBUG]` | Short-term memory operations |
| `[CTX]` | Context block assembly |
| `[TEAM_BUFFER]` | Internal team message buffer |
| `[RUN]` | Team execution (team.run) |
| `[TOKEN_USAGE]` | Token consumption breakdown |
| `[VALIDATION_FAILED]` | Input validation error |
| `[CACHE_*]` | Agent cache operations |

---

## Common Issues & Solutions

### Issue 1: "STM not pruning" (more than 20 rows)

**Symptom:**
```
[STM_DEBUG] loaded_from_db=25 rows
```

**Cause:** Deadlock or transaction failure in prune operation.

**Solution:**
```python
# Check database logs for error 40001
SELECT * FROM sys.event_log WHERE error_number = 40001

# Manually clean:
DELETE FROM app_short_memory
WHERE application_id = 'app_id'
  AND id NOT IN (
    SELECT TOP 20 id FROM app_short_memory
    WHERE application_id = 'app_id'
    ORDER BY created_at DESC
  )

# Verify: Should be exactly 20 rows
SELECT COUNT(*) FROM app_short_memory WHERE application_id = 'app_id'  -- Should be 20
```

---

### Issue 2: "Timeout acquiring connection"

**Symptom:**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached
```

**Cause:** All 15 connection pool slots in use, new request waits > 30s.

**Solution:**
```python
# Increase pool size in session.py:
engine = create_async_engine(
    ASYNC_DB_URL,
    pool_size=10,        # ← Increase from 5
    max_overflow=15,     # ← Increase from 10
    pool_pre_ping=True
)

# Or reduce concurrent requests via rate limiter:
CHAT_ENABLE_RATE_LIMITER=1 python demo.py
```

---

### Issue 3: "Rate limit error 429" frequently

**Symptom:**
```
[LLM error during team.run()] 429 - Rate limit exceeded
```

**Cause:** Azure OpenAI hitting rate limits.

**Solution:**
```bash
# Option 1: Reduce tokens per request
CHAT_MAX_CONTEXT_TOKENS=2000 python demo.py

# Option 2: Reduce agent turns
CHAT_MAX_TURNS_PER_RUN=2 python demo.py

# Option 3: Enable rate limiter (queue requests)
CHAT_ENABLE_RATE_LIMITER=1 python demo.py

# Option 4: Request higher quota from Azure
# Contact Azure Support to increase TPM (tokens per minute) limit
```

---

### Issue 4: "No response generated" error

**Symptom:**
```json
{
  "content": "No response generated",
  "metadata": {"error_type": "no_response"}
}
```

**Cause:** Team.run() completed but no messages returned.

**Solution:**
```python
# Increase max turns to give team more chance
CHAT_MAX_TURNS_PER_RUN=5 python demo.py

# Check agent prompts are correct:
# See src/agents/ files for any prompt issues

# Check LLM responses aren't empty:
# Enable debug logging
LOG_LEVEL=DEBUG python demo.py
```

---

### Issue 5: "Scope not found" for new application

**Symptom:**
```
HTTPException: 404 - Scope not found for app_12345: ...
```

**Cause:** Application not registered in scope tables.

**Solution:**
```sql
-- Add application to scope tables:
INSERT INTO applicationonboarding$applicationonboarding (appid, appname)
VALUES ('app_12345', 'My Application')

INSERT INTO [applicationonboarding$tool] (appid, toolkey, toolid)
VALUES ('app_12345', 'JIRA', 'PROJ')  -- JIRA project key

INSERT INTO [applicationonboarding$keyid] (appid, idtype, idvalue)
VALUES ('app_12345', 'AHA', 'AHA001')  -- AHA ID
```

---

### Issue 6: "Duplicate messages" in history

**Symptom:**
```
app_chat_history has multiple identical messages
```

**Cause:** Fingerprint logic failing or bypassed.

**Solution:**
```python
# Check fingerprint calculation:
def _msg_fingerprint(source: str, content: Any) -> str:
    # Should use SHA256, not simple equality
    import hashlib
    h = hashlib.sha256()
    h.update((source + "\n" + str(content)).encode("utf-8"))
    return h.hexdigest()

# Manual cleanup:
DELETE FROM app_chat_history
WHERE application_id = 'app_id'
  AND id NOT IN (
    SELECT MAX(id) FROM app_chat_history
    WHERE application_id = 'app_id'
    GROUP BY role, content
  )
```

---

### Issue 7: "Context truncated" warning

**Symptom:**
```
logger.warning(f"Context block truncated from {tokens} to ~{MAX_CONTEXT_TOKENS} tokens")
```

**Cause:** Too many messages in STM exceeding token budget.

**Solution:**
```bash
# Reduce context budget
CHAT_MAX_CONTEXT_TOKENS=5000 python demo.py

# Or reduce messages included
CHAT_MAX_RECENT_MESSAGES=3 python demo.py
CHAT_MAX_HISTORICAL_MESSAGES=10 python demo.py
```

---

### Issue 8: "Agent cache stale after scope change"

**Symptom:**
```
Agent behaves with old scope after updating scope table
```

**Cause:** Agent cached with old scope, scope change not detected.

**Solution:**
```bash
# Clear cache manually:
POST /clear-cache/app_12345
Response: {"status": "success", "message": "Agent cache cleared for application app_12345"}

# Agents rebuilt on next request with new scope
```

---

End of Documentation
 