# Chat Endpoint - Visual Flowcharts & Diagrams

## 1. Complete Request-Response Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER SENDS /chat REQUEST                            │
│                    POST /chat with content + app_id                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ 1. VALIDATE INPUT             │
                    │ ├─ Is content empty?          │
                    │ ├─ Is app_id present?         │
                    │ └─ Response Type: ChatOut     │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │ 2. TRACK REQUEST                 │
                    │ ├─ Increment ACTIVE_REQUESTS     │
                    │ ├─ Generate run_id (UUID)        │
                    │ └─ Start timing (t0)             │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │ 3. LOAD SHORT-TERM MEMORY (STM)          │
                    │ ├─ Query app_short_memory for app_id    │
                    │ ├─ Return: Last 20 messages (if exist)   │
                    │ ├─ Order: oldest → newest               │
                    │ └─ Database: MSSQL Read                 │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │ 4. STORE USER MESSAGE                │
                    │ ├─ INSERT into app_chat_history      │
                    │ ├─ INSERT into app_short_memory      │
                    │ ├─ AUTO-PRUNE STM to 20 rows         │
                    │ └─ Database: MSSQL Write + Prune     │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │ 5. LOAD APPLICATION SCOPE                │
                    │ ├─ Query scope service for app_id       │
                    │ ├─ Get JIRA projects, AHA IDs, etc.     │
                    │ ├─ Normalize to uppercase              │
                    │ └─ Fallback: empty scope if not found   │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────────┐
                    │ 6. BUILD/REUSE AGENTS                             │
                    │ ├─ Check agent cache for app_id                 │
                    │ ├─ If cached + scope unchanged:                 │
                    │ │   └─ Reuse agents (~100ms)                    │
                    │ ├─ If not cached or scope changed:              │
                    │ │   ├─ Build 5 new agents (~2s)                 │
                    │ │   ├─ Orchestrator, PO, QA, TM, Critic        │
                    │ │   └─ Cache with scope hash                    │
                    │ └─ Return: dict of agents                       │
                    └───────────────┬──────────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │ 7. SETUP TEAM ORCHESTRATION                 │
                    │ ├─ Create selector_func (who speaks next?)  │
                    │ ├─ Create termination conditions:           │
                    │ │   ├─ "APPROVE" mentioned → Stop          │
                    │ │   └─ Max {3} turns → Stop                │
                    │ ├─ Build SelectorGroupChat team            │
                    │ └─ Add backoff-enabled model client        │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────────┐
                    │ 8. BUILD HYBRID CONTEXT BLOCK                    │
                    │ ├─ Split STM into:                              │
                    │ │   ├─ Recent (last 5): Full content            │
                    │ │   └─ Historical (6-20): LLM-summarized        │
                    │ ├─ Add Current Scope section                    │
                    │ ├─ Add Guidance section                         │
                    │ ├─ Enforce token budget (3500 max)              │
                    │ └─ Return: Formatted context text               │
                    └───────────────┬──────────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │ 9. WRAP USER MESSAGE                 │
                    │ ├─ Prepend context block             │
                    │ ├─ Append new user request           │
                    │ ├─ Create TextMessage("user")        │
                    │ └─ Result: Single wrapped message    │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────▼────────────────────────────────────┐
        │ 10. RUN MULTI-AGENT TEAM (with error handling)                 │
        │     ┌───────────────────────────────────────────────┐          │
        │     │ try:                                          │          │
        │     │   result = await team.run(task=wrapped_msg)   │          │
        │     └───────────────────────┬───────────────────────┘          │
        │                             │                                  │
        │     ┌───────────────────────▼──────────────────────┐           │
        │     │ TEAM ORCHESTRATION LOOP:                    │           │
        │     │ ├─ Orchestrator reads message               │           │
        │     │ ├─ Selects: PO Agent / QA / TM / Critic    │           │
        │     │ ├─ Agent responds                           │           │
        │     │ ├─ Check termination? (APPROVE or max=3)    │           │
        │     │ ├─ If not terminated: back to step 1        │           │
        │     │ └─ Return: result.messages (all outputs)    │           │
        │     └───────────────────────┬──────────────────────┘           │
        │                             │                                  │
        │     ┌───────────────────────▼──────────────────────┐           │
        │     │ except Exception as team_error:             │           │
        │     │ ├─ _classify_llm_error(team_error)          │           │
        │     │ ├─ 401/403? → Fail immediately              │           │
        │     │ ├─ 429? → Retry 1-2 times (limited)         │           │
        │     │ ├─ 500/502/503/504? → Retry 5 times         │           │
        │     │ ├─ Create error TextMessage                 │           │
        │     │ └─ result = MockResult([error_message])     │           │
        │     └───────────────────────┬──────────────────────┘           │
        └───────────────────────────┬─────────────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │ 11. SAVE TEAM STATE (if no error)        │
                    │ ├─ Serialize team.messages               │
                    │ ├─ Save to app_team_state (JSON)         │
                    │ └─ Database: MSSQL Write                │
                    └───────────────┬──────────────────────────┘
                                    │
                                                      ┌───────────────▼──────────────────────────────┐
                    │ 12. PERSIST TO FULL HISTORY (if no error)    │
                    │ ├─ For each message (not user):             │
                    │ │   ├─ Generate fingerprint                 │
                    │ │   ├─ Skip if duplicate (in prev_fps)      │
                    │ │   └─ INSERT into app_chat_history         │
                    │ └─ Database: MSSQL Write (bulk)            │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │ 13. EXTRACT FINAL OUTPUT                 │
                    │ ├─ pick_final_output_from_last_main()   │
                    │ ├─ Find last message from main agent    │
                    │ ├─ Return: {source, content}             │
                    │ └─ Skip intermediate messages           │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │ 14. PERSIST FINAL TO STM (only)              │
                    │ ├─ INSERT final response into STM            │
                    │ ├─ AUTO-PRUNE STM to 20 rows                │
                    │ └─ Database: MSSQL Write + Prune           │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │ 15. EXTRACT CRITIC REVIEW                │
                    │ ├─ Find critic_agent message             │
                    │ ├─ Extract payload (quality feedback)    │
                    │ └─ Store in critic_review field          │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │ 16. AGGREGATE TOKEN USAGE                     │
                    │ ├─ Sum all prompt_tokens from messages       │
                    │ ├─ Sum all completion_tokens from messages   │
                    │ ├─ Calculate per-agent breakdown             │
                    │ ├─ Log token usage metrics                   │
                    │ └─ Return: ModelsUsage object                │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │ 17. BUILD ChatOut RESPONSE                    │
                    │ ├─ id = final message ID                     │
                    │ ├─ source = final agent name                 │
                    │ ├─ content = final response text             │
                    │ ├─ data = all conversation messages          │
                    │ ├─ models_usage = token breakdown            │
                    │ ├─ critic_review = quality feedback          │
                    │ ├─ created_at = now (UTC)                    │
                    │ └─ Return: ChatOut object                    │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │ 18. FINALLY BLOCK (cleanup)              │
                    │ ├─ Decrement ACTIVE_REQUESTS             │
                    │ ├─ Log [CONCURRENT] completion           │
                    │ └─ Guarantee cleanup (async safe)        │
                    └───────────────┬──────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────┐
│                         RESPONSE SENT TO CLIENT                      │
│                      HTTP 200 + ChatOut (JSON)                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ {                                                           │   │
│  │   "content": "Final response text...",                      │   │
│  │   "source": "po_agent",                                     │   │
│  │   "final_output": "Final response text...",                 │   │
│  │   "critic_review": "Well-structured...",                    │   │
│  │   "models_usage": {"prompt_tokens": 1500, ...},             │   │
│  │   "data": [{...all messages...}],                           │   │
│  │   "created_at": "2026-03-25T10:30:00Z"                      │   │
│  │ }                                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Selection Logic

```
                    User Message
                          ↓
                  ORCHESTRATOR AGENT
                          │
             ┌────────────┼────────────┐
             │            │            │
        Is it about   Is it about   Is it about
        specs/stories? testing?      test planning?
             │            │            │
             YES           YES          YES
             │            │            │
             ▼            ▼             ▼
        ┌─────────┐  ┌──────────┐  ┌──────────────┐
        │ PO AGENT│  │ QA AGENT │  │ TM AGENT     │
        └────┬────┘  └─────┬────┘  └──────┬───────┘
             │             │               │
      Create User       Validate      Plan Testing
      Stories          Requirements   Strategy
             │             │               │
             └─────────────┬───────────────┘
                           │
                ORCHESTRATOR REVIEWS
                           │
             ┌─────────────┴──────────────┐
             │                            │
        Task Complete?            More Work Needed?
             │                            │
             YES                          NO
             │                            │
             ▼                            ▼
        SEND TO CRITIC            REPEAT (Select Next Agent)
             │                            │
             └────────┬───────────────────┘
                      │
            ┌─────────▼────────────┐
            │ CRITIC AGENT REVIEWS  │
            │ ├─ Quality check     │
            │ ├─ Compliance check  │
            │ └─ Final approval    │
            └─────────┬────────────┘
                      │
            ┌─────────▼────────────┐
            │ FINAL RESPONSE READY │
            │ Return to User       │
            └──────────────────────┘
```

---

## 3. Context Building Strategy

```
INPUT: Short-Term Memory (up to 20 messages)
┌─────────────────────────────────────────────┐
│ Messages by created_at:                     │
│  1. User: "Create specs"                    │
│  2. Orch: "I'll delegate"                   │
│  3. PO: "Here's a draft"                    │
│  ...                                        │
│ 15. User: "Make it testable"               │
│ 16. Orch: "Routing to QA"                  │
│ 17. QA: "I'll validate"                    │
│ 18. PO: "Updated specs here"               │
│ 19. Critic: "Looks good"                   │
│ 20. Orch: "Proceeding"                     │
└─────────────────────────────────────────────┘
             │
             ▼
    SPLIT INTO TWO GROUPS
             │
    ┌────────┴────────┐
    │                 │
    ▼ (Last 5)        ▼ (6-20, 15 messages)
┌──────────────┐  ┌───────────────────────────┐
│ RECENT       │  │ HISTORICAL (for summary)  │
├──────────────┤  ├───────────────────────────┤
│ 16. Orch: ...│  │ Messages 1-15 combined    │
│ 17. QA: ...  │  │                           │
│ 18. PO: ...  │  │ Fed to LLM for            │
│ 19. Critic...│  │ summarization (~500 words)│
│ 20. Orch: ...│  │                           │
└──────────────┘  │ Output: "In earlier      │
                  │ turns, we discussed... " │
                  └───────────────────────────┘
             │
             ▼
    BUILD CONTEXT BLOCK
    ┌─────────────────────────────────────┐
    │ [CURRENT SCOPE]                     │
    │ JIRA: PROJ                          │
    │ AHA: AHA001                         │
    │ Confluence: FIN                     │
    │                                     │
    │ [Earlier Summary — 15 messages]     │
    │ In earlier turns, we discussed:     │
    │ - Authentication module design      │
    │ - Key decisions: Use OAuth 2.0      │
    │ - Action items: Create stories      │
    │                                     │
    │ [Recent Messages — Last 5 turns]    │
    │ 1) Orch: "Routing to QA"           │
    │ 2) QA: "I'll validate..."          │
    │ 3) PO: "Updated specs here"        │
    │ 4) Critic: "Looks good"            │
    │ 5) Orch: "Proceeding"              │
    │                                     │
    │ [Guidance]                          │
    │ You are a coordinated team...       │
    │                                     │
    │ Token Count: ~2800 / 3500 budget    │
    └─────────────────────────────────────┘
             │
             ▼
    INJECT INTO TEAM.run()
```

---
## 4. Error Handling Decision Tree

```
EXCEPTION OCCURS IN team.run()
        │
        ▼
_classify_llm_error(exception)
        │
    ┌───┴────────────────────────────────────────┐
    │                                            │
    ▼ Extract HTTP Status Code                   ▼ Check Error Message
    
    401?    └─ "Unauthorized"
    403?    └─ "Forbidden"
    429?    └─ "Rate limit"
    500/502?└─ "Server error"
    503?    └─ "Unavailable"
    
        │
        ├──────────────────┬─────────────────┐
        │                  │                 │
    401/403 (Auth)      429 (Rate)      5xx (Server)
        │                  │                 │
        ▼                  ▼                 ▼
    ❌ FAIL             ✅ RETRY          ✅ RETRY
    IMMEDIATELY         Limited           Full Backoff
        │               (1-2 times)      (5 times)
        │                  │                 │
        │              Wait: 1s          Wait: 1s→2s→4s→8s→16s
        │                  │                 │
        ├──────────────────┼─────────────────┤
        │                  │                 │
        └──────────┬───────┴─────────────────┘
                   │
            ┌──────▼──────┐
            │ Still error?│
            ├─────┬────────┤
            │     NO      │ YES
            ▼             ▼
         ✅ SUCCESS   ❌ FAILED
         (Retry      (Return
          worked)     to user)
            │             │
            │    ┌────────┴──────────┐
            │    │                   │
            │    ▼                   ▼
            │  Create Error    Create Error
            │  TextMessage     ChatOut
            │  (in result)     (final response)
            │    │                   │
            └────┴───────────────────┘
                 │
         Continue with
         persistence
         (Steps 13-17)
```

---

## 5. Database Write Pattern

```
REQUEST LIFECYCLE - DATABASE OPERATIONS
│
├─ STEP 1: Store User Message
│  ├─ INSERT app_chat_history
│  │  └─ role='user', content=...
│  │
│  ├─ INSERT app_short_memory
│  │  └─ role='user', content=...
│  │
│  └─ DELETE (prune to 20)
│     └─ Keep only TOP 20 by created_at DESC
│        Result: STM now has ≤20 rows
│
├─ STEP 2: Team processes (in-memory, no DB writes)
│  └─ No persistence during orchestration
│
├─ STEP 3: Persist Non-User Messages
│  ├─ For each message (not user):
│  │  ├─ Calculate fingerprint
│  │  ├─ If NOT in prev_fps:
│  │  │  └─ INSERT app_chat_history
│  │  └─ Else: Skip (duplicate)
│  │
│  └─ Result: Full history log appended
│
├─ STEP 4: Save Team State
│  ├─ Serialize team.messages to JSON
│  └─ UPSERT app_team_state
│     └─ One row per app_id (overwrite previous)
│
└─ STEP 5: Persist Final Output
   ├─ INSERT app_short_memory
   │  └─ role='assistant', content=final_response
   │
   ├─ DELETE (prune to 20)
   │  └─ Keep only TOP 20 by created_at DESC
   │     Result: STM back to ≤20 rows
   │
   └─ Result: Next request sees these 20 messages


FINAL STATE:
┌────────────────────────────────────────────────────────┐
│ app_chat_history                                       │
├────────────────────────────────────────────────────────┤
│ id │ app_id    │ role      │ content          │ time    │
├────┼───────────┼───────────┼──────────────────┼─────────┤
│ 1  │ app_123   │ user      │ Create stories   │ 10:29:00│
│ 2  │ app_123   │ orch      │ I'll delegate    │ 10:29:30│
│ 3  │ app_123   │ po_agent  │ Here's a draft   │ 10:30:00│
│ 4  │ app_123   │ critic    │ Looks good       │ 10:30:30│
│ 5  │ app_123   │ orch      │ Proceeding       │ 10:31:00│
│    │ ...       │ ...       │ ...              │ ...     │
└────┴───────────┴───────────┴──────────────────┴─────────┘

┌────────────────────────────────────────────────────────┐
│ app_short_memory (exactly 20 rows per app)             │
├────────────────────────────────────────────────────────┤
│ id │ app_id    │ role      │ content          │ time    │
├────┼───────────┼───────────┼──────────────────┼─────────┤
│ 1  │ app_123   │ user      │ Create stories   │ 10:29:00│
│ 2  │ app_123   │ orch      │ I'll delegate    │ 10:29:30│
│ 3  │ app_123   │ po_agent  │ Here's a draft   │ 10:30:00│
│ 4  │ app_123   │ critic    │ Looks good       │ 10:30:30│
│ 5  │ app_123   │ orch      │ Proceeding       │ 10:31:00│
│    │           │           │                  │         │
│ ...│ ...       │ ...       │ (14 more rows)   │ ...     │
│ 20 │ app_123   │ assistant │ Final response   │ 10:31:30│
└────┴───────────┴───────────┴──────────────────┴─────────┘
```

---
## 6. Concurrent Request Timeline

```
TIME    APP_123          APP_456          APP_789
────────────────────────────────────────────────────
10:00   Request 1a
        (running)
        
10:00.5              Request 2a
                     (running)
        
10:01   Request 1b   Request 2b        Request 3a
        (running)    (running)         (running)
        
        ↓            ↓                  ↓
        Agents       Agents             Agents
        Processing   Processing         Processing
        (parallel)   (parallel)         (parallel)
        ↓            ↓                  ↓
        
10:02   Response 1b✅ Response 2a✅     Response 3a✅
        COMPLETED     COMPLETED        COMPLETED
        
10:02.5 Request 1c
        (running)
        
        DB Writes:
        - app_123: 2 users (1b, 1c concurrent)
        - app_456: 1 user (2a completed)
        - app_789: 1 user (3a completed)

ACTIVE_REQUESTS at 10:02:
{
  "app_123": 2,    ← User 1b + 1c running in parallel
  "app_456": 1,    ← User 2b
  "app_789": 1     ← User 3a
}

TOTAL ACTIVE: 4 concurrent requests
NO THROTTLING: All run at full speed
```

---

## 7. Agent Cache Lifecycle

```
APP_ID: app_123
SCOPE: {jira: ["PROJ"], aha: ["AHA001"]}

┌──────────────────────────────────────────────────────────┐
│ REQUEST 1 (app_123, same scope)                          │
├──────────────────────────────────────────────────────────┤
│ Check cache: AGENT_CACHE.get("app_123") → None           │
│                                                          │
│ ├─ BUILD agents                                          │
│ │  ├─ Orchestrator (~400ms)                              │
│ │  ├─ PO Agent (~400ms)                                  │
│ │  ├─ QA Agent (~400ms)                                  │
│ │  ├─ TM Agent (~400ms)                                  │
│ │  └─ Critic Agent (~400ms)                              │
│ │  Total: ~2 seconds                                     │
│ │                                                        │
│ └─ CACHE agents                                          │
│    AGENT_CACHE["app_123"] = (scope_hash, agents)         │
│    scope_hash = "abc123def456..." (SHA1 of scope)        │
│                                                          │
│ Response time: 5-10 seconds (includes team.run)          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ REQUEST 2 (app_123, same scope)                          │
├──────────────────────────────────────────────────────────┤
│ Check cache: AGENT_CACHE.get("app_123") → Found!         │
│                                                          │
│ ├─ old_h = "abc123def456..."                             │
│ ├─ scope_h = hash({jira: ["PROJ"], aha: ["AHA001"]})     │
│ │  └─ Matches old_h? YES ✅                              │
│ │                                                        │
│ ├─ REUSE agents (cached)                                 │
│ │  └─ Lookup time: <100ms                                │
│ │                                                        │
│ Response time: 2-3 seconds (no agent rebuild!)           │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ ADMIN CHANGES SCOPE (e.g., add new JIRA project)         │
├──────────────────────────────────────────────────────────┤
│ Database update:                                         │
│   NEW SCOPE: {jira: ["PROJ", "NEWPROJ"], aha: ["AHA"]}   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ REQUEST 3 (app_123, CHANGED scope)                       │
├──────────────────────────────────────────────────────────┤
│ Check cache: AGENT_CACHE.get("app_123") → Found!         │
│                                                          │
│ ├─ old_h = "abc123def456..."                             │
│ ├─ scope_h = hash({jira: ["PROJ", "NEWPROJ"], ...})      │
│ │  └─ Matches old_h? NO ❌                               │
│ │                                                        │
│ ├─ SCOPE CHANGED! Rebuild agents                         │
│ │  ├─ Delete from cache                                  │
│ │  ├─ Build 5 new agents (~2s)                           │
│ │  └─ Cache new agents with new hash                     │
│ │                                                        │
│ Response time: 5-10 seconds (agents rebuilt)             │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ REQUEST 4 (app_123, stable again)                        │
├──────────────────────────────────────────────────────────┤
│ Check cache: AGENT_CACHE.get("app_123") → Found!         │
│                                                          │
│ ├─ Scope unchanged (compared to cache)                   │
│ ├─ REUSE agents                                          │
│ │  └─ Lookup time: <100ms                                │
│ │                                                        │
│ Response time: 2-3 seconds (cached again!)               │
└──────────────────────────────────────────────────────────┘

TIMELINE:
────────────────────────────────────────────────────
Time    Event                           Cache Status
────────────────────────────────────────────────────
0s      Req 1: Build agents (~2s)       ← Cache miss
2s      Agents cached                   ✅ Cached
5s      Req 2: Reuse agents (<100ms)    ✅ Cache hit
5.3s    Req 3: Scope changed            ❌ Cache invalid
7.3s    Agents rebuilt (~2s)            ✅ New cache
9.3s    Req 4: Reuse agents (<100ms)    ✅ Cache hit
────────────────────────────────────────────────────
```

---
## 8. Token Budget Allocation

```
TOTAL TOKEN BUDGET: 3500 tokens per request

┌─────────────────────────────────────────────────────────┐
│ Context Block (~1200 tokens)                            │
├─────────────────────────────────────────────────────────┤
│ [CURRENT SCOPE] section          ~100 tokens            │
│ [Earlier Summary] section         ~500 tokens           │
│ [Recent Messages] section         ~500 tokens           │
│ [Guidance] section                ~100 tokens           │
│ Subtotal:                         ~1200 tokens          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ LLM Conversation (~2300 tokens)                         │
├─────────────────────────────────────────────────────────┤
│ Orchestrator reasoning            ~400 tokens           │
│   ├─ Read user request                                  │
│   ├─ Analyze task type                                  │
│   └─ Decide which agent                                 │
│                                                         │
│ PO Agent (if selected)            ~900 tokens           │
│   ├─ Create user stories                                │
│   ├─ Format RAID issues                                 │
│   └─ Generate Jira content                              │
│                                                         │
│ QA Agent (if selected)            ~400 tokens           │
│   ├─ Validate requirements                              │
│   ├─ Plan test scenarios                                │
│   └─ Review compliance                                  │
│                                                         │
│ Critic Agent (always)             ~400 tokens           │
│   ├─ Quality assessment                                 │
│   ├─ Compliance check                                   │
│   └─ Final approval                                     │
│                                                         │
│ System/Buffer overhead            ~200 tokens           │
│ Subtotal:                         ~2100 tokens          │
└─────────────────────────────────────────────────────────┘

TOTAL ALLOCATION:    ~3300 tokens
SAFETY MARGIN:       ~200 tokens remaining

If request exceeds budget:
  └─ Context truncated with warning in logs
```

---

## 9. Message Fingerprinting (Duplicate Detection)

```
MESSAGE LIFECYCLE - DEDUPLICATION

Step 1: INITIAL STM LOAD
┌─────────────────────────────────────┐
│ app_short_memory (STM):             │
│ ID | Role      | Content             │
├────┼───────────┼─────────────────────┤
│ 1  │ user      │ "Create stories"    │
│ 2  │ orch      │ "I'll delegate"     │
│ 3  │ po_agent  │ "Here are stories"  │
└─────────────────────────────────────┘
             │
             ▼
Step 2: BUILD FINGERPRINTS
┌─────────────────────────────────────────┐
│ For each STM message:                   │
│                                         │
│ Message 1:                              │
│   source = "user"                       │
│   content = "Create stories"            │
│   fingerprint = sha256(source + content)│
│   result = "a1b2c3d4e5f6..."           │
│                                         │
│ Message 2:                              │
│   fingerprint = "g7h8i9j0k1l2..."       │
│                                         │
│ Message 3:                              │
│   fingerprint = "m3n4o5p6q7r8..."       │
│                                         │
│ prev_fps = {                            │
│   "a1b2c3d4e5f6...",                    │
│   "g7h8i9j0k1l2...",                    │
│   "m3n4o5p6q7r8..."                     │
│ }                                       │
└─────────────────────────────────────────┘

Step 3: TEAM RUNS & GENERATES MESSAGES
┌──────────────────────────────────────┐
│ result.messages = [                  │
│   {role: "user", content: "..."},     │
│   {role: "orch", content: "..."},     │
│   {role: "po_agent", content: "..."}, │
│   {role: "critic", content: "..."}    │
│ ]                                    │
└──────────────────────────────────────┘

Step 4: PERSIST TO HISTORY WITH DEDUP
┌──────────────────────────────────────────────┐
│ For each message (skip user):                │
│                                              │
│ Message: orch, "I'll delegate"               │
│   ├─ Calculate fingerprint                   │
│   ├─ fingerprint = "g7h8i9j0k1l2..."         │
│   ├─ Check: Is in prev_fps?                  │
│   │   └─ YES → SKIP (duplicate!)             │
│   └─ Not inserted to history                 │
│                                              │
│ Message: po_agent, "Here are stories"        │
│   ├─ Calculate fingerprint                   │
│   ├─ fingerprint = "m3n4o5p6q7r8..."         │
│   ├─ Check: Is in prev_fps?                  │
│   │   └─ YES → SKIP (duplicate!)             │
│   └─ Not inserted to history                 │
│                                              │
│ Message: critic, "Great work!"               │
│   ├─ Calculate fingerprint                   │
│   ├─ fingerprint = "s9t0u1v2w3x4..."         │
│   ├─ Check: Is in prev_fps?                  │
│   │   └─ NO → INSERT to app_chat_history     │
│   └─ Inserted (first time seeing this)       │
│                                              │
│ Result: Only new messages persisted!         │
└──────────────────────────────────────────────┘

WHY NEEDED:
──────────
If team.run() is called multiple times and agents
repeat their responses, fingerprinting prevents
storing the same message twice to history.
```

---

## 10. Request Timing Breakdown (Typical)

```
TOTAL TIME: ~5-10 seconds per request

┌─────────────────────────────────────────────────┐
│ Time (ms)   Activity                            │
├─────────────────────────────────────────────────┤
│ 0-50        Validation                          │
│ 50-150      Load STM from DB (read)             │
│ 150-200     Store user message (DB write)       │
│ 200-300     Load scope (DB read)                │
│                                                 │
│ 300-400     Build agents (if cache miss)        │
│ 300-50      Reuse agents (if cache hit) ✅      │
│             └─ Saves ~2000ms!                   │
│                                                 │
│ 400-500     Setup team                          │
│ 500-600     Build context block                 │
│ 600-700     Wrap message                        │
│                                                 │
│ 700-4700    TEAM.RUN() with agents              │
│             ├─ Orchestrator thinks (~1500ms)    │
│             ├─ PO Agent works (~2000ms)         │
│             ├─ Critic reviews (~1200ms)         │
│             ├─ LLM calls + retries              │
│             └─ Agent collaboration              │
│                                                 │
│ 4700-4800   Extract final output                │
│ 4800-4900   Persist to history (DB write)       │
│ 4900-4950   Persist final to STM (DB write)     │
│ 4950-5000   Build response                      │
│ 5000-5050   Log metrics                         │
│ 5050-5100   Cleanup                             │
│                                                 │
│ TOTAL: ~5100ms (5.1 seconds)                    │
└─────────────────────────────────────────────────┘

BREAKDOWN BY COMPONENT:
  DB Operations:        ~200ms (5%)
  Agent Init/Setup:     ~2000ms (40%) [if no cache]
  Team.run() (LLM):     ~4000ms (80%) [biggest cost]
  Response Building:    ~100ms (2%)
  Cleanup:              ~50ms (1%)
  ──────────────────
  TOTAL:                ~5000ms

OPTIMIZATION OPPORTUNITY:
  Agent cache (from 5s → 2.5s):  Save ~2000ms
  Reduce agent turns (3 → 2):    Save ~1000ms
  Reduce context (3500 → 2000):  Save ~500ms
  Smaller LLM model (gpt-4 → 4o): Save ~1000ms
```

---

**End of Visual Diagrams**
