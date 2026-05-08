# Database Model - Visual Architecture & Diagrams

## 1. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI Application (demo.py)               │
│                          /chat endpoint                           │
└─────────────────────────┬──────────────────────────────────────┬──┘
                          │                                      │
                ┌─────────▼────────────┐        ┌────────────────▼──────┐
                │   Chat Service       │        │  Team State Service   │
                │ ┌─────────────────┐  │        │ ┌──────────────────┐  │
                │ │store_user_msg   │  │        │ │save(app_id,state)│  │
                │ │store_agent_msg  │  │        │ │load(app_id)      │  │
                │ │store_final_msg  │  │        │ └──────────────────┘  │
                │ │load_short_memory│  │        │                       │
                │ │load_full_history│  │        └───────┬──────────────┘
                │ └────────┬────────┘  │                │
                └──────────┼───────────┘                │
                           │                            │
                    ┌──────▼────────────────────────────▼──────────┐
                    │  ChatRepository & TeamStateRepository       │
                    │  ┌────────────────────────────────────────┐ │
                    │  │ add_history()                          │ │
                    │  │ add_short_memory()  ← Deadlock Retry   │ │
                    │  │ get_short_memory()                     │ │
                    │  │ get_full_history()                     │ │
                    │  │ upsert_state()  ← Upsert Pattern       │ │
                    │  │ get_state()                            │ │
                    │  └────────────────────────────────────────┘ │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼────────────────────────┐
                    │  SQLAlchemy ORM + AsyncSession        │
                    │  ┌──────────────────────────────────┐ │
                    │  │ engine (aioodbc driver)          │ │
                    │  │ AsyncSessionLocal (sessionmaker) │ │
                    │  │ Connection Pool (5 + 10 overflow)│ │
                    │  └──────────────────────────────────┘ │
                    └──────────────┬────────────────────────┘
                                   │
                    ┌──────────────▼────────────────────────┐
                    │  MSSQL Database                       │
                    │  ┌──────────────────────────────────┐ │
                    │  │ [dbo].[app_chat_history]         │ │
                    │  │   └─ Immutable audit log         │ │
                    │  │                                  │ │
                    │  │ [dbo].[app_short_memory]         │ │
                    │  │   └─ Windowed context (20 rows)  │ │
                    │  │                                  │ │
                    │  │ [dbo].[app_team_state]           │ │
                    │  │   └─ Agent state (JSON)          │ │
                    │  │                                  │ │
                    │  │ Scope Tables (external)          │ │
                    │  │   └─ applicationonboarding...    │ │
                    │  └──────────────────────────────────┘ │
                    └─────────────────────────────────────────┘
```

---

## 2. Data Model ER Diagram

```
┌─────────────────────────────────┐
│   app_chat_history              │
├─────────────────────────────────┤
│ PK  id: INT (IDENTITY)          │
├─────────────────────────────────┤
│ FK  application_id: VARCHAR(64) │◄─────────┐
│     role: VARCHAR(32)           │          │
│     content: TEXT               │          │
│     created_at: DATETIMEOFFSET  │          │
│     INDEX: (application_id)     │          │
└─────────────────────────────────┘          │
        ▲                                     │
        │ Write: USER & AGENT messages      │
        │ Growth: Unbounded                 │
        │ Used for: Audit trail, Analysis   │
        │                                   │
┌───────┴───────────────────────────────────┴──────┐
│           Multi-Tenant Application                │
│  (One application_id → Many conversations)       │
└───────┬───────────────────────────────────┬──────┘
        │                                   │
        │ Write: USER & FINAL messages     │
        │ Growth: Bounded (20 rows)        │
        │ Used for: LLM Context injection  │
        ▼                                   ▼
┌─────────────────────────────────┐
│   app_short_memory              │
├─────────────────────────────────┤
│ PK  id: INT (IDENTITY)          │
├─────────────────────────────────┤
│ FK  application_id: VARCHAR(64) │
│     role: VARCHAR(32)           │
│     content: TEXT               │
│     created_at: DATETIMEOFFSET  │
│     INDEX: (application_id)     │
└─────────────────────────────────┘

        ▼ Read: Team state
        │ Update: Workflow checkpoints

┌─────────────────────────────────┐
│   app_team_state                │
├─────────────────────────────────┤
│ PK  id: INT (IDENTITY)          │
├─────────────────────────────────┤
│ FK  application_id: VARCHAR(64) │ (UNIQUE)
│     state: JSON (NVARCHAR)      │
│     updated_at: DATETIMEOFFSET  │
│     INDEX: (application_id)     │
│     CHECK: ISJSON([state]) = 1  │
└─────────────────────────────────┘
```

---

## 3. Data Flow: Message Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                     User Sends Message                              │
└────────────────────────┬────────────────────────────────────────────┘
                         │
         ┌───────────────▼────────────────────┐
         │ ChatService.store_user_message()   │
         └───────────────┬────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    ┌─────────────────┐      ┌──────────────────┐
    │ add_history()   │      │add_short_memory()│
    │                 │      │   (with prune)   │
    └────────┬────────┘      └────────┬─────────┘
             │                        │
             ▼                        ▼
    ┌──────────────────────────────────────────┐
    │   MSSQL Database Updates                 │
    │   • app_chat_history += 1 row            │
    │   • app_short_memory += 1 row            │
    │   • OLD rows deleted (keep 20)           │
    └──────────────────────────────────────────┘
             │                        │
             └──────────────┬─────────┘
                            │
         ┌──────────────────▼─────────────────┐
         │ Agent Orchestrator Gets Context    │
         │                                    │
         │ load_short_memory() ──────┐        │
         │   Returns: 20 newest msgs │        │
         │   Order: oldest → newest  │        │
         │                           │        │
         └───────────────────────────┼────────┘
                                     │
                    ┌────────────────▼──────────┐
                    │ LLM System Prompt         │
                    │ + Context Injection       │
                    │ + Current Request         │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │ Agents Process Request    │
                    │ (Orchestrator → Agents)   │
                    └────────────┬──────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          │                                             │
    ┌─────▼──────────┐                    ┌────────────▼──────┐
    │ Intermediate   │                    │  Final Response   │
    │ Agent Messages │                    │  From Assistant   │
    │                │                    │                   │
    │ store_agent_   │                    │ store_final_      │
    │  message()     │                    │  message()        │
    │  (HISTORY ONLY)│                    │  (STM ONLY)       │
    └─────┬──────────┘                    └────────────┬──────┘
          │                                            │
          ▼                                            ▼
    [app_chat_history]                        [app_short_memory]
    (For audit/analysis)                       (For next context)
```

---

## 4. Deadlock Retry Flow

```
┌────────────────────────────────────────────────┐
│  add_short_memory(app_id, role, content)       │
│  (Called when concurrent users interact)       │
└────────────────┬─────────────────────────────┬─┘
                 │                             │
        ┌────────▼────────────┐      ┌─────────▼──────────┐
        │  Attempt 1          │      │  retry_on_deadlock │
        │  Execute transaction│      │  max_retries = 3   │
        └────────┬────────────┘      └───────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ INSERT new msg  │
        │ FLUSH to DB     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────────────┐
        │ DELETE old messages     │
        │ (Keep only 20)          │
        └────────┬────────────────┘
                 │
                 ▼
        ┌─────────────────────────────┐
        │ COMMIT transaction          │
        └────────┬────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
     ▼ SUCCESS               ▼ DEADLOCK (Error 40001)
  ┌─────┐              ┌─────────────────────────┐
  │ OK  │              │ Catch DBAPIError (40001)│
  └─────┘              │ Attempt < max_retries?  │
                       └──────────┬──────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │ YES                        │
                    │ Wait exponentially:        │
                    │ Attempt 1: 0.1s (2^0*0.1) │
                    │ Attempt 2: 0.2s (2^1*0.1) │
                    │ Attempt 3: 0.4s (2^2*0.1) │
                    │ Log: [DEADLOCK] Retry...  │
                    └──────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Retry (Go to: INSERT)│
                    └─────────────────────┘
                               │
                    ┌──────────▴──────────┐
                    │ SUCCESS or          │
                    │ Final Attempt Failed│
                    └─────────────────────┘
```

---
## 5. Scope Loading Flow

```
┌─────────────────────────────────────────┐
│  Build Product Owner Agent               │
│  scope = get_scope(application_id)      │
└──────────────┬──────────────────────────┘
               │
    ┌──────────▼──────────────────┐
    │ ScopeService.get_scope()    │
    │  Validates app exists       │
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │ SQL Pivot Query on External Tables   │
    │                                      │
    │ FROM applicationonboarding$...       │
    │  WHERE appid = :appid                │
    │  GROUP BY appid, confluence          │
    │  PIVOT: JIRA/AHA to columns          │
    └──────────────┬────────────────────────┘
                   │
       ┌───────────▼────────────────┐
       │ ScopeContext Created       │
       │                            │
       │ app_id: "app_123"          │
       │ jira_projectkey: "PROJ"    │
       │ aha_id: "AHA001"           │
       │ confluence_spacekey: "FIN" │
       └───────────┬────────────────┘
                   │
       ┌───────────▼──────────────────────┐
       │ Normalize to lowercase            │
       │ all_ids_lower()                   │
       │ for case-insensitive comparison   │
       └───────────┬──────────────────────┘
                   │
       ┌───────────▼────────────────────────┐
       │ Create Scope Header String         │
       │                                    │
       │ APPLICATION SCOPE (STRICT)        │
       │ application_id: app_123           │
       │ JIRA Projects: [PROJ]             │
       │ AHA IDs: [AHA001]                 │
       │ CONFLUENCE Spaces: [FIN]          │
       │ BOUNDARY RULES: Do NOT exceed...  │
       └───────────┬────────────────────────┘
                   │
       ┌───────────▼────────────────────────┐
       │ Inject into Agent System Message   │
       │                                    │
       │ system_msg = (                     │
       │   scope_header +                   │
       │   agent_prompt +                   │
       │   other_instructions               │
       │ )                                  │
       └───────────┬────────────────────────┘
                   │
       ┌───────────▼────────────────────────┐
       │ Agent Now Knows Boundaries         │
       │                                    │
       │ "I can only work with:"            │
       │  • PROJ (not other projects)       │
       │  • AHA001 features                 │
       │  • FIN space in Confluence         │
       │ "I will reject out-of-scope reqs"  │
       └────────────────────────────────────┘
```

---

## 6. Connection Pool Lifecycle

```
┌──────────────────────────────────────────────────────────┐
│           Application Startup                            │
│  engine = create_async_engine(                           │
│      ASYNC_DB_URL,                                       │
│      pool_size=5,       # Core pool                      │
│      max_overflow=10,   # Burst capacity                 │
│  )                                                       │
└───────────────┬────────────────────────────────────────┬─┘
                │                                        │
        ┌───────▼────────────────────┐        ┌──────────▼──────┐
        │ Connection Pool Created    │        │ Session Factory │
        │ (Empty initially)          │        │ Ready           │
        │ Max size: 5 + 10 = 15      │        │                 │
        └───────┬────────────────────┘        └─────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  Request 1-5 Arrive (Normal)           │
    │  Pool: [CONN] [CONN] [CONN] [CONN] [CONN]
    │                                        │
    └───────────┬────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  Requests 6-15 Arrive (Burst)          │
    │  Overflow: [CONN] [CONN]...[CONN]      │
    │  Pool + Overflow: 15 connections       │
    │                                        │
    └───────────┬────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  Request 16 Arrives (Exceeds Max)      │
    │  Error: TimeoutError                   │
    │  "Timeout acquiring connection"        │
    │  → Increase pool_size or max_overflow  │
    │                                        │
    └───────────┬────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  Requests Complete → Return Connections
    │  Returned conn: ping() → OK            │
    │  (pool_pre_ping=True verified conn)    │
    │  Conn recycled for next request        │
    │                                        │
    └────────────────────────────────────────┘
```

---
## 7. Message Storage Strategy

```
CONVERSATION FLOW:

Turn 1:
  User: "List all epics"
  ├─ Store in: [HISTORY] + [STM]
  │
  Agent: "Found 3 epics..."
  ├─ Store in: [HISTORY] ONLY
  │
  Final: "✅ Here are the epics..."
  ├─ Store in: [STM] ONLY

Turn 2:
  [STM] Contains: ← Used for LLM context
  1. User: "List all epics" (oldest)
  2. Orchestrator: "Found 3 epics..."
  3. Assistant: "✅ Here are the epics..." (newest)

  [HISTORY] Contains: ← Used for audit
  1. User: "List all epics"
  2. Orchestrator: "Found 3 epics..."
  3. Assistant: "✅ Here are the epics..."

Turn N (after 20 STM messages):
  [STM] pruned to keep 20 most recent
  [HISTORY] still has ALL messages

RATIONALE:
┌─────────────┬────────────────────┬────────────────────┐
│ Scenario    │ Full History       │ Short-Term Memory  │
├─────────────┼────────────────────┼────────────────────┤
│ User wants  │ ✅ Available        │ ❌ Pruned away     │
│ old context │ (unlimited rows)    │ (20 row limit)     │
│             │                    │                    │
│ LLM needs   │ ❌ Too verbose      │ ✅ Perfect fit     │
│ context     │ (1000s of messages)│ (last 20)          │
│             │                    │                    │
│ Compliance/ │ ✅ Complete log     │ ❌ Not applicable  │
│ Audit       │ (everything)        │ (partial)          │
├─────────────┼────────────────────┼────────────────────┤
│ Storage     │ Unbounded growth   │ Bounded (20)       │
│ Query speed │ O(n) - slow        │ O(n) - fast (n=20) │
│ Use case    │ Analysis, reports  │ LLM context        │
└─────────────┴────────────────────┴────────────────────┘
```

---

## 8. Team State JSON Structure

```
Example app_team_state Record:

{
  "application_id": "app_123",
  "state": {
    "current_agent": "orchestrator",
    "agents_available": [
      "orchestrator",
      "po_agent",
      "qa_agent",
      "critic_agent",
      "test_manager_agent"
    ],
    "conversation": {
      "turn": 5,
      "messages": [
        {
          "role": "user",
          "content": "Create user stories..."
        },
        {
          "role": "orchestrator",
          "content": "I'll delegate to PO..."
        }
      ]
    },
    "po_agent": {
      "status": "active",
      "tasks_completed": 2,
      "last_output": "Posted 3 user stories"
    },
    "qa_agent": {
      "status": "pending",
      "tasks_completed": 0
    },
    "metrics": {
      "total_tokens": 1500,
      "agent_breakdown": {
        "orchestrator": 300,
        "po_agent": 1000,
        "qa_agent": 200
      }
    }
  },
  "updated_at": "2026-03-24T10:30:00Z"
}

Key Points:
• Stored as JSON in NVARCHAR(MAX)
• Contains entire team state snapshot
• Allows resuming workflows
• Used for debugging/auditing
• Can be extended with custom fields
```

---

## 9. Table Size Projections

```
Assumption: 100 applications, 50 turns per day each

MONTHLY GROWTH:

app_chat_history:
  Rows per turn: 2 (user + agent response)
  Rows per app per day: 100 turns
  Rows per day total: 100 apps × 100 rows = 10,000 rows
  Rows per month: 10,000 × 30 = 300,000 rows
  
  Storage per row: ~500 bytes (content varies)
  Monthly storage: 300,000 × 500 bytes = 150 MB
  Yearly storage: 150 MB × 12 = 1.8 GB

app_short_memory:
  Rows per app: 20 (fixed via pruning)
  Total rows: 100 apps × 20 = 2,000 rows
  Storage: ~1 MB (constant)
  
app_team_state:
  Rows: 100 (one per app)
  Storage: ~100 KB
  
RECOMMENDATION:
• Archive history monthly to separate table
• Keep last 90 days in app_chat_history
• Auto-prune messages older than 90 days
• Monitor storage growth quarterly
```

---
## 10. Troubleshooting Decision Tree

```
SYMPTOM: Database Connection Error

│
├─ "Timeout acquiring connection"?
│  └─ Check: Database running?
│           Connection string correct?
│           Network/firewall blocking?
│           Pool size sufficient?
│
├─ "Connection refused"?
│  └─ Check: Server name (FQDN)?
│           Port 1433 open?
│           Credentials correct?
│
├─ "Deadlock error 40001"?
│  └─ Check: Multiple agents writing STM?
│           Retry logic executing?
│           Exponential backoff working?
│
├─ "STM not pruning" (>20 rows)?
│  └─ Check: add_short_memory() called?
│           DELETE statement executing?
│           Transaction committed?
│
├─ "Scope not found"?
│  └─ Check: Application in scope tables?
│           JIRA/AHA mappings exist?
│           Correct app_id format?
│
└─ "JSON validation failed"?
   └─ Check: State is dict, not string?
            Valid JSON structure?
            No circular references?
```

---

## Summary Table

```
┌────────────────────┬──────────────────┬──────────────┬──────────────┐
│ Table              │ Purpose          │ Growth       │ Cleanup      │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ app_chat_history   │ Audit log        │ ~10K/day     │ Manual (90d) │
│                    │ Full history     │ Unbounded    │ Archive      │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ app_short_memory   │ LLM context      │ Bounded      │ Automatic    │
│                    │ Recent messages  │ 20 rows/app  │ (Pruned)     │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ app_team_state     │ Agent state      │ 1 row/app    │ None         │
│                    │ Resumable        │ Fixed        │ Manual       │
└────────────────────┴──────────────────┴──────────────┴──────────────┘
```

---

**For detailed information, see:** `DATABASE_MODEL_KT.md`  
**Last Updated:** March 24, 2026
  