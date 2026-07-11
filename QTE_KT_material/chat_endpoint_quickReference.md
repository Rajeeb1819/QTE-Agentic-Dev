# Chat Endpoint - Quick Reference Guide

## Quick Navigation

**[Full Documentation](./CHAT_ENDPOINT_DOCUMENTATION.md)** - Complete 22-step breakdown  
**This Page** - Quick reference (5-minute read)

---

## What Does This Endpoint Do?

The `/chat` endpoint orchestrates a **team of 5 AI agents** working together to handle your request:

```
Your Message
    ↓
[Orchestrator Agent]  ← Decides who should handle the task
    ├→ [Product Owner Agent]     (creates specs/stories)
    ├→ [QA Agent]                (validates/tests)
    └→ [Test Manager Agent]      (plans testing)
         ↓
    [Critic Agent]               ← Reviews quality
    ↓
Your Response
```

---

## API Usage

### Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Create user stories for login feature",
    "application_id": "app_12345"
  }'
```

### Response

```json
{
  "content": "Here are 3 user stories for login feature...",
  "final_output": "Here are 3 user stories for login feature...",
  "source": "po_agent",
  "critic_review": "Well-structured and testable stories.",
  "models_usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 800
  },
  "data": [
    {"role": "user", "content": "Create user stories..."},
    {"role": "orchestrator", "content": "I'll delegate to PO..."},
    {"role": "po_agent", "content": "Here are 3 stories..."},
    {"role": "critic_agent", "content": "Well-structured..."}
  ],
  "created_at": "2026-03-25T10:30:00Z"
}
```

---

## 22-Step Flow (Simplified)

| # | Step | Purpose | Database Impact |
|---|------|---------|-----------------|
| 1 | Validate input | Check not empty | - |
| 2 | Track request | Monitor concurrency | - |
| 3 | Load STM | Get last 20 messages | Read only |
| 4 | Build fingerprints | Prevent duplicates | - |
| 5 | Store user message | Save to history + STM | INSERT × 2 |
| 6 | Load scope | Get JIRA/AHA boundaries | Read only |
| 7 | Build agents | Create/reuse agent instances | - |
| 8 | Build team | Setup orchestration | - |
| 9 | Build context | Assemble hybrid context | - |
| 10 | Inject context | Feed to team | - |
| 11 | Wrap message | Combine context + request | - |
| 12 | Run team | Agents collaborate (retry on errors) | - |
| 13 | Save state | Team workflow state | INSERT/UPDATE |
| 14 | Persist history | Save agent messages | INSERT |
| 15 | Extract final | Pick response to return | - |
| 16 | Persist final | Save to STM + auto-prune | INSERT + DELETE |
| 17 | Extract critic | Get quality review | - |
| 18 | Aggregate tokens | Calculate usage | - |
| 19 | Build response | Format ChatOut | - |
| 20 | Handle errors | Classify + retry/fail | - |
| 21 | Cleanup | Decrement active requests | - |
| 22 | Return response | Send to client | - |

---

## Key Concepts

### 1. Short-Term Memory (STM)

**What:** Last 20 messages for each application  
**Why:** Provides recent context to agents  
**Auto-maintained:** Prunes to exactly 20 rows after each message

```
STM Flow:
user message arrives
    ↓
INSERT into app_short_memory
    ↓
AUTO-PRUNE (keep only last 20)
    ↓
Next request reads these 20 messages
```

### 2. Hybrid Context Strategy

**Recent (Last 5):** Full detail
```
User: "Create stories"
Orchestrator: "I'll delegate to PO"
PO Agent: "Here are 3 stories"
Critic: "Well-structured"
Orchestrator: "Proceeding"
```

**Historical (6-20):** LLM-summarized (~500 words)
```
"In earlier turns, we discussed:
- Authentication module design
- Key decisions: Use OAuth 2.0
- Action items: Create user stories"
```

**Result:** Agents see detailed recent context + key decisions from history

### 3. Agent Caching

**First request (app_id=app_123):**
- Build 5 agents (~2 seconds)
- Cache with scope hash

**Subsequent requests (same app_id, same scope):**
- Reuse cached agents (<100ms)

**Scope changes:**
- Auto-detect via hash
- Rebuild agents automatically
- No cache invalidation needed

### 4. Error Classification

```
Non-Retryable (Fail immediately):
  401 Unauthorized      ❌ Auth issue
  403 Forbidden         ❌ Permission issue

Retryable (Auto-retry):
  429 Rate Limited      ✅ Retry up to 1-2 times
  500 Server Error      ✅ Retry up to 5 times
  503 Unavailable       ✅ Retry up to 5 times

Backoff: 1s → 2s → 4s → 8s → 16s
```

### 5. Concurrent Access (No Throttling)

```
ACTIVE_REQUESTS = {
  "app_123": 5,      ← 5 concurrent users
  "app_456": 3,      ← 3 concurrent users
  "app_789": 12      ← 12 concurrent users
}

Total: 20 concurrent requests, all running in parallel
```

---

## Common Scenarios

### Scenario 1: First User Request

```
Step 1: User sends "Create user stories"
Step 2: STM is empty → Load 0 messages
Step 3: Agents not cached → Build 5 agents (2 seconds)
Step 4: Team runs → Orchestrator delegates to PO
Step 5: Response returned (5-10 seconds total)
Step 6: Next request → Cache hit, agents reused (100ms)
```

### Scenario 2: Follow-up Request Same User

```
Step 1: User sends "Make those testable"
Step 2: STM loaded → 4 messages from first request
Step 3: Agents cached → Reuse (no rebuild)
Step 4: Context built → Recent 4 messages + guidance
Step 5: Team runs → Better understanding from context
Step 6: Response faster (2-3 seconds, not 5-10)
```

### Scenario 3: High Concurrency

```
Request 1 (app_123): STM loaded, team running
Request 2 (app_123): STM loaded separately (parallel), team running
Request 3 (app_456): Different app, separate agent instances, running
...
All in parallel, no semaphores, no throttling
```

### Scenario 4: Rate Limited (429 Error)

```
Step 1: Team.run() fails with HTTP 429
Step 2: Classified as RateLimitError → Retryable
Step 3: Max retries for 429 = 1 (can't hammer Azure)
Step 4: Retry once after ~1 second
Step 5: Still 429? → Return error to user
Step 6: User sees: "Too many requests, please wait a moment"
```

---
## Configuration Quick Ref

```bash
# How many messages to include in context (full detail)
CHAT_MAX_RECENT_MESSAGES=5

# How many older messages to summarize
CHAT_MAX_HISTORICAL_MESSAGES=15

# Max tokens for entire context block
CHAT_MAX_CONTEXT_TOKENS=3500

# Max agent turns per request
CHAT_MAX_TURNS_PER_RUN=3

# Enable request rate limiting (usually not needed)
CHAT_ENABLE_RATE_LIMITER=0

# Debug: Log STM loading details
CHAT_LOG_MEMORY=0
```

---

## Monitoring

### Real-Time Metrics

```bash
# See concurrent requests by app
curl http://localhost:8000/metrics/rate-limit

# See token pricing info
curl http://localhost:8000/metrics/tokens

# Check health
curl http://localhost:8000/health
```

### Logs to Watch

```
[CONCURRENT] Processing request abc-123 for app app_123 (active: 5)
[STM_DEBUG] loaded_from_db=15 rows
[TOKEN_USAGE] Total: 2600 tokens (prompt: 1500, completion: 1100)
  └─ po_agent: 1500 tokens
  └─ orchestrator: 400 tokens
  └─ critic_agent: 400 tokens
[CONCURRENT] Completed request abc-123
```

---

## Debugging

### Problem: "Timeout acquiring connection"

**Cause:** All database connections in use

**Fix:**
```bash
# Increase pool size (session.py)
pool_size=10, max_overflow=15

# OR enable rate limiting
CHAT_ENABLE_RATE_LIMITER=1
```

### Problem: "429 Too many requests"

**Cause:** Azure OpenAI rate limited

**Fix:**
```bash
# Reduce tokens per request
CHAT_MAX_CONTEXT_TOKENS=2000

# OR reduce agent turns
CHAT_MAX_TURNS_PER_RUN=2

# OR enable rate limiter
CHAT_ENABLE_RATE_LIMITER=1
```

### Problem: "STM not pruning" (more than 20 rows)

**Cause:** Deadlock in prune operation

**Fix:**
```bash
# Check database logs
SELECT * FROM sys.event_log WHERE error_number = 40001

# Manual prune:
DELETE FROM app_short_memory
WHERE application_id = 'app_id'
  AND id NOT IN (
    SELECT TOP 20 id FROM app_short_memory
    WHERE application_id = 'app_id'
    ORDER BY created_at DESC
  )
```

### Problem: Agents using old scope

**Cause:** Cache not invalidated

**Fix:**
```bash
POST /clear-cache/app_123
# Agents rebuilt on next request with new scope
```

---

## Database Tables

| Table | Rows | Purpose | Growth |
|-------|------|---------|--------|
| `app_chat_history` | ∞ | Full audit log | ~10K/day per app |
| `app_short_memory` | 20/app | Recent context | Fixed (20 rows) |
| `app_team_state` | 1/app | Workflow state | Fixed (1 row) |

---
## Performance Tips

1. **Reduce context tokens**
   ```bash
   CHAT_MAX_CONTEXT_TOKENS=2500
   ```

2. **Reduce agent turns**
   ```bash
   CHAT_MAX_TURNS_PER_RUN=2
   ```

3. **Reduce recent messages**
   ```bash
   CHAT_MAX_RECENT_MESSAGES=3
   ```

4. **Enable rate limiting** (queue requests)
   ```bash
   CHAT_ENABLE_RATE_LIMITER=1
   ```

5. **Monitor token usage**
   ```bash
   curl http://localhost:8000/metrics/tokens
   ```

---

## For New Team Members

**Read in this order:**

1. **This page** (5 minutes) - Get the big picture
2. **CHAT_ENDPOINT_DOCUMENTATION.md** (30 minutes) - Understand each step
3. **demo.py lines 950-1400** (20 minutes) - Read actual code
4. **DATABASE_MODEL_KT.md** (15 minutes) - Understand persistence
5. **src/agents/** (30 minutes) - See how individual agents work

**Total:** ~100 minutes to become competent

---

## Key Takeaways

✅ **Concurrent:** All requests run in parallel, no throttling  
✅ **Intelligent:** Hybrid context (recent + summarized historical)  
✅ **Resilient:** Auto-retry on transient errors  
✅ **Observable:** Per-agent token tracking  
✅ **Maintainable:** Cached agents, auto-purged STM  
✅ **Scalable:** Supports 100+ concurrent users

---

**Questions?** See full documentation: `CHAT_ENDPOINT_DOCUMENTATION.md`
