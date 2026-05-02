# fastapi_app.py
#rajeeb
# git feature branch test
import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Dict, Optional, Tuple, Literal, Sequence, List
import requests
from requests.auth import HTTPBasicAuth

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from autogen_agentchat.messages import TextMessage, BaseChatMessage, BaseAgentEvent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

# --- Use only builders; remove singleton agent imports ---
from src.agents.product_owner_agent import build_product_owner_agent
from src.agents.qa_agent import build_qa_agent
from src.agents.test_manager_agent import build_test_manager_agent
from src.agents.orchestrator_agent import build_orchestrator_agent
from src.agents.critic_agent import build_critic_agent

# Shared model client
from src.utils.llm_config_utils.agent_config import az_model_client

# LLM utility for summarization
from src.utils.llm_config_utils.llm_initiate import get_LLM_response

# DB services
from src.db_model_mssql.chat_service import ChatService as PgChatService
from src.db_model_mssql.chat_service import team_state_service as pg_team_state_service

# Scope from MSSQL
from src.db_model_mssql.scope_service import ScopeService

# Persist team state (and TRY to load it back)
from src.db_model_mssql.teams_registry import save_team_state
# from src.utils.llm_config_utils.llm_setup_for_ragas import RAGAS_CONFIG
try:
    from src.db_model_mssql.teams_registry import load_team_state
except Exception:
    load_team_state = None  # fallback later

#selector func
from src.utils.selector_func import extract_payload_from_message, make_selector_and_helpers


# ======================================================================================
# Custom Exception Classes for LLM Errors
# ======================================================================================

class LLMAPIError(Exception):
    """Base exception for LLM API errors"""
    def __init__(self, message: str, status_code: int, retry_after: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(self.message)


class UnauthorizedError(LLMAPIError):
    """401 - Invalid API key or authentication failure"""
    def __init__(self, message: str = "Authentication failed. Invalid API key or token."):
        super().__init__(message, status_code=401)


class ForbiddenError(LLMAPIError):
    """403 - Access denied to resource"""
    def __init__(self, message: str = "Access denied. You don't have permission to access this model or resource."):
        super().__init__(message, status_code=403)


class RateLimitError(LLMAPIError):
    """429 - Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded. Please try again later.", retry_after: Optional[int] = None):
        super().__init__(message, status_code=429, retry_after=retry_after)


class ModelUnavailableError(LLMAPIError):
    """503 - Model or service unavailable"""
    def __init__(self, message: str = "The model or service is temporarily unavailable. Please try again later."):
        super().__init__(message, status_code=503)


# ======================================================================================
# FastAPI app scaffolding
# ======================================================================================

app = FastAPI(title="SelectorGroupChat - Dynamic Orchestrator Delegation (PO/QA/TM) + Critic Review")


from rag_demo import rag_router
app.include_router(rag_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================================
# Global Exception Handler for LLM Errors
# ======================================================================================

@app.exception_handler(LLMAPIError)
async def llm_api_error_handler(request: Request, exc: LLMAPIError):
    """Handle all LLM API errors gracefully - return as ChatOut format"""
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    
    logger.error(f"LLM API Error: {exc.status_code} - {exc.message}")
    
    # Format error message for user
    user_message = exc.message
    
    # Add helpful context based on error type
    if exc.status_code == 401:
        user_message += "\n\n💡 This appears to be a configuration issue. Please contact your administrator."
    elif exc.status_code == 403:
        user_message += "\n\n💡 You may need different permissions. Please contact your administrator."
    elif exc.status_code == 429:
        retry_info = f" Please wait {exc.retry_after} seconds and try again." if exc.retry_after else " Please try again in a moment."
        user_message += f"\n\n⏳{retry_info}"
    elif exc.status_code in {500, 502, 503, 504}:
        user_message += "\n\n🔄 This is a temporary issue. Please try again in a moment."
    
    # Return in ChatOut format (same as successful responses)
    return JSONResponse(
        status_code=200,  # Return 200 so UI handles it as normal response
        headers=headers,
        content={
            "id": None,
            "source": "system",
            "models_usage": None,
            "metadata": {
                "error_type": "llm_error",
                "error_code": exc.status_code,
                "recoverable": exc.status_code in {429, 500, 502, 503, 504},
                "retry_after": exc.retry_after
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content": user_message,
            "data": [{
                "id": None,
                "source": "system",
                "metadata": {"error": True},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content": user_message,
                "type": "TextMessage"
            }],
            "mime_type": "text/plain",
            "type": "TextMessage",
            "final_output": user_message,
            "critic_review": None
        }
    )

# Static
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return "team (SelectorGroupChat) is running"


# ======================================================================================
# Pydantic models
# ======================================================================================

class ChatIn(BaseModel):
    content: str
    application_id: str   # REQUIRED


class ModelsUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatOut(BaseModel):
    id: Optional[str] = Field(default=None)
    source: str = "assistant"
    models_usage: Optional[ModelsUsage] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: Optional[str] = None
    data: Optional[Any] = None
    mime_type: Optional[str] = None
    type: Literal["TextMessage", "ToolResult"] = "TextMessage"
    final_output: Optional[str] = None
    # ragas_metrics: Optional[Dict[str, float]] = None
    critic_review: Optional[str] = None


# ======================================================================================
# Helpers for state/history and message dict conversion
# ======================================================================================

def msg_to_dict(msg: TextMessage) -> Dict[str, Any]:
    created_at = getattr(msg, "created_at", datetime.now(timezone.utc))
    created_at_iso = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)
    return {
        "id": getattr(msg, "id", None),
        "source": getattr(msg, "source", "unknown"),
        "metadata": getattr(msg, "metadata", {}) or {},
        "created_at": created_at_iso,
        "content": getattr(msg, "content", None),
        "type": "TextMessage",
    }



# ======================================================================================
# Services & globals
# ======================================================================================

chat_service = PgChatService()
team_state_service = pg_team_state_service
scope_service = ScopeService()

# ----------------------------
# Configurable knobs (env vars)
# ----------------------------
MAX_SEEDED_HISTORY = int(os.getenv("CHAT_MAX_SEEDED_HISTORY", "5"))  # Legacy - kept for backward compatibility
MAX_TURNS_PER_RUN = int(os.getenv("CHAT_MAX_TURNS_PER_RUN", "3"))
MAX_RETRIES_RATE_LIMIT = int(os.getenv("CHAT_MAX_RETRIES_RATE_LIMIT", "1"))
ENABLE_RATE_LIMITER = os.getenv("CHAT_ENABLE_RATE_LIMITER", "0") == "1"  # Optional: Can enable token bucket rate limiter

# New context configuration
MAX_RECENT_MESSAGES_ENV = int(os.getenv("CHAT_MAX_RECENT_MESSAGES", "5"))       # Last N messages in full
MAX_HISTORICAL_MESSAGES_ENV = int(os.getenv("CHAT_MAX_HISTORICAL_MESSAGES", "15"))  # Messages to summarize (6-20)

# ----------------------------
# Request tracking (monitoring only - no throttling)
# ----------------------------
ACTIVE_REQUESTS = defaultdict(int)
ACTIVE_REQUESTS_LOCK = asyncio.Lock()

# ----------------------------
# Request tracking (monitoring only - no throttling)
# ----------------------------
ACTIVE_REQUESTS = defaultdict(int)
ACTIVE_REQUESTS_LOCK = asyncio.Lock()

# Optional: Token bucket rate limiter (disabled by default)
class TokenBucketRateLimiter:
    """Optional rate limiter - only used if ENABLE_RATE_LIMITER=1"""
    def __init__(self, rate_per_second: float, capacity: int):
        self.rate = rate_per_second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            
            wait_time = (tokens - self.tokens) / self.rate
            logger.info(f"Rate limiter: Waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self.last_update = time.time()

# Initialize rate limiter (only used if ENABLE_RATE_LIMITER=1)
RATE_LIMITER = TokenBucketRateLimiter(rate_per_second=1.0, capacity=10)

# ----------------------------
# Rate-limit aware backoff with error classification
# ----------------------------
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

def _classify_llm_error(e: Exception) -> Optional[LLMAPIError]:
    """
    Classify exception and convert to appropriate LLMAPIError.
    Returns None if error is not an LLM API error.
    """
    # Extract status code from various exception types
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    
    # Check error message for hints
    error_msg = str(e).lower()
    
    # 401 - Unauthorized
    if status == 401 or "unauthorized" in error_msg or "invalid api key" in error_msg or "authentication failed" in error_msg:
        return UnauthorizedError(
            "Authentication failed. Please check your API key or credentials."
        )
    
    # 403 - Forbidden
    if status == 403 or "forbidden" in error_msg or "access denied" in error_msg or "don't have access" in error_msg:
        return ForbiddenError(
            "Access denied. You don't have permission to use this model or the deployment is restricted."
        )
    
    # 429 - Rate Limit
    if status == 429 or "rate limit" in error_msg or "too many requests" in error_msg or "quota exceeded" in error_msg:
        # Try to extract retry-after value
        retry_after = None
        resp = getattr(e, "response", None)
        if resp:
            headers = {str(k).lower(): v for k, v in getattr(resp, "headers", {}).items()}
            retry_after_hdr = headers.get("retry-after")
            if retry_after_hdr:
                try:
                    retry_after = int(float(retry_after_hdr))
                except Exception:
                    pass
        
        if retry_after is None:
            retry_after = _parse_retry_after_from_message(str(e))
        
        return RateLimitError(
            "Rate limit exceeded. The system has sent too many requests. Please try again in a few moments.",
            retry_after=retry_after or 30
        )
    
    # 503 - Service Unavailable
    if status == 503 or "service unavailable" in error_msg or "model unavailable" in error_msg:
        return ModelUnavailableError(
            "The AI model is temporarily unavailable. Please try again in a moment."
        )
    
    # 500-504 - Server errors (generic)
    if status in {500, 502, 504}:
        return ModelUnavailableError(
            f"The AI service encountered a temporary error (HTTP {status}). Please try again."
        )
    
    return None


def _parse_retry_after_from_message(msg: str) -> Optional[int]:
    if not msg:
        return None
    m = re.search(r"retry\s+after\s+(\d+)\s*seconds?", msg, re.IGNORECASE)
    if m:
        try:
            return int(float(m.group(1)))
        except Exception:
            return None
    return None

async def call_with_backoff_async(create_call, *, max_retries=5, base=1.0, max_wait=60.0, **kwargs):
    attempt = 0
    last_error = None
    
    while True:
        try:
            return await create_call(**kwargs)
        except Exception as e:
            last_error = e
            
            # Check if this is a non-retryable LLM error (401, 403)
            llm_error = _classify_llm_error(e)
            if llm_error and llm_error.status_code in {401, 403}:
                # Don't retry auth/permission errors
                raise llm_error
            
            # REDUCED RETRIES FOR RATE LIMITS (429) - NEW
            effective_max_retries = max_retries
            if llm_error and llm_error.status_code == 429:
                effective_max_retries = MAX_RETRIES_RATE_LIMIT
                logger.warning(f"Rate limit detected, using reduced retry limit: {effective_max_retries}")
            
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            headers: Dict[str, str] = {}
            resp = getattr(e, "response", None)
            if resp is not None:
                headers = {str(k).lower(): v for k, v in getattr(resp, "headers", {}).items()}

            if status in RETRYABLE_STATUS and attempt < effective_max_retries:
                attempt += 1
                retry_after_hdr = headers.get("retry-after")
                wait = None
                if retry_after_hdr:
                    try:
                        wait = min(float(retry_after_hdr), max_wait)
                    except Exception:
                        wait = None
                if wait is None:
                    parsed = _parse_retry_after_from_message(str(getattr(e, "message", None) or e))
                    if parsed is not None:
                        wait = min(parsed, max_wait)
            if wait is None:
                    wait = min(base * (2 ** (attempt - 1)) * (1.0 + 0.25 * (2 * os.urandom(1)[0] / 255)), max_wait)
                
            logger.warning(f"Retryable error (attempt {attempt}/{effective_max_retries}), waiting {wait:.1f}s: {str(e)[:200]}")
                await asyncio.sleep(wait)
                continue
            
            # If we exhausted retries or it's not retryable, check if it's a classified LLM error
            if llm_error:
                raise llm_error
            
            # Otherwise raise the original exception
            raise


        class BackoffModelClient:
    def __init__(self, inner_client):
        self._inner = inner_client

    async def create(self, **kwargs):
        return await call_with_backoff_async(self._inner.create, **kwargs)

    def __getattr__(self, item):
        return getattr(self._inner, item)

# ----------------------------
# Agent & helpers
# ----------------------------

def _scope_hash(scope_kwargs: dict) -> str:
    return sha1(json.dumps(scope_kwargs, sort_keys=True).encode("utf-8")).hexdigest()

AGENT_CACHE: dict[str, tuple[str, dict]] = {}

def get_or_build_agents(app_id: str, scope_kwargs: dict) -> Dict[str, Any]:
    scope_h = _scope_hash(scope_kwargs)
    cached = AGENT_CACHE.get(app_id)
    if cached:
        old_h, agents = cached
        if old_h == scope_h:
            logger.info(f"[AGENT_CACHE] Using cached agents for app {app_id} (scope unchanged)")
            return agents
        else:
            logger.warning(f"[AGENT_CACHE] Scope changed for app {app_id}! Rebuilding agents with new scope: {scope_kwargs}")
    else:
        logger.info(f"[AGENT_CACHE] Building new agents for app {app_id} with scope: {scope_kwargs}")

    agents = {
        "po": build_product_owner_agent(application_id=app_id, **scope_kwargs),
        "qa": build_qa_agent(application_id=app_id, **scope_kwargs),
        "tm": build_test_manager_agent(application_id=app_id, **scope_kwargs),
        "orch": build_orchestrator_agent(application_id=app_id, **scope_kwargs),
        "critic": build_critic_agent(application_id=app_id, **scope_kwargs),
    }
    AGENT_CACHE[app_id] = (scope_h, agents)
    return agents

# --- STM load (no cache) ---
async def get_short_memory_db(application_id: str) -> List[dict]:
    raw = await chat_service.load_short_memory(application_id)  # ascending by created_at
    return raw or []

# --- Team state save ---
async def save_team_state_direct(application_id: str, team, saver):
    await save_team_state(application_id, team, saver)

# ----------------------------
# Logging helpers for STM
# ----------------------------
LOG_MEMORY = os.getenv("CHAT_LOG_MEMORY", "0") == "1"
MEMORY_LOG_MAX_CHARS = int(os.getenv("CHAT_MEMORY_LOG_MAX_CHARS", "1000"))
DISABLE_TRUNCATION = os.getenv("CHAT_LOG_NO_TRUNCATE", "0") == "1"  # NEW

def _truncate(s: Optional[str], limit: int = MEMORY_LOG_MAX_CHARS) -> str:
    if s is None:
        return ""
    s = str(s)
    if DISABLE_TRUNCATION:
        return s
    return s if len(s) <= limit else s[:limit] + "…"
    

def _to_safe_log_item(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": m.get("id"),
        "role": m.get("role") or m.get("source") or "unknown",
        "created_at": m.get("created_at") or m.get("timestamp"),
        "content": _truncate(m.get("content") or ""),
    }

def _emit_log(logger: logging.Logger, payload: Dict[str, Any]):
    try:
        s = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        s = str(payload)
    try:
        logger.info(s)
    except Exception:
        pass
    print(s, flush=True)

def _log_memory_sample(logger: logging.Logger, *, run_id: str, app_id: str, short_memory: List[Dict[str, Any]], max_preview: int = 5):
    if not LOG_MEMORY:
        return
    n = len(short_memory)
    head = [_to_safe_log_item(m) for m in short_memory[:max_preview]]
    tail = [_to_safe_log_item(m) for m in short_memory[-max_preview:]] if n > max_preview else []
    _emit_log(logger, {
        "event": "short_memory_loaded",
        "run_id": run_id,
        "application_id": app_id,
        "total": n,
        "head": head,
        "tail": tail if tail else None,
    })

def _log_seeded_memory(logger: logging.Logger, *, run_id: str, app_id: str, seeded_rows: List[Dict[str, Any]]):
    if not LOG_MEMORY:
        return
    _emit_log(logger, {
        "event": "short_memory_seeded",
        "run_id": run_id,
        "application_id": app_id,
        "count": len(seeded_rows),
        "messages": [_to_safe_log_item(m) for m in seeded_rows],
    })

# ----------------------------
# Logger setup
# ----------------------------
def tnow() -> float:
    return time.perf_counter()

logger = logging.getLogger("chat_perf")
level_name = os.getenv("LOG_LEVEL", "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
logger.setLevel(level)

if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.getLogger().setLevel(level)

# ======================================================================================
# Persistence helpers (history only for non-user messages)
# ======================================================================================

def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return str(content)
    if isinstance(content, (list, dict, tuple)):
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(content)
    return str(content)

def _msg_fingerprint(source: str, content: Any) -> str:
    import hashlib
    s = (source or "unknown").strip()
    c = _normalize_content(content).strip()
    h = hashlib.sha256()
    h.update((s + "\n" + c).encode("utf-8"))
    return h.hexdigest()

async def persist_run_messages(application_id, messages, chat_service, skip_fingerprints=None):
    """
    Store all non-user messages to FULL history ONLY.
    This function no longer writes to STM.
    """
    skip_fingerprints = skip_fingerprints or set()
    for m in messages:
        source = getattr(m, "source", None) or getattr(m, "role", None) or "unknown"
        raw_content = getattr(m, "content", None)
        content = _normalize_content(raw_content)

        fp = _msg_fingerprint(source, content)
        if fp in skip_fingerprints:
            continue

        if source == "user":
            continue
        else:
            await chat_service.store_agent_message(
                application_id=application_id,
                role=source,
                content=content
            )

# ======================================================================================
# Optional: fallback team-state loader
# ======================================================================================

async def _try_load_team_state(application_id: str):
    if load_team_state:
        try:
            return await load_team_state(application_id, team_state_service.load)
        except Exception:
            pass
    try:
        loader = getattr(team_state_service, "load", None)
        if callable(loader):
            return await loader(application_id)
    except Exception:
        pass
    return None

# ======================================================================================
# Endpoints
# ======================================================================================

@app.get("/scope/jira/{application_id}")
async def get_jira_scope(application_id: str):
    try:
        scope = await scope_service.get_scope(application_id)
        def _to_upper_or_none(s: Optional[str]) -> Optional[str]:
            s = (s or "").strip()
            return s.upper() if s else None

        proj = _to_upper_or_none(scope.jira_projectkey)
        aha  = _to_upper_or_none(scope.aha_id)
        conf = _to_upper_or_none(scope.confluence_spacekey)
        return {
            "application_id": application_id,
            "jira_projects": [proj] if proj else [],
            "jira_issues": [],
            "aha_ids": [aha] if aha else [],
            "confluence_spacekeys": [conf] if conf else [],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Scope not found for {application_id}: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/clear-cache/{application_id}")
async def clear_agent_cache(application_id: str):
    """
    Clear cached agents for a specific application.
    Use this when scope has changed and you want to force agent rebuild.
    """
    global AGENT_CACHE
    
    if application_id in AGENT_CACHE:
        old_scope_hash, _ = AGENT_CACHE[application_id]
        del AGENT_CACHE[application_id]
        logger.info(f"[CACHE_CLEAR] Cleared agent cache for app {application_id} (old scope hash: {old_scope_hash[:8]}...)")
        return {
            "status": "success",
            "message": f"Agent cache cleared for application {application_id}",
            "application_id": application_id,
            "note": "Agents will be rebuilt with current scope on next request"
        }
    else:
        return {
            "status": "not_found",
            "message": f"No cached agents found for application {application_id}",
            "application_id": application_id,
            "note": "Agents will be built from scratch on next request"
        }

@app.post("/clear-short-memory/{application_id}")
async def clear_short_memory(application_id: str):
    deleted_count = await chat_service.delete_short_memory(application_id)
    if deleted_count > 0:
        return {
            "application_id": application_id,
            "deleted_rows": deleted_count,
            "message": "Short term chat history deleted."
        }
    else:
        return {
            "application_id": application_id,
            "deleted_rows": 0,
            "message": "No short term chat history found."
        }

@app.get("/metrics/rate-limit")
async def rate_limit_metrics():
    """Monitor current concurrency and optional rate limiting status"""
    async with ACTIVE_REQUESTS_LOCK:
        active_by_app = dict(ACTIVE_REQUESTS)
    
    return {
        "active_requests": {
            "total": sum(active_by_app.values()),
            "by_app": active_by_app,
        },
        "rate_limiter": {
            "enabled": ENABLE_RATE_LIMITER,
            "tokens_available": round(RATE_LIMITER.tokens, 2) if ENABLE_RATE_LIMITER else None,
            "capacity": RATE_LIMITER.capacity if ENABLE_RATE_LIMITER else None,
            "rate_per_second": RATE_LIMITER.rate if ENABLE_RATE_LIMITER else None,
        },
        "config": {
            "max_turns_per_run": MAX_TURNS_PER_RUN,
            "max_retries_rate_limit": MAX_RETRIES_RATE_LIMIT,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_recent_messages": MAX_RECENT_MESSAGES,
            "max_historical_messages": MAX_HISTORICAL_MESSAGES,
            "total_context_messages": MAX_RECENT_MESSAGES + MAX_HISTORICAL_MESSAGES,
        },
        "context_strategy": {
            "recent_messages": f"Last {MAX_RECENT_MESSAGES} messages included in full",
            "historical_messages": f"Messages 6-{MAX_RECENT_MESSAGES + MAX_HISTORICAL_MESSAGES} summarized by LLM (max 500 words)",
            "note": "Hybrid approach: Recent context is detailed, historical context is summarized"
        },
        "note": "✅ Semaphores removed - full concurrent access enabled"
    }

@app.get("/metrics/tokens")
async def token_usage_info():
    """
    Information about token usage and Azure OpenAI pricing.
    Use this to understand your costs and optimize token usage.
    """
    # Azure OpenAI pricing (as of 2024 - verify current rates)
    pricing = {
        "gpt-4": {
            "prompt": 0.03,      # $ per 1K tokens
            "completion": 0.06    # $ per 1K tokens
        },
        "gpt-4-32k": {
            "prompt": 0.06,
            "completion": 0.12
        },
        "gpt-4o": {
            "prompt": 0.005,
            "completion": 0.015
        },
        "gpt-35-turbo": {
            "prompt": 0.0015,
            "completion": 0.002
        }
    }
    
    return {
        "message": "Token usage is tracked per request",
        "how_to_see_usage": {
            "per_request": "Check the response's 'models_usage' field",
            "in_logs": "Search logs for [TOKEN_USAGE] entries",
            "breakdown": "Logs show per-agent token consumption"
        },
        "azure_pricing_reference": pricing,
        "cost_calculation": "cost = (prompt_tokens / 1000 * prompt_price) + (completion_tokens / 1000 * completion_price)",
        "optimization_tips": [
            "Reduce MAX_RECENT_MESSAGES if last 5 messages contain too much detail",
            "Reduce MAX_HISTORICAL_MESSAGES to limit how far back to look (6-20 range)",
            "Adjust MAX_CONTEXT_TOKENS to limit total conversation history size",
            "Use MAX_TURNS_PER_RUN to limit agent interactions",
            "Monitor per-agent usage to identify inefficient agents"
        ],
        "current_config": {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_recent_messages": MAX_RECENT_MESSAGES,
            "max_historical_messages": MAX_HISTORICAL_MESSAGES,
            "total_context_window": MAX_RECENT_MESSAGES + MAX_HISTORICAL_MESSAGES,
            "max_turns_per_run": MAX_TURNS_PER_RUN
        }
    }

@app.get("/history/{application_id}")
async def history(application_id: str) -> list[dict[str, Any]]:
    try:
        return await chat_service.load_full_history(application_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

JIRA_DOMAIN = os.environ["JIRA_DOMAIN"]
EMAIL = os.environ["JIRA_EMAIL"]
API_TOKEN = os.environ["JIRA_API_TOKEN"]

auth = HTTPBasicAuth(EMAIL, API_TOKEN)
headers = {"Accept": "application/json"}
def check_jira_access(project_key):
    try:
        url = f"{JIRA_DOMAIN}/rest/api/3/mypermissions"
        params = {
            "projectKey": project_key,
            "permissions": "BROWSE_PROJECTS"
        }
        response = requests.get(url,
                                params=params,
                                headers=headers, auth=auth, verify=False)
        # print("response3--> ", response.json())
        data = response.json()

        if ("permissions" in data) and ("BROWSE_PROJECTS" in data["permissions"]) and ("havePermission" in data["permissions"]["BROWSE_PROJECTS"]):
        # if data["permissions"]["BROWSE_PROJECTS"]["havePermission"]:
            print("Functional account is part of this JIRA project")
            return "Functional account is part of this JIRA project"
        else:
            print("Functional account is not part of this JIRA project")
            return "Functional account is not part of this JIRA project"
    except Exception as e:
        return "Error occurred related to JIRA project URL/Key. Please contact Administrator."

@app.post("/jira_validation/{project_key}")
def check_jira_access_method(project_key: str):
    try:
        result = check_jira_access(project_key)
        print("result-->  ", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

CONFLUENCE_BASE_URL = os.environ["CONF_BASE"]
def validate_confluence_page(page_id, space_key='RnDTOSS'):
    url = f"{CONFLUENCE_BASE_URL}/wiki/rest/api/content/{page_id}"
    params = {
        "expand": "space"
    }
    response = requests.get(
        url,
        params=params,
        auth=HTTPBasicAuth(EMAIL, API_TOKEN),
        headers = {"Accept": "application/json"},
    )
    if response.status_code != 200:
        return False
    try:
        data = response.json()
    except Exception as e:
        print("Invalid response from confluence API-->", response.text)
        return False
    return data["space"]["key"] == space_key
 
@app.post("/confluence_validation/{page_id}")
def check_confluence_access(page_id: str):
    try:
        result = validate_confluence_page(page_id)
        # print("result-->  ", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from uuid import uuid4

# ----------------------------
# Helpers for contextual seeding
# ----------------------------
MAX_CONTEXT_TOKENS = int(os.getenv("CHAT_MAX_CONTEXT_TOKENS", "3500"))  # Total budget: 3500 tokens
MAX_RECENT_MESSAGES = MAX_RECENT_MESSAGES_ENV      # Last 5 messages go as-is (configurable via env)
MAX_HISTORICAL_MESSAGES = MAX_HISTORICAL_MESSAGES_ENV  # Messages 6-20 (15 messages) get summarized (configurable via env)

def _estimate_tokens(s: str) -> int:
    # Rough heuristic: 1 token ~ 4 chars (varies by model)
    return max(1, len(s) // 4)

async def _summarize_historical_messages(messages: list[dict]) -> str:
    """
    Summarizes a list of historical messages (6-20) using LLM.
    Returns a concise summary (max 500 words) of the conversation flow.
    """
    if not messages:
        return ""
    
    # Format messages for summarization
    conversation_text = []
    for m in messages:
        role = (m.get("role") or m.get("source") or "unknown").upper()
        content = (m.get("content") or "").strip()
        if content:
            conversation_text.append(f"{role}: {content}")
    
    full_conversation = "\n\n".join(conversation_text)
    
    # Create summarization prompt
    summarization_messages = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. Your task is to create a concise summary "
                "of the conversation history below. Focus on:\n"
                "1. Key topics discussed\n"
                "2. Important decisions made\n"
                "3. Action items or tasks mentioned\n"
                "4. Any important context or requirements\n\n"
                "Keep the summary under 500 words and maintain chronological flow."
            )
        },
        {
            "role": "user",
            "content": f"Please summarize this conversation:\n\n{full_conversation}"
        }
    ]
    
    try:
        summary = await asyncio.to_thread(
            get_LLM_response,
            summarization_messages,
            temperature=0.3  # Lower temperature for more focused summaries
        )
        return summary.strip()
    except Exception as e:
        logger.warning(f"Failed to summarize historical messages: {e}")
        # Fallback: return truncated raw conversation
        return full_conversation[:2000] + "...[truncated]"

def _format_recent_messages(messages: list[dict]) -> str:
    """
    Formats the last 5 messages as-is (full content).
    Each row is a dict with {role/source, content, ...}
    """
    lines = []
    for idx, m in enumerate(messages, 1):
        role = (m.get("role") or m.get("source") or "unknown").upper()
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{idx}) {role}: {content}")
    return "\n".join(lines)

async def _build_context_block(seeded_rows: list[dict], scope_kwargs: dict | None = None) -> str:
    """
    Builds a hybrid context block:
    - Last 5 messages: Full content (as-is)
    - Messages 6-20: Summarized by LLM (max 500 words)
    
    This provides detailed recent context while keeping older context concise.
    """
    total_messages = len(seeded_rows)
    print(f"[CONTEXT_DEBUG] Building context block from {total_messages} seeded messages", flush=True)
    # Split messages into recent (last 5) and historical (6-20)
    recent_messages = seeded_rows[-MAX_RECENT_MESSAGES:] if total_messages > 0 else []
    
    # Get historical messages (6-20 from the end)
    historical_start = max(0, total_messages - MAX_RECENT_MESSAGES - MAX_HISTORICAL_MESSAGES)
    historical_end = max(0, total_messages - MAX_RECENT_MESSAGES)
    historical_messages = seeded_rows[historical_start:historical_end] if historical_end > historical_start else []
    
    # Build context sections
    context_sections = []
    
    # 1. CURRENT SCOPE (PRIORITY - Always first and emphasized)
    # This ensures latest scope takes precedence over historical context
    scope_lines = []
    if scope_kwargs:
        if scope_kwargs.get("jira_projects"):
            scope_lines.append(f"JIRA Projects: {', '.join(scope_kwargs['jira_projects'])}")
        if scope_kwargs.get("aha_ids"):
            scope_lines.append(f"Aha IDs: {', '.join(scope_kwargs['aha_ids'])}")
        if scope_kwargs.get("confluence_spacekeys"):
            scope_lines.append(f"Confluence Spaces: {', '.join(scope_kwargs['confluence_spacekeys'])}")
    
    scope_text = ("\n".join(scope_lines)).strip()
    if scope_text:
        context_sections.append(
            f"[CURRENT SCOPE - Use These Values]\n{scope_text}\n"
            f"Note: Always use the above JIRA/Confluence/Aha IDs from current scope. "
            f"If historical context mentions different project IDs, ignore them and use the current scope values above."
        )
    
    # 2. Historical summary (if exists)
    if historical_messages:
        logger.info(f"Summarizing {len(historical_messages)} historical messages (positions {historical_start+1} to {historical_end})")
        summary = await _summarize_historical_messages(historical_messages)
        if summary:
            context_sections.append(
                f"[Earlier Conversation Summary — {len(historical_messages)} messages]\n{summary}"
            )
    
    # 3. Recent messages (full content)
    if recent_messages:
        recent_text = _format_recent_messages(recent_messages)
        context_sections.append(
            f"[Recent Messages — Last {len(recent_messages)} turns]\n{recent_text}"
        )
    
    # 4. Guidance
    header = (
        "You are a coordinated team. Use the conversation context faithfully and continue the dialogue"
        " without repeating or contradicting prior decisions. If context is insufficient, ask a brief"
        " clarifying question before proceeding.\n"
        "- Prefer concise, actionable answers.\n"
        "- Do not restate the entire context; refer to it implicitly.\n"
        "- If referencing artifacts (JIRA/Confluence/Aha), keep IDs explicit.\n"
        "- please always use project IDs from CURRENT SCOPE section above, not from historical context."
    )
    context_sections.append(f"[Guidance]\n{header}")
    
    # Combine all sections
    block = "\n\n".join(context_sections)
    
    # Truncate entire block to fit MAX_CONTEXT_TOKENS budget (safety check)
    tokens = _estimate_tokens(block)
    if tokens > MAX_CONTEXT_TOKENS:
        approx_chars = MAX_CONTEXT_TOKENS * 4
        block = block[:approx_chars] + "\n\n…[context truncated to fit token budget]"
        logger.warning(f"Context block truncated from {tokens} to ~{MAX_CONTEXT_TOKENS} tokens")
    
    logger.info(f"Built context block: {len(historical_messages)} summarized + {len(recent_messages)} recent messages (~{_estimate_tokens(block)} tokens)")
    
    return block


@app.post("/chat", response_model=ChatOut)
async def chat(request: ChatIn) -> ChatOut:
    """
    Simplified chat endpoint with true concurrent multi-user support.
    No semaphores - all requests run in parallel.
    Optional rate limiter can be enabled via ENABLE_RATE_LIMITER=1.
    """
    # Validate input
    if not request.content or not request.content.strip():
        logger.warning(f"[VALIDATION_FAILED] Empty user input for app {request.application_id}")
        return ChatOut(
            id=None,
            source="system",
            models_usage=None,
            metadata={
                "error": True,
                "error_type": "validation_error",
                "error_message": "Empty input provided"
            },
            created_at=datetime.now(timezone.utc),
            content="Please enter a valid query. Your message cannot be empty.",
            data=[{
                "id": None,
                "source": "system",
                "metadata": {"error": True},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content": "Please enter a valid query. Your message cannot be empty.",
                "type": "TextMessage"
            }],
            mime_type="text/plain",
            type="TextMessage",
            final_output="Please enter a valid query. Your message cannot be empty.",
            critic_review=None
        )
    
    # Optional rate limiting - only if explicitly enabled
    if ENABLE_RATE_LIMITER:
        await RATE_LIMITER.acquire(1)
    
    # Track active requests for monitoring (non-blocking)
    async with ACTIVE_REQUESTS_LOCK:
        ACTIVE_REQUESTS[request.application_id] += 1
    
    t0 = tnow()
    run_id = str(uuid4())
    
    logger.info(f"[CONCURRENT] Processing request {run_id} for app {request.application_id} (active: {ACTIVE_REQUESTS[request.application_id]})")

    try:
        # 1) Load STM from DB (no cache)
        short_memory = await get_short_memory_db(request.application_id)
        _log_memory_sample(logger, run_id=run_id, app_id=request.application_id, short_memory=short_memory)

        print(f"[STM_DEBUG] loaded_from_db={len(short_memory)} rows", flush=True)

        # Prepare up to last 20 messages for context (5 recent + 15 historical for summary)
        max_context_messages = MAX_RECENT_MESSAGES + MAX_HISTORICAL_MESSAGES  # 5 + 15 = 20
        seeded_rows = short_memory[-max_context_messages:] if len(short_memory) > max_context_messages else short_memory
        
        print(f"[STM_DEBUG] to_seed={len(seeded_rows)} rows (last {max_context_messages} messages max)", flush=True)
        print(
            "[STM_DEBUG] seeded_ids_roles=" +
            str([(m.get('id'), m.get('role')) for m in seeded_rows]),
            flush=True
        )

        # Build fingerprints from STM to avoid duplicates when persisting history
        prev_fps: set[str] = set()
        for m in short_memory:
            role = m.get("role") or m.get("source") or "unknown"
            content = m.get("content") or ""
            prev_fps.add(_msg_fingerprint(role, content))
        print(f"[STM_DEBUG] prev_fps_count={len(prev_fps)}", flush=True)

        # 2) Store USER message first (history + STM)
        await chat_service.store_user_message(request.application_id, request.content)

        # 3) Fetch scope
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

        # 4) Build/reuse agents
        agents = get_or_build_agents(request.application_id, scope_kwargs)
        po_agent, qa_agent, tm_agent, orch_agent, critic_agent = (
            agents["po"], agents["qa"], agents["tm"], agents["orch"], agents["critic"]
        )
        selector_func, pick_final_output_from_last_main, MAIN_SOURCES_SET = make_selector_and_helpers(
            ORCHESTRATOR_NAME=orch_agent.name,
            CRITIC_NAME=critic_agent.name,
            PO_NAME=po_agent.name,
            QA_NAME=qa_agent.name,
            TM_NAME=tm_agent.name
        )
        # 5) Build team with backoff-enabled client and termination
        termination = TextMentionTermination("APPROVE") | MaxMessageTermination(MAX_TURNS_PER_RUN)
        backoff_model_client = BackoffModelClient(az_model_client)
        team = SelectorGroupChat(
            participants=[orch_agent, po_agent, qa_agent, tm_agent, critic_agent],
            model_client=backoff_model_client,
            selector_func=selector_func,
            allow_repeated_speaker=False,
            termination_condition=termination,
        )

        # 6) Build USER-WRAPPED CONTEXT MESSAGE with Hybrid Context (recent + summarized historical)
        _log_seeded_memory(logger, run_id=run_id, app_id=request.application_id, seeded_rows=seeded_rows)

        # Build context block with hybrid approach:
        # - Last 5 messages: Full content
        # - Messages 6-20: Summarized by LLM (max 500 words)
        context_block = await _build_context_block(seeded_rows, scope_kwargs)

        # Post ONCE as a strong system message so all agents see the same authoritative context
        context_msg = TextMessage(content=context_block, source="system")

        if hasattr(team, "post_message"):
            await team.post_message(context_msg)
        elif hasattr(team, "add_message"):
            team.add_message(context_msg)

        print(f"[CTX] Injected context system message (~{_estimate_tokens(context_block)} tokens)", flush=True)
        print(f"context {context_msg.content}", flush=True)

        # Try to introspect team buffer (if available in your autogen version)
        try:
            internal_msgs = getattr(team, "messages", None) or getattr(team, "_messages", None)
            if internal_msgs is not None:
                preview = [
                    {"source": getattr(m, "source", "unknown"),
                     "content": (getattr(m, "content", "") or "")[:160]}
                    for m in internal_msgs
                ]
                print(f"[TEAM_BUFFER] count={len(preview)} preview={preview[:8]}", flush=True)
            else:
                print("[TEAM_BUFFER] no buffer attribute exposed", flush=True)
        except Exception as _e:
            print(f"[TEAM_BUFFER] inspection failed: {_e}", flush=True)

        # 7) Run the team with the USER message           
        # Wrap context + new user message into ONE user prompt
        wrapped_user_prompt = f"""
        Below is the recent conversation context for this application. Please use it faithfully and continue the dialogue:

        {context_block} 

        Now here is the user's new message. Respond only to this new message, but using the above context whenever needed.

        [User Input]
        {request.content}
        """

        user_msg = TextMessage(content=wrapped_user_prompt.strip(), source="user")
        print(f"user_msg content: {user_msg.content[:500]}", flush=True)
        print(f"[RUN] Wrapped user_msg preview: {wrapped_user_prompt[:250]}", flush=True)

        # 7) Run the team with wrapped user message
        error_occurred = False
        error_message_obj = None
        
        try:
            result = await team.run(task=user_msg)
        except Exception as team_error:
            error_occurred = True
            
            # Classify and handle errors from team execution
            llm_error = _classify_llm_error(team_error)
            if llm_error:
                logger.error(f"LLM error during team.run(): {llm_error.message}")
                
                # Create error message in agent format
                error_content = llm_error.message
                if llm_error.status_code == 401:
                    error_content += "\n\n💡 This appears to be a configuration issue. Please contact your administrator."
                elif llm_error.status_code == 403:
                    error_content += "\n\n💡 You may need different permissions. Please contact your administrator."
                elif llm_error.status_code == 429:
                    retry_info = f" Please wait {llm_error.retry_after} seconds and try again." if llm_error.retry_after else " Please try again in a moment."
                    error_content += f"\n\n⏳{retry_info}"
                elif llm_error.status_code in {500, 502, 503, 504}:
                    error_content += "\n\n🔄 This is a temporary issue. Please try again in a moment."
                
                error_message_obj = TextMessage(
                    content=error_content,
                    source="system",
                    metadata={
                        "error": "True",
                        "error_type": "llm_error",
                        "error_code": llm_error.status_code,
                        "recoverable": llm_error.status_code in {429, 500, 502, 503, 504},
                        "retry_after": llm_error.retry_after
                    }
                )
            else:
                logger.error(f"Error during team.run(): {str(team_error)[:500]}")
                
                # Create generic error message
                error_message_obj = TextMessage(
                    content="An unexpected error occurred while processing your request. Please try again in a moment.",
                    source="system",
                    metadata={
                        "error": "True",
                        "error_type": "unexpected_error",
                        "error_message": str(team_error)[:500] if os.getenv("DEBUG") == "1" else "Unexpected error"
                    }
                )
                 # Create a mock result with just the error message
            class MockResult:
                def __init__(self, messages):
                    self.messages = messages
            
            result = MockResult(messages=[error_message_obj])

        if not hasattr(result, "messages") or not result.messages:
            # Create error message for no response
            error_message_obj = TextMessage(
                content="I apologize, but I couldn't generate a response. Please try again.",
                source="system",
                metadata={
                    "error": "True",
                    "error_type": "no_response",
                    "error_message": "Team returned no messages"
                }
            )
            result.messages = [error_message_obj]
            error_occurred = True

        # 8) Save team state (only if no error)
        if not error_occurred:
            await save_team_state_direct(request.application_id, team, team_state_service.save)

        # 9) Persist all non-user messages to FULL history only (only if no error)
        if not error_occurred:
            await persist_run_messages(
                application_id=request.application_id,
                messages=result.messages,
                chat_service=chat_service,
                skip_fingerprints=prev_fps
            )

        # 10) Final selection - works for both success and error messages
        final = pick_final_output_from_last_main(result.messages)

        # 10.1) Persist FINAL to STM (only)
        final_content = (final.get("content") or "").strip()
        # Normalize final role to "assistant" for cleaner STM (optional but recommended)
        final_source  = "assistant"
        if final_content:
            await chat_service.store_final_message(
                application_id=request.application_id,
                role=final_source,
                content=final_content
            )
            print("[STM_DEBUG] final persisted to STM", flush=True)

        # 11) Critic review (optional)
        final_critic_msg_obj = None
        for m in reversed(result.messages):
            if getattr(m, "source", None) == critic_agent.name:
                final_critic_msg_obj = m
                break
        critic_review_str = None
        if final_critic_msg_obj is not None:
            critic_review_str, _, _ = extract_payload_from_message(final_critic_msg_obj)

        # 12) Token usage aggregation with per-agent breakdown
        usage = None
        token_breakdown = {}
        try:
            prompt_total, completion_total = 0, 0
            
            # Track tokens per agent
            for m in result.messages:
                source = getattr(m, "source", "unknown")
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
                
                # Log detailed token breakdown
                logger.info({
                    "event": "token_usage",
                    "run_id": run_id,
                    "application_id": request.application_id,
                    "total": {
                        "prompt_tokens": prompt_total,
                        "completion_tokens": completion_total,
                        "total_tokens": prompt_total + completion_total
                    },
                    "per_agent": token_breakdown
                })
                
                print(f"[TOKEN_USAGE] Total: {prompt_total + completion_total} tokens (prompt: {prompt_total}, completion: {completion_total})", flush=True)
                for agent, tokens in token_breakdown.items():
                    print(f"  └─ {agent}: {tokens['total_tokens']} tokens (prompt: {tokens['prompt_tokens']}, completion: {tokens['completion_tokens']})", flush=True)
        except Exception as e:
            logger.warning(f"Failed to calculate token usage: {e}")
            usage = None
            # 13) Build response
        created_at = datetime.now(timezone.utc)
        final_main_msg_obj = None
        if final.get("source"):
            for m in reversed(result.messages):
                if getattr(m, "source", None) == final["source"] and getattr(m, "content", None) == final["content"]:
                    final_main_msg_obj = m
                    break

        msg_id = getattr(final_main_msg_obj, "id", None) if final_main_msg_obj else None
        msg_source = final.get("source") or "assistant"
        msg_metadata = getattr(final_main_msg_obj, "metadata", {}) if final_main_msg_obj else {}
        msg_type = "TextMessage"

        # 14) Timings
        timings_text = f"""
timings (seconds):
  total:            {tnow() - t0:.3f}
"""
        print(timings_text)
        logger.info({"event": "timings", "run_id": run_id, "application_id": request.application_id, "timings": timings_text})

        # Data payload (full run messages as dicts)
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

    except LLMAPIError as e:
        # LLM-specific errors - already classified, just re-raise for handler
        logger.error(f"LLM API Error in chat endpoint: {e.status_code} - {e.message}")
        raise
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"Unhandled error in chat endpoint: {str(e)}")
        traceback.print_exc()
        
        # Build user-friendly error message based on error type
        error_str = str(e).lower()
        
        if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
            error_message = "⏳ Too many requests. Please wait a moment and try again."
            error_type = "rate_limit_error"
            recoverable = True
            logger.error("429 error received --> ", e)
        elif "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
            error_message = "🔐 Authentication issue detected. Please contact your administrator."
            error_type = "auth_error"
            recoverable = False
            logger.error("401 error received --> ", e)
        elif "403" in error_str or "forbidden" in error_str or "access denied" in error_str:
            error_message = "🚫 Access denied. Please contact your administrator for permissions."
            error_type = "permission_error"
            recoverable = False
        elif "503" in error_str or "service unavailable" in error_str:
            error_message = "🔄 Service temporarily unavailable. Please try again shortly."
            error_type = "service_unavailable"
            recoverable = True
            logger.error("503 error received --> ", e)
        elif "500" in error_str or "502" in error_str or "504" in error_str:
            error_message = "⚠️ Server error encountered. Please try again in a moment."
            error_type = "server_error"
            recoverable = True
            logger.error("500/502/504 error received --> ", e)
        elif "timeout" in error_str:
            error_message = "⏱️ Request timed out. Please try again."
            error_type = "timeout_error"
            recoverable = True
            logger.error("Timeout error received --> ", e)
        elif "database" in error_str or "connection" in error_str:
            error_message = "💾 Connection issue detected. Please try again shortly."
            error_type = "connection_error"
            recoverable = True
            logger.error("Database/connection error received --> ", e)
        elif "400" in error_str or "bad request" in error_str:
            error_message = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry."
            error_type = "bad_request_error"
            recoverable = False
            logger.error("400 error received --> ", e)
        elif "connection error" in error_str.lower():
            error_message = "Failed to establish a new connection. Please try after sometime or contact system administrator."
            error_type = "connection_error"
            recoverable = True
            logger.error("connection error received --> ", e)
        else:
            error_message = "❌ Unexpected error occurred. Please try again or contact support."
            error_type = "unexpected_error"
            recoverable = True
            logger.error("Unexpected error received --> ", e)
        return ChatOut(
            id=None,
            source="system",
            models_usage=None,
            metadata={
                "error": "True",
                "error_type": error_type,
                "error_message": str(e)[:500] if os.getenv("DEBUG") == "1" else "Error details hidden",
                "recoverable": recoverable
            },
            created_at=datetime.now(timezone.utc),
            content=error_message,
            data=[{
                "id": None,
                "source": "system",
                "metadata": {"error": "True"},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content": error_message,
                "type": "TextMessage"
            }],
            mime_type="text/plain",
            type="TextMessage",
            final_output=error_message,
            critic_review=None
        )
    finally:
        # Decrement active request counter (guaranteed cleanup)
        async with ACTIVE_REQUESTS_LOCK:
            ACTIVE_REQUESTS[request.application_id] -= 1
            if ACTIVE_REQUESTS[request.application_id] <= 0:
                del ACTIVE_REQUESTS[request.application_id]
        
        logger.info(f"[CONCURRENT] Completed request {run_id} for app {request.application_id}")
# ----------------------------
# DB bootstrap
# ----------------------------
from src.db_model_mssql.session import init_db

@app.on_event("startup")
async def on_startup():
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed (non-fatal): {e}")
        # Continue startup even if DB fails - allows testing without DB connection
        
