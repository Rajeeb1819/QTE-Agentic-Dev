
import os, time, threading, requests, asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

class KongBearerTokenProvider:
    """
    A callable, concurrency-safe provider that returns a valid Bearer token.
    It refreshes automatically before expiry and avoids thundering herds.
    """
    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str = "openid email profile",
        refresh_margin_sec: int = 300
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.refresh_margin_sec = refresh_margin_sec
        self._token = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            now = time.time()
            if self._token and now < (self._expires_at - self.refresh_margin_sec):
                return self._token

            resp = requests.post(
                self.token_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                    "scope": self.scope,
                },
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 3600))
            self._expires_at = now + expires_in
            return self._token


kong_token_provider = KongBearerTokenProvider(
    token_url=os.environ["LLM_AUTH_URL"],
    client_id=os.environ["LLM_CLIENT_ID"],
    client_secret=os.environ["LLM_CLIENT_SECRET"],
)

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o",                       # must match your upstream deployment
    model="gpt-4o",
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["LLM_OPENAI_URL"],    # Kong base URL that proxies Azure OpenAI-compatible API
    azure_ad_token_provider=kong_token_provider,    # <-- provider (sync)
)

__all__ = ["az_model_client", "kong_token_provider"]



# llm/retry_wrapper.py
from typing import AsyncIterator, List, Union
from autogen_core.models import ChatCompletionClient, CreateResult
from openai import AuthenticationError

class RetryingChatCompletionClient(ChatCompletionClient):
    """
    Wraps a ChatCompletionClient and retries once on AuthenticationError (401).
    On retry, it forces a token refresh via the provided provider.
    """
    def __init__(self, inner: ChatCompletionClient, token_provider, retries: int = 1):
        self._inner = inner
        self._token_provider = token_provider
        self._retries = retries

    async def create(self, messages, **kwargs) -> CreateResult:
        for attempt in range(self._retries + 1):
            try:
                return await self._inner.create(messages=messages, **kwargs)
            except AuthenticationError:
                self._token_provider.force_refresh()
                if attempt >= self._retries:
                    raise

    async def create_stream(self, messages, **kwargs) -> AsyncIterator[Union[str, CreateResult]]:
        for attempt in range(self._retries + 1):
            try:
                async for chunk in self._inner.create_stream(messages=messages, **kwargs):
                    yield chunk
                return
            except AuthenticationError:
                # If any partial stream was yielded, we cannot "resume" it;
                # callers should handle partial output if needed.
                self._token_provider.force_refresh()
                if attempt >= self._retries:
                    raise

from openai import AuthenticationError

async def create_with_retry(messages, retries=1):
    try:
        return await az_model_client.create(messages)
    except AuthenticationError:
        # Force provider to refresh on next call
        if isinstance(kong_token_provider, KongBearerTokenProvider):
            kong_token_provider._token = None
            kong_token_provider._expires_at = 0
        if retries > 0:
            return await create_with_retry(messages, retries - 1)
        raise



# retrying_model_client = RetryingChatCompletionClient(
#     inner=az_model_client,
#     token_provider=kong_token_provider,
#     retries=1,   # 1 retry is usually enough
# )


# async def main():
#     result = await create_with_retry([
#         UserMessage(content="What is bhubaneswar?", source="user")
#     ])
#     print(result)

# asyncio.run(main())
