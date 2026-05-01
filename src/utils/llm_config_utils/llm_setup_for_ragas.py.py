import os
import requests
from urllib import parse
from typing import Tuple
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

RAGAS_CONFIG=False

# def get_access_token() -> str:
#     """Fetch OAuth2 client-credentials bearer token from env-configured auth endpoint."""
#     headers = {"Content-Type": "application/x-www-form-urlencoded"}
#     data = {
#         "client_id": os.environ["LLM_CLIENT_ID"],
#         "client_secret": os.environ["LLM_CLIENT_SECRET"],
#         "grant_type": "client_credentials",
#         "scope": os.environ.get("LLM_SCOPE", "openid email profile"),
#     }
#     resp = requests.post(os.environ["LLM_AUTH_URL"], headers=headers, data=data, timeout=20)
#     resp.raise_for_status()
#     return resp.json()["access_token"]


# def derive_model_and_base_url(
#     full_chat_url: str | None = None,
#     base_url_env: str | None = None,
# ) -> Tuple[str, str]:
#     """Return (model, base_url ending with /openai/deployments/<model>)."""
#     model = os.environ.get("LLM_MODEL", "gpt-4")
#     base_url = base_url_env or os.environ.get("OPENAI_BASE_URL")

#     full_chat_url = (full_chat_url or os.environ.get("LLM_OPENAI_URL", "")).strip().strip('"')
#     if full_chat_url and "/openai/deployments/" in full_chat_url:
#         parsed = parse.urlparse(full_chat_url)
#         path = parsed.path
#         try:
#             after = path.split("/openai/deployments/")[1]
#             dep = after.split("/")[0]
#             model = dep or model
#         except Exception:
#             pass
#         root = f"{parsed.scheme}://{parsed.netloc}" + path.split("/openai/")[0]
#         base_url = f"{root.rstrip('/')}/openai/deployments/{model}"

#     if not base_url:
#         raise RuntimeError("OPENAI_BASE_URL is not set or derivable from LLM_OPENAI_URL.")

#     return model, base_url


def build_async_openai_client(token: str, base_url: str, api_version: str | None = None) -> AsyncOpenAI:
    """Create AsyncOpenAI configured for Azure gateway with api-version query."""
    default_query = {"api-version": api_version or os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview")}
    return AsyncOpenAI(api_key=token, base_url=base_url, default_query=default_query)


def build_sync_openai_client(token: str, base_url: str, api_version: str | None = None) -> OpenAI:
    """Create sync OpenAI client (useful for embeddings)."""
    default_query = {"api-version": api_version or os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview")}
    return OpenAI(api_key=token, base_url=base_url, default_query=default_query)


def build_ragas_llm(async_client: AsyncOpenAI, model: str):
    """Create Ragas LLM via llm_factory using the provided AsyncOpenAI client and model."""
    return llm_factory(model=model, client=async_client, temperature= 1, max_tokens=10000)



def build_embeddings(sync_client: OpenAI) -> OpenAIEmbeddings:
    """Create OpenAIEmbeddings using the provided sync client."""
    return OpenAIEmbeddings(client=sync_client)


def build_async_embeddings(async_client: AsyncOpenAI) -> OpenAIEmbeddings:
    """Create OpenAIEmbeddings using the provided async client for aembed_text support."""
    return OpenAIEmbeddings(client=async_client)
