# rag_cli.py
# Robust RAG CLI using Kong AOAI + Azure Cognitive Search (Hybrid + BM25 exact)
import requests
import json
import re,os
from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.identity import ClientSecretCredential


TENANT_ID = os.environ.get("TENANT_ID")
SP_CLIENT_ID = os.environ.get("SP_CLIENT_ID")
SP_CLIENT_SECRET = os.environ.get("SP_CLIENT_SECRET")
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_JIRA = os.environ.get("AZURE_SEARCH_INDEX_JIRA")
AZURE_SEARCH_INDEX_pdf = os.environ.get("AZURE_SEARCH_INDEX_pdf")

llm_client_id=os.environ.get("LLM_CLIENT_ID")
llm_client_secret=os.environ.get("LLM_CLIENT_SECRET")
llm_auth_url=os.environ.get("LLM_AUTH_URL")
kong_base_url=os.environ.get("kong_base_url")
open_ai_embedding_model=os.environ.get("open_ai_embedding_model")
open_ai_chat_model_deploy_name=os.environ.get("LLM_MODEL")

# ---------------- Authentication ----------------
def get_kong_token_with_client_id_and_client_secret(client_id, client_secret):
    """Obtain a Kong OAuth2 access token using client credentials from config."""
    url = "https://federation-qa.gsk.com/as/token.oauth2"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": llm_client_id,
        "client_secret": llm_client_secret,
        "grant_type": "client_credentials",
        "scope": "openid email profile",
    }
    resp = requests.post(url, headers=headers, data=data)
    resp.raise_for_status()
    return json.loads(resp.text)["access_token"]

# Kong AOAI client
bearer_access_token = get_kong_token_with_client_id_and_client_secret(
    client_id=llm_client_id, client_secret=llm_client_secret
)
kong_aoai_client = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint=kong_base_url,
    azure_ad_token=bearer_access_token
)

# Azure Search client (Service Principal)
spn_credential = ClientSecretCredential(
    tenant_id=TENANT_ID,
    client_id=SP_CLIENT_ID,
    client_secret=SP_CLIENT_SECRET,
)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_pdf,
    credential=spn_credential
)

# ---------------- Helpers ----------------
def aoai_embedder(content_to_embed, aoai_client, embed_model):
    """Create embeddings for the given string using Azure OpenAI."""
    response = aoai_client.embeddings.create(input=content_to_embed, model=embed_model)
    return response.data[0].embedding[:]  # list[float]

def _is_id_like(q: str) -> bool:
    """Detect IDs like ABC-12345 anywhere in the query."""
    return bool(re.search(r"[A-Za-z]{2,}-\d{2,}", q or ""))

def extract_id_from_query(q: str) -> str:
    """Extract the first ID-like token from the query."""
    m = re.search(r"[A-Za-z]{2,}-\d{2,}", q or "")
    return m.group(0) if m else q

def _truncate_context(text: str, max_chars: int = 12000) -> str:
    """Truncate large context blocks to keep LLM prompt size under control."""
    return text[:max_chars] + "\n\n[...truncated...]" if len(text) > max_chars else text

# ---------------- BM25 Exact Phrase ----------------
def bm25_exact(search_client, query_text, select_fields=None, top=10):
    """
    Exact phrase BM25 search (quotes force phrase matching).
    Uses only fields that exist in your index: id, content.
    """
    if select_fields is None:
        select_fields = ["id", "content"]
    phrase = f"\"{query_text}\""  # exact phrase
    results_iter = search_client.search(
        search_text=phrase,
        search_fields=["content"],   # restrict BM25 to content field
        select=select_fields,
        top=top,
        search_mode="all",           # all terms must match
        query_type="full"            # Lucene syntax (quotes matter)
    )
    results = []
    for r in results_iter:
        d = dict(r)
        if "@search.score" not in d and hasattr(r, "score"):
            d["@search.score"] = r.score
        results.append(d)
    return results

# ---------------- Hybrid Search (BM25 + Vector) ----------------
def vector_search(query_text, knn, vector_field, search_client, search_fields=None, select_fields=None):
    """
    HYBRID: BM25 + VectorizedQuery. Returns list[dict] with @search.score preserved.
    """
    if select_fields is None:
        select_fields = ["id", "content"]

    # Embed the plain text query with the SAME model used for indexing (3-large → 3072 dims).
    embedded_query = aoai_embedder(query_text, kong_aoai_client, open_ai_embedding_model)
    vq = VectorizedQuery(vector=embedded_query, k_nearest_neighbors=knn, fields=vector_field)

    bm25_text = query_text or "*"
    search_kwargs = {
        "search_text": bm25_text,
        "vector_queries": [vq],
        "select": select_fields,
        "top": knn
    }
    if search_fields:
        search_kwargs["search_fields"] = search_fields

    results_iter = search_client.search(**search_kwargs)
    results = []
    for r in results_iter:
        d = dict(r)
        if "@search.score" not in d and hasattr(r, "score"):
            d["@search.score"] = r.score
        results.append(d)
    return results

def vector_search_results_list_to_str_block(results_list):
    """Convert search result dicts to a compact text block for prompting."""
    if not results_list:
        return ""
    keys_to_use = ["id", "content", "@search.score"]
    retrieved_content_text = ""
    for r in results_list:
        for key in keys_to_use:
            if key in r:
                retrieved_content_text += f"{key}:{str(r[key])}\n"
        retrieved_content_text += "\n\n"
    return retrieved_content_text

def make_openai_call_with_retrieved_data(user_message, retrieved_content_text, aoai_client):
    """Send RAG-augmented prompt to the chat model and return the text answer."""
    prompt = f"""
Answer strictly from the CONTEXT. If the answer is not present, say you don't know.

QUESTION:
{user_message}

CONTEXT:
{retrieved_content_text}
"""
    messages = [
        {"role": "system", "content": "You are an AI assistant that answers accurately and concisely."},
        {"role": "user", "content": prompt}
    ]
    response = aoai_client.chat.completions.create(
        model=open_ai_chat_model_deploy_name,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content

# ---------------- RAG Chat (robust flow) ----------------
def rag_chat(
    user_message,
    aoai_client,
    search_client,
    knn_n=10,
    minimum_similarity_score=0.7,
    verbose=False
):
    """
    Robust RAG flow:
      1) If the query contains an ID-like token (e.g., RTDE-31789),
         run BM25 exact phrase on that token first (deterministic hit).
      2) Otherwise, run hybrid search (BM25 + vector).
      3) If thresholding yields nothing, fall back to top-K by score.
      4) Print Top-10 IDs and scores when verbose=True.
    """

    # ---------- Path A: ID-like query → BM25 exact phrase ----------
    if _is_id_like(user_message):
        id_value = extract_id_from_query(user_message)
        exact_hits = bm25_exact(search_client, id_value, select_fields=["id", "content"], top=10)

        if verbose:
            print(f"\n[debug] BM25 exact phrase for ID '{id_value}' → {len(exact_hits)} hits")
            print("[debug] Top results:")
            for i, r in enumerate(exact_hits[:10], start=1):
                snippet = str(r.get("content", ""))[:100].replace("\n", " ")
                print(f"{i}. id: {r.get('id')} | score: {r.get('@search.score')} | content: {snippet}...")
            print("\n--------------------------------------\n")

        if exact_hits:
            retrieved_content_text = vector_search_results_list_to_str_block(exact_hits)
            retrieved_content_text = _truncate_context(retrieved_content_text, max_chars=12000)
            return make_openai_call_with_retrieved_data(
                user_message=user_message,
                retrieved_content_text=retrieved_content_text,
                aoai_client=aoai_client
            )
        # If no exact hits (rare), fall through to hybrid.

    # ---------- Path B: General query → Hybrid (BM25 + Vector) ----------
    results_list = vector_search(
        query_text=user_message,
        knn=knn_n,
        vector_field="contentVector",        # your vector field
        search_client=search_client,
        search_fields=["content"],           # restrict BM25 to content
        select_fields=["id", "content"]
    )

    # Filter by score (be lenient because hybrid yields varied scales)
    filtered_results = [
        r for r in results_list
        if float(r.get("@search.score", 0.0)) >= float(minimum_similarity_score)
    ]

    # Fallback to top-K by score if nothing passes the threshold
    if not filtered_results:
        filtered_results = sorted(
            results_list,
            key=lambda x: x.get("@search.score", 0.0),
            reverse=True
        )[:max(5, knn_n)]
        if verbose:
            print("[warn] No results passed threshold; using top-K fallback.")

    # ---------- Debug: show Top-10 IDs and scores ----------
    if verbose:
        print(f"\n[debug] Hybrid hits: {len(filtered_results)}")
        print("[debug] Top results:")
        for i, r in enumerate(filtered_results[:10], start=1):
            snippet = str(r.get("content", ""))[:100].replace("\n", " ")
            print(f"{i}. id: {r.get('id')} | score: {r.get('@search.score')} | content: {snippet}...")
        print("\n--------------------------------------\n")

    # Build context and call LLM
    retrieved_content_text = vector_search_results_list_to_str_block(filtered_results)
    retrieved_content_text = _truncate_context(retrieved_content_text, max_chars=12000)
    return make_openai_call_with_retrieved_data(
        user_message=user_message,
        retrieved_content_text=retrieved_content_text,
        aoai_client=aoai_client
    )

# ---------------- CLI ----------------
if __name__ == "__main__":
    print("RAG via Kong AOAI + Azure AI Search")
    print("Type your question and press Enter. Ctrl+C to exit.\n")
    try:
        while True:
            q = input(">> ").strip()
            if not q:
                continue
            answer = rag_chat(
                q,
                aoai_client=kong_aoai_client,
                search_client=search_client,
                knn_n=10,
                minimum_similarity_score=0.05,
                verbose=True  # set to False if you prefer quiet mode
            )
            print("\nAnswer:\n", answer, "\n")
    except KeyboardInterrupt:
        print("\nExiting...")