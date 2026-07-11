#!/usr/bin/env python3
"""
embed_cli.py
------------
Embed multiple PDFs using Azure OpenAI (via Kong) and upload to Azure Cognitive Search.
- Uses credentials and endpoints from `config.py` (already wired in your environment).
- No Streamlit; simple Python CLI.
- Single file combining PDF processing + embedding + upload.
- No helper function to rename IDs; uses chunk IDs as-is.
- No exceptions for missing config values; assumes `config.py` provides them.

Usage:
  python embed_cli.py file1.pdf file2.pdf --chunk-size 500 --chunk-overlap 200 --batch-size 1000
  python embed_cli.py file1.pdf --no-upload
"""

import os
import io
import json
import time
import argparse
from typing import List, Dict, Union, Iterable
from pathlib import Path

import pdfplumber
import requests
from openai import AzureOpenAI
from azure.identity import ClientSecretCredential
from azure.search.documents import SearchClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

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

# -----------------------------


def process_pdf(
    pdf_bytes: bytes,
    source_name: str,
    applicationId: int,
    chunk_size: int = 500,
    chunk_overlap: int = 200,
) -> Dict:
 
    print(f"file: {source_name}")
    print(f"applicationId: {applicationId}")
    all_text: str = ""
    all_tables: List[Dict] = []
    text_chunks: List[Dict] = []
 
    pdf_stream = io.BytesIO(pdf_bytes)
 
    with pdfplumber.open(pdf_stream) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
 
            # -------- TEXT --------
            text = page.extract_text() or ""
            all_text += text + "\n"
 
            # -------- TABLES --------
            tables = page.extract_tables() or []
            for table in tables:
                if not table or len(table) < 2:
                    continue
 
                headers = table[0]
                headers = [
                    h if (h and str(h).strip()) else f"col_{i}"
                    for i, h in enumerate(headers)
                ]
 
                for row in table[1:]:
                    row = (row or []) + [None] * max(0, len(headers) - len(row or []))
 
                    row_dict = {
                        headers[i]: (row[i] if i < len(row) else None)
                        for i in range(len(headers))
                    }
                    row_dict["page"] = page_num
 
                    all_tables.append(row_dict)
 
    # -------- CHUNKING --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
 
    chunks = splitter.split_text(all_text)
 
    stem = source_name.rsplit(".", 1)[0]
 
    for i, chunk in enumerate(chunks):
        text_chunks.append({
            "id": f"{stem}_chunk{i+1}",
            "content": chunk,
            "applicationId": applicationId
        })
 
    return {
        "source": source_name,
        "text_chunks": text_chunks,
        "tables": all_tables,
        "applicationId":applicationId
    }


# -----------------------------
# Azure OpenAI (via Kong)
# -----------------------------

def get_kong_token() -> str:
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
    return json.loads(resp.text)["access_token"]


def build_aoai_client() -> AzureOpenAI:
    token = get_kong_token()
    return AzureOpenAI(
        api_version="2024-02-15-preview",
        azure_endpoint=kong_base_url,
        azure_ad_token=token,
    )

def aoai_embedder(content_to_embed: str, aoai_client: AzureOpenAI) -> List[float]:
    resp = aoai_client.embeddings.create(input=content_to_embed, model=open_ai_embedding_model)
    return resp.data[0].embedding[:]

# -----------------------------
# Embedding workflow (multiple PDFs)
# -----------------------------

def embed_pdfs(pdf_bytes,source_name,applicationId, chunk_size: int = 500, chunk_overlap: int = 200) -> Dict[str, List[Dict]]:
    aoai_client = build_aoai_client()
    results: Dict[str, List[Dict]] = {}
    data = process_pdf(pdf_bytes,source_name, applicationId,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Dict] = list(data["text_chunks"])  
    stem = os.path.splitext(source_name)[0]
    # tables -> text chunks
    
    for ch in chunks:
        ch["applicationId"] = applicationId
        ch["source"] = source_name

    for i, row in enumerate(data.get("tables", [])):
        row_text = " \n".join([f"{k}: {v}" for k, v in row.items() if v is not None])
        #chunks.append({"id": f"{stem}_table_row_{i+1}", "content": row_text})
        
        chunks.append({
                    "id": f"{stem}_table_row_{i+1}",
                    "content": row_text,
                    "applicationId": applicationId,
                    "source": source_name,
                    "type": "table"
                })

    # embeddings
    start_time = time.time()
    total = len(chunks)
    for idx, ch in enumerate(chunks, start=1):
        ch["contentVector"] = aoai_embedder(ch["content"], aoai_client)
        if idx % max(1, total // 10) == 0 or idx == total:
            print(f"  -> {stem}: {idx}/{total} chunks embedded")
    print(f"  Finished '{source_name}' in {time.time() - start_time:.1f}s")

    results[source_name] = chunks

    return results

# -----------------------------
# Azure Cognitive Search upload
# -----------------------------

def batch(iterable: Iterable, n: int) -> Iterable[List]:
    bucket: List = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) == n:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def to_search_docs(all_chunks: List[Dict]) -> List[Dict]:
    docs = []
    for i, ch in enumerate(all_chunks):
        docs.append({
            "id": ch.get("id", f"chunk_{i}"),
            "content": ch["content"],
            "contentVector": ch["contentVector"],
            "applicationId": ch["applicationId"],  # ✅ REQUIRED
        })
    return docs


def upload_embeddings(all_docs: List[Dict], batch_size: int = 1000) -> None:
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id= SP_CLIENT_ID,
        client_secret=SP_CLIENT_SECRET,
    )
    client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_pdf,
        credential=credential,
    )
    uploaded = 0
    for b in batch(all_docs, batch_size):
        result = client.upload_documents(documents=b)
        uploaded += len(b)
        print(f"  Uploaded {uploaded} documents (last batch status: {result})")
    print("Upload complete.")



from typing import List, Dict
 
def embedding_Chunks(
    pdf_bytes,
    source_name,
    applicationId: int,
    chunk_size: int = 500,
    chunk_overlap: int = 200,
    batch_size: int = 1000,
    no_upload: bool = False
):
    
    results = embed_pdfs(
        pdf_bytes,
        source_name,
        applicationId,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
 
    if not no_upload:
        combined: List[Dict] = []
        for chunks in results.values():
            combined.extend(chunks)

        search_docs = to_search_docs(combined)
        upload_embeddings(search_docs, batch_size=batch_size)
 
    total_chunks = sum(len(chs) for chs in results.values())
    print(f"Done. Embedded {total_chunks} chunks across {len(results)} file(s).")
 
    return total_chunks
 