import os
from azure.identity import ClientSecretCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
        SimpleField, SearchFieldDataType, SearchableField, SearchField,
        VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile, SearchIndex
    )

TENANT_ID = os.environ.get("TENANT_ID")
SP_CLIENT_ID = os.environ.get("SP_CLIENT_ID")
SP_CLIENT_SECRET = os.environ.get("SP_CLIENT_SECRET")
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_pdf = os.environ.get("AZURE_SEARCH_INDEX_pdf")


def setup_pdf_index(applicationId):
    """
    Creates or updates an Azure AI Search index for PDFs with:
      - id (key)
      - content (searchable text)
      - contentVector (vector field, 3072 dims, HNSW profile)

    Reads required settings from environment variables:
      TENANT_ID
      SP_CLIENT_ID
      SP_CLIENT_SECRET
      AZURE_SEARCH_SERVICE_ENDPOINT   (e.g., https://<service>.search.windows.net)
      AZURE_SEARCH_INDEX_pdf          (index name)

    Prints basic info and returns a small dict with stats.
    """
    print("applicationId",applicationId)
    # ---- Auth & client ----
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=SP_CLIENT_ID,
        client_secret=SP_CLIENT_SECRET
    )
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=credential
    )

    # ---- Define fields (schema) ----
    fields = [
        SimpleField(
            name="id", type=SearchFieldDataType.String,
            key=True, sortable=True, filterable=True, facetable=True
        ),
        
        SimpleField(
                name="applicationId",
                type=SearchFieldDataType.Int32,
                filterable=True,
                facetable=True
            ),

        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,         # must match your embedding model
            vector_search_profile_name="myHnswProfile"
        ),
    ]

    # ---- Vector search (HNSW) ----
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw"
        )]
    )

    # ---- Build and create/update index ----
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_pdf,
        fields=fields,
        vector_search=vector_search
        # (optional) semantic_search can be added later
    )

    result = index_client.create_or_update_index(index)

    # ---- Stats ----
    stats = index_client.get_index_statistics(AZURE_SEARCH_INDEX_pdf)

    print(f"Index name  : {AZURE_SEARCH_INDEX_pdf}")
    print(f"Result name : {result.name} (created/updated)")
    print(f"Doc count   : {getattr(stats, 'document_count', None)}")
    print(f"Storage (B) : {getattr(stats, 'storage_size', None)}")

    return {
        "index_name": AZURE_SEARCH_INDEX_pdf,
        "result_name": result.name,
        "stats": {
            "document_count": getattr(stats, "document_count", None),
            "storage_size": getattr(stats, "storage_size", None)
        }
    }
