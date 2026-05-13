import os
from azure.identity import ClientSecretCredential
from azure.search.documents import SearchClient

TENANT_ID = os.environ["TENANT_ID"]
SP_CLIENT_ID = os.environ["SP_CLIENT_ID"]
SP_CLIENT_SECRET = os.environ["SP_CLIENT_SECRET"]
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
AZURE_SEARCH_INDEX_pdf = os.environ["AZURE_SEARCH_INDEX_pdf"]

KEY_FIELD = "id"  # 🔴 change this if your key is different


def delete_pdf_documents_by_application_id(applicationId: int):
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=SP_CLIENT_ID,
        client_secret=SP_CLIENT_SECRET
    )

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_pdf,
        credential=credential
    )

    try:
        print(f"🔍 Searching documents for applicationId={applicationId}")

        results = search_client.search(
            search_text="*",
            filter=f"applicationId eq {applicationId}",
            select=[KEY_FIELD],
            top=1000
        )

        docs = list(results)  # ✅ force evaluation

        if not docs:
            print(f"⚠️ No documents found for applicationId={applicationId}")
            return

        print(f"🗑️ Found {len(docs)} documents to delete")

        # ✅ Delete in batches of 1000
        for i in range(0, len(docs), 1000):
            batch = docs[i:i + 1000]
            delete_payload = [{KEY_FIELD: doc[KEY_FIELD]} for doc in batch]

            search_client.delete_documents(documents=delete_payload)
            print(f"✅ Deleted batch {i // 1000 + 1}")

        print(f"🎉 Completed deletion for applicationId={applicationId}")

    except Exception as e:
        print(f"❌ Delete failed: {e}")
        raise

     