from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File

# --- Use only builders; remove singleton agent imports ---

from src.rag.embedding.embedding_pdf import embedding_Chunks

from src.rag.index.index_pdf import setup_pdf_index
from fastapi import APIRouter
from src.rag.index.delete_index import delete_pdf_documents_by_application_id

# app = FastAPI(title="SelectorGroupChat - Dynamic Orchestrator Delegation (PO/QA/TM) + Critic Review")

rag_router=APIRouter()
# ======================================================================================
# FastAPI app scaffolding
# ======================================================================================

@rag_router.post("/upload/files")
async def upload_files(applicationId: int, 
                       files: List[UploadFile] = File(...),
                       ):
    try:
        uploaded_files = []
        for file in files:
            pdf_bytes = await file.read()
            source_name = file.filename
            uploaded_files.append({
                "filename": source_name,
                "content_type": file.content_type
            })
            index=setup_pdf_index(applicationId)
            print("embeddingfile", file.filename)
            embedding_Chunk=embedding_Chunks(pdf_bytes,source_name,applicationId,chunk_size=500,
            chunk_overlap=200,
            batch_size=1000,
            no_upload=False)
 
        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.delete("/delete/files")
async def delete_pdf_documents(applicationId: int):
    """
    Deletes the Azure AI Search PDF index for a given applicationId.
    ⚠️ Admin / destructive operation.
    """
    try:
        delete_pdf_documents_by_application_id(applicationId)
        return {
            "status": "success",
            "message": f"Azure AI Search PDF index deleted successfully for applicationId={applicationId}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
