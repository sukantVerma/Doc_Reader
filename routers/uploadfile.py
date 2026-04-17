"""Router layer for file upload and knowledge-index operations.

The router is intentionally thin:
- read request data
- validate simple request models
- delegate the actual work to `knowledge_index_service`
"""

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from services.knowledge_index import knowledge_index_service

router = APIRouter(prefix="/files", tags=["uploadfile"])


class DeleteFilesRequest(BaseModel):
    """Request body for deleting multiple indexed documents."""

    filenames: list[str]


class QueryRequest(BaseModel):
    """Request body for similarity search against the knowledge index."""

    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    """Request body for retrieval-augmented answer generation."""

    query: str
    top_k: int = 5


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload one file and ingest it into the knowledge index."""
    content = await file.read()
    return knowledge_index_service.ingest_document(file.filename or "", content)


@router.delete("/{filename}")
async def delete_file(filename: str) -> dict[str, str]:
    """Delete one indexed document by filename."""
    return knowledge_index_service.delete_document(filename)


@router.delete("/")
async def delete_multiple_files(payload: DeleteFilesRequest) -> dict[str, list[str]]:
    """Delete many indexed documents in one request."""
    return knowledge_index_service.delete_documents(payload.filenames)


@router.post("/query")
async def query_documents(payload: QueryRequest) -> dict:
    """Search the knowledge index and return the closest chunks."""
    return knowledge_index_service.query_index(payload.query, payload.top_k)


@router.post("/ask")
async def ask_documents(payload: AskRequest) -> dict:
    """Retrieve relevant chunks and generate an answer with OpenAI."""
    return knowledge_index_service.ask_index(payload.query, payload.top_k)
