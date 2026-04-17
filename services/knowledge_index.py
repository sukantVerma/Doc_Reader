"""Knowledge-index service used by the file router.

This module keeps the "real" document logic out of the router.

High-level flow:
1. Save the uploaded file to `uploads/`
2. Parse the file bytes into plain text
3. Split the text into smaller chunks
4. Create one OpenAI embedding vector per chunk
5. Store vectors in a FAISS index
6. Store chunk metadata in JSON so we can map `chunk_id -> text`
7. Support retrieval and answer generation against the stored index
"""

import html
import json
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any

import faiss
import numpy as np
from fastapi import HTTPException
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class KnowledgeIndexService:
    """Owns document ingestion, indexing, querying, and deletion.

    The router should stay thin and call this service.
    This service is the single place where knowledge-index behavior lives.
    """

    def __init__(
        self,
        upload_dir: str = settings.upload_dir,
        storage_dir: str = settings.knowledge_storage_dir,
        embedding_dim: int = settings.embedding_dim,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.upload_dir = Path(upload_dir)
        self.storage_dir = Path(storage_dir)
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_path = self.storage_dir / "metadata.json"
        self.index_path = self.storage_dir / "faiss.index"
        self.lock = Lock()
        self.embedding_model = settings.openai_embedding_model
        self.generation_model = settings.openai_generation_model
        self.openai_api_key = settings.openai_api_key
        self.client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = self._load_metadata()
        self.index = self._load_index()

    def ingest_document(self, filename: str, content: bytes) -> dict[str, Any]:
        """Save, parse, embed with OpenAI, and index one uploaded document.

        If a document with the same filename already exists, we treat this as
        an update and replace its old chunks with newly generated ones.
        """
        self._require_openai()
        document_id = self._sanitize_filename(filename)
        logger.info("Upload started | filename=%s | bytes=%s", document_id, len(content))

        file_path = self.upload_dir / document_id
        file_path.write_bytes(content)
        logger.info("Upload saved to disk | filename=%s | path=%s", document_id, file_path)

        logger.info("Document parsing started | filename=%s", document_id)
        text = self._parse_document(file_path, content)
        logger.info("Document parsing completed | filename=%s | characters=%s", document_id, len(text))

        logger.info("Chunking started | filename=%s", document_id)
        chunks = self._chunk_text(text)
        if not chunks:
            logger.warning("Chunking failed | filename=%s | reason=no indexable text", document_id)
            raise HTTPException(
                status_code=400,
                detail="Document does not contain enough text to index.",
            )
        logger.info("Chunking completed | filename=%s | chunks=%s", document_id, len(chunks))

        with self.lock:
            self._ensure_index_ready_locked()
            if document_id in self.metadata["documents"]:
                logger.info("Existing document found, replacing old index entries | filename=%s", document_id)
                self._remove_document_locked(document_id, remove_file=False)

            logger.info(
                "Creating OpenAI vector spaces pending | filename=%s | chunks=%s | model=%s",
                document_id,
                len(chunks),
                self.embedding_model,
            )
            embeddings = self._embed_texts([chunk["content"] for chunk in chunks])
            chunk_ids: list[int] = []
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = int(self.metadata["next_chunk_id"])
                self.metadata["next_chunk_id"] += 1
                self.metadata["chunks"][str(chunk_id)] = {
                    "document_id": document_id,
                    "content": chunk["content"],
                    "chunk_index": chunk_index,
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                }
                chunk_ids.append(chunk_id)
            logger.info(
                "Creating OpenAI vector spaces completed | filename=%s | indexed_chunks=%s",
                document_id,
                len(chunk_ids),
            )

            self.metadata["documents"][document_id] = {
                "filename": document_id,
                "source_path": str(file_path),
                "chunk_ids": chunk_ids,
                "size_bytes": len(content),
                "num_chunks": len(chunk_ids),
                "content_preview": text[:200],
            }

            logger.info("Adding vectors to FAISS pending | filename=%s", document_id)
            ids = np.array(chunk_ids, dtype=np.int64)
            self.index.add_with_ids(embeddings, ids)
            logger.info("Adding vectors to FAISS completed | filename=%s | total_vectors=%s", document_id, self.index.ntotal)

            logger.info("Persisting metadata and FAISS index pending | filename=%s", document_id)
            self._persist_locked()
            logger.info("Persisting metadata and FAISS index completed | filename=%s", document_id)

        logger.info(
            "Upload completed | filename=%s | chunks_indexed=%s | total_vectors=%s",
            document_id,
            len(chunks),
            self.index.ntotal,
        )

        return {
            "message": "Document ingested successfully.",
            "filename": document_id,
            "chunks_indexed": len(chunks),
            "size_bytes": len(content),
        }

    def query_index(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the FAISS index using an OpenAI embedding of the query."""
        self._require_openai()
        normalized_query = query.strip()
        if not normalized_query:
            raise HTTPException(status_code=400, detail="Query text is required.")

        with self.lock:
            self._ensure_index_ready_locked()
            if self.index.ntotal == 0:
                return {"query": normalized_query, "results": []}

            k = min(max(top_k, 1), self.index.ntotal)
            logger.info("Query embedding pending | query=%s | model=%s", normalized_query, self.embedding_model)
            query_vector = self._embed_texts([normalized_query])
            logger.info("Query embedding completed | query=%s", normalized_query)
            scores, ids = self.index.search(query_vector, k)

            results: list[dict[str, Any]] = []
            for score, chunk_id in zip(scores[0], ids[0]):
                if chunk_id == -1:
                    continue
                chunk = self.metadata["chunks"].get(str(int(chunk_id)))
                if not chunk:
                    continue
                results.append(
                    {
                        "chunk_id": int(chunk_id),
                        "document_id": chunk["document_id"],
                        "content": chunk["content"],
                        "score": float(score),
                        "chunk_index": chunk["chunk_index"],
                    }
                )

        return {"query": normalized_query, "results": results}

    def ask_index(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve relevant chunks, then ask an OpenAI model to answer from them."""
        self._require_openai()
        retrieval = self.query_index(query, top_k)
        results = retrieval["results"]
        if not results:
            return {
                "query": query,
                "answer": "I could not find relevant content in the indexed documents.",
                "results": [],
                "model": self.generation_model,
            }

        context_blocks = []
        for item in results:
            context_blocks.append(
                f"Document: {item['document_id']}\nChunk: {item['chunk_index']}\nContent: {item['content']}"
            )

        logger.info("LLM answer generation pending | model=%s | retrieved_chunks=%s", self.generation_model, len(results))
        response = self.client.responses.create(
            model=self.generation_model,
            instructions=(
                "Answer the user's question using only the provided document context. "
                "If the answer is not supported by the context, say that clearly."
            ),
            input=(
                "Question:\n"
                f"{query}\n\n"
                "Document context:\n"
                + "\n\n".join(context_blocks)
            ),
        )
        logger.info("LLM answer generation completed | model=%s", self.generation_model)

        return {
            "query": query,
            "answer": response.output_text,
            "results": results,
            "model": self.generation_model,
        }

    def delete_document(self, filename: str) -> dict[str, str]:
        """Delete one document from both disk storage and the knowledge index."""
        document_id = self._sanitize_filename(filename)
        with self.lock:
            self._ensure_index_ready_locked()
            self._remove_document_locked(document_id, remove_file=True)
            self._rebuild_index_locked()
            self._persist_locked()

        return {"message": "Document deleted successfully.", "filename": document_id}

    def delete_documents(self, filenames: list[str]) -> dict[str, list[str]]:
        """Delete multiple documents and rebuild the FAISS index once."""
        deleted_files: list[str] = []
        missing_files: list[str] = []

        with self.lock:
            self._ensure_index_ready_locked()
            for filename in filenames:
                document_id = self._sanitize_filename(filename)
                if document_id not in self.metadata["documents"]:
                    missing_files.append(document_id)
                    continue
                self._remove_document_locked(document_id, remove_file=True)
                deleted_files.append(document_id)

            self._rebuild_index_locked()
            self._persist_locked()

        return {"deleted_files": deleted_files, "missing_files": missing_files}

    def _remove_document_locked(self, document_id: str, remove_file: bool) -> None:
        """Remove a document's metadata and chunk mapping.

        This helper expects the caller to already hold the service lock.
        """
        document = self.metadata["documents"].get(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")

        for chunk_id in document.get("chunk_ids", []):
            self.metadata["chunks"].pop(str(chunk_id), None)

        self.metadata["documents"].pop(document_id, None)

        if remove_file:
            file_path = self.upload_dir / document_id
            if file_path.exists():
                file_path.unlink()

    def _rebuild_index_locked(self) -> None:
        """Recreate the FAISS index from the current chunk metadata.

        This is simple and reliable for small projects.
        For large datasets, you would usually update the index incrementally.
        """
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))

        chunk_items = sorted(
            (int(chunk_id), chunk_data)
            for chunk_id, chunk_data in self.metadata["chunks"].items()
        )
        if not chunk_items:
            return

        ids = np.array([chunk_id for chunk_id, _ in chunk_items], dtype=np.int64)
        self._require_openai()
        embeddings = self._embed_texts([chunk_data["content"] for _, chunk_data in chunk_items])
        self.index.add_with_ids(embeddings, ids)

    def _persist_locked(self) -> None:
        """Persist both the metadata JSON and the FAISS index to disk."""
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2),
            encoding="utf-8",
        )
        faiss.write_index(self.index, str(self.index_path))

    def _load_metadata(self) -> dict[str, Any]:
        """Load the JSON metadata that tracks documents and chunk contents."""
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            metadata.setdefault("documents", {})
            metadata.setdefault("chunks", {})
            metadata.setdefault("next_chunk_id", 1)
            return metadata
        return {"documents": {}, "chunks": {}, "next_chunk_id": 1}

    def _load_index(self) -> faiss.Index:
        """Load the persisted FAISS index if it exists, else create a new one."""
        if self.index_path.exists():
            loaded_index = faiss.read_index(str(self.index_path))
            if loaded_index.d == self.embedding_dim:
                return loaded_index
            logger.warning(
                "Stored FAISS index dimension mismatch | stored=%s | expected=%s | rebuilding required",
                loaded_index.d,
                self.embedding_dim,
            )
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))

    def _ensure_index_ready_locked(self) -> None:
        """Keep the in-memory FAISS index aligned with stored metadata."""
        expected_vectors = len(self.metadata["chunks"])
        current_vectors = self.index.ntotal
        if self.index.d != self.embedding_dim or current_vectors != expected_vectors:
            logger.info(
                "FAISS index consistency check triggered rebuild | expected_vectors=%s | current_vectors=%s",
                expected_vectors,
                current_vectors,
            )
            self._rebuild_index_locked()
            self._persist_locked()

    def _sanitize_filename(self, filename: str) -> str:
        """Keep only a safe basename so users cannot write outside `uploads/`."""
        sanitized_name = Path(filename).name.strip()
        if not sanitized_name:
            raise HTTPException(status_code=400, detail="Filename is required.")
        return sanitized_name

    def _parse_document(self, file_path: Path, content: bytes) -> str:
        """Convert uploaded bytes into plain text for chunking/indexing.

        Current support is intentionally simple:
        - plain text-like files: use decoded text directly
        - JSON: pretty-print it into readable text
        - HTML: strip tags/scripts/styles before indexing
        """
        suffix = file_path.suffix.lower()
        text = self._decode_bytes(content)

        if suffix == ".json":
            try:
                parsed = json.loads(text)
                return json.dumps(parsed, indent=2, ensure_ascii=True)
            except json.JSONDecodeError:
                return text

        if suffix in {".html", ".htm"}:
            without_scripts = re.sub(
                r"<(script|style)[^>]*>.*?</\1>",
                " ",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            stripped = re.sub(r"<[^>]+>", " ", without_scripts)
            return html.unescape(stripped)

        return text

    def _decode_bytes(self, content: bytes) -> str:
        """Try a few common text encodings before rejecting the file."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="Unsupported file encoding.")

    def _chunk_text(self, text: str) -> list[dict[str, Any]]:
        """Split long text into overlapping chunks for retrieval.

        Overlap helps preserve context between neighboring chunks so a query
        does not miss information that falls near a chunk boundary.
        """
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        chunks: list[dict[str, Any]] = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            if end < text_length:
                split_at = normalized.rfind(" ", start, end)
                if split_at > start + (self.chunk_size // 2):
                    end = split_at

            chunk_content = normalized[start:end].strip()
            if chunk_content:
                chunks.append(
                    {
                        "content": chunk_content,
                        "start_char": start,
                        "end_char": end,
                    }
                )

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Create OpenAI embeddings for one or more texts."""
        self._require_openai()
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float",
        )
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return vectors

    def _require_openai(self) -> None:
        """Ensure the service can call the OpenAI API."""
        if self.client is None:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY is not configured. Set it before using document ingestion or query.",
            )


knowledge_index_service = KnowledgeIndexService()
