from fastapi.testclient import TestClient

from main import app
from util.exceptions import IntegrityError


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_endpoint_returns_html() -> None:
    response = client.get("/home/")

    assert response.status_code == 200
    assert "FastAPI sample app is running." in response.text
    assert "text/html" in response.headers["content-type"]


def test_openapi_contains_custom_metadata() -> None:
    schema = app.openapi()

    assert schema["info"]["contact"]["name"] == "ICG Sample Team"
    assert schema["servers"][0]["url"]
    assert {tag["name"] for tag in schema["tags"]} == {"home", "health", "uploadfile"}


def test_upload_endpoint_uses_knowledge_service(monkeypatch) -> None:
    def fake_ingest_document(filename: str, content: bytes) -> dict[str, object]:
        assert filename == "notes.txt"
        assert content == b"hello world"
        return {
            "message": "Document ingested successfully.",
            "filename": filename,
            "chunks_indexed": 1,
            "size_bytes": len(content),
        }

    monkeypatch.setattr(
        "routers.uploadfile.knowledge_index_service.ingest_document",
        fake_ingest_document,
    )

    response = client.post(
        "/files/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "notes.txt"


def test_query_endpoint_uses_knowledge_service(monkeypatch) -> None:
    def fake_query_index(query: str, top_k: int) -> dict[str, object]:
        assert query == "what is in the file?"
        assert top_k == 3
        return {"query": query, "results": [{"chunk_id": 1, "score": 0.9}]}

    monkeypatch.setattr(
        "routers.uploadfile.knowledge_index_service.query_index",
        fake_query_index,
    )

    response = client.post(
        "/files/query",
        json={"query": "what is in the file?", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["chunk_id"] == 1


def test_ask_endpoint_uses_knowledge_service(monkeypatch) -> None:
    def fake_ask_index(query: str, top_k: int) -> dict[str, object]:
        assert query == "summarize it"
        assert top_k == 2
        return {
            "query": query,
            "answer": "Summary from stub.",
            "results": [],
            "model": "gpt-5.4-mini",
        }

    monkeypatch.setattr(
        "routers.uploadfile.knowledge_index_service.ask_index",
        fake_ask_index,
    )

    response = client.post(
        "/files/ask",
        json={"query": "summarize it", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Summary from stub."


def test_delete_file_endpoint_uses_knowledge_service(monkeypatch) -> None:
    def fake_delete_document(filename: str) -> dict[str, str]:
        assert filename == "notes.txt"
        return {"message": "Document deleted successfully.", "filename": filename}

    monkeypatch.setattr(
        "routers.uploadfile.knowledge_index_service.delete_document",
        fake_delete_document,
    )

    response = client.delete("/files/notes.txt")

    assert response.status_code == 200
    assert response.json()["filename"] == "notes.txt"


def test_delete_multiple_files_endpoint_uses_knowledge_service(monkeypatch) -> None:
    def fake_delete_documents(filenames: list[str]) -> dict[str, list[str]]:
        assert filenames == ["a.txt", "b.txt"]
        return {"deleted_files": filenames, "missing_files": []}

    monkeypatch.setattr(
        "routers.uploadfile.knowledge_index_service.delete_documents",
        fake_delete_documents,
    )

    response = client.request(
        "DELETE",
        "/files/",
        json={"filenames": ["a.txt", "b.txt"]},
    )

    assert response.status_code == 200
    assert response.json()["deleted_files"] == ["a.txt", "b.txt"]


def test_integrity_error_handler_returns_expected_payload() -> None:
    @app.get("/test-integrity-error")
    async def raise_integrity_error() -> None:
        raise IntegrityError("duplicate record")

    response = client.get("/test-integrity-error")

    assert response.status_code == 409
    assert response.json() == {
        "error": "IntegrityError",
        "message": "duplicate record",
        "path": "/test-integrity-error",
    }
