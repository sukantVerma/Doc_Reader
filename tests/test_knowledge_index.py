from pathlib import Path

import pytest
from fastapi import HTTPException

from services.knowledge_index import KnowledgeIndexService


@pytest.fixture
def service(tmp_path: Path) -> KnowledgeIndexService:
    return KnowledgeIndexService(
        upload_dir=str(tmp_path / "uploads"),
        storage_dir=str(tmp_path / "storage"),
        embedding_dim=8,
        chunk_size=20,
        chunk_overlap=5,
    )


def test_sanitize_filename_keeps_basename(service: KnowledgeIndexService) -> None:
    assert service._sanitize_filename("../unsafe/report.txt") == "report.txt"


def test_sanitize_filename_rejects_empty_values(service: KnowledgeIndexService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        service._sanitize_filename("   ")

    assert exc_info.value.status_code == 400


def test_parse_document_formats_json_for_indexing(service: KnowledgeIndexService) -> None:
    result = service._parse_document(
        Path("payload.json"),
        b'{"project":"icg","enabled":true}',
    )

    assert '"project": "icg"' in result
    assert '"enabled": true' in result


def test_parse_document_strips_html_tags(service: KnowledgeIndexService) -> None:
    html_bytes = b"<html><body><h1>Hello</h1><script>ignored()</script><p>World</p></body></html>"

    result = service._parse_document(Path("page.html"), html_bytes)

    assert "Hello" in result
    assert "World" in result
    assert "ignored()" not in result


def test_chunk_text_creates_overlapping_chunks(service: KnowledgeIndexService) -> None:
    chunks = service._chunk_text(
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    )

    assert len(chunks) >= 2
    assert all(chunk["content"] for chunk in chunks)
    assert chunks[0]["start_char"] == 0
    assert chunks[1]["start_char"] < chunks[0]["end_char"]


def test_decode_bytes_falls_back_to_latin_1(service: KnowledgeIndexService) -> None:
    assert service._decode_bytes(b"\xff\xfe\xfd\xfc") == "\xff\xfe\xfd\xfc"
