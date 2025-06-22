import os, sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.ingestion import Ingest, Document
from app.settings import SETTINGS


@pytest.fixture
def ingest():
    # Patch Langfuse, Chroma, and embeddings to avoid real external calls
    with patch("app.ingestion.Langfuse"), \
         patch("app.ingestion.Chroma") as chroma_mock, \
         patch("app.ingestion.OllamaEmbeddings"), \
         patch("app.ingestion.OpenAIEmbeddings"):
        chroma_instance = chroma_mock.return_value
        chroma_instance.aadd_documents = AsyncMock()
        chroma_instance.asimilarity_search = AsyncMock(return_value=[
            Document("Test content", metadata={"page_label": 1})
        ])
        yield Ingest()

@pytest.mark.asyncio
async def test_pdf_loader_pages_yields_documents(ingest, tmp_path):
    # Create a dummy PDF file (use a real PDF for integration, here we mock)
    with patch("app.ingestion.PyPDFLoader") as loader_mock:
        page = Document("Dummy PDF file", metadata={"page_label": 1})
        loader_instance = loader_mock.return_value
        loader_instance.alazy_load.return_value = AsyncMock()
        loader_instance.alazy_load.return_value.__aiter__.return_value = [page]
        docs = []
        async for doc in ingest.pdf_loader_pages(pdf_path="dummy.pdf"):
            docs.append(doc)
        assert len(docs) == 1
        assert docs[0].page_content == "Dummy PDF file"
        assert docs[0].metadata["page_label"] == 1

@pytest.mark.asyncio
async def test_chunking_document_splits_long_text(ingest):
    long_text = "a" * 56400
    doc = Document(long_text, metadata={"page_label": 1})
    chunks = await ingest.chunking_document(doc)
    assert len(chunks) > 1
    assert sum(len(c.page_content) for c in chunks) == len(long_text) + SETTINGS.text_splitter.chunk_overlap * (len(chunks) - 1)

@pytest.mark.asyncio
async def test_similarity_search_returns_results(ingest):
    results = await ingest.similarity_search("test query", k=1)
    assert isinstance(results, list)
    assert "text" in results[0]
    assert "page" in results[0]

@pytest.mark.asyncio
async def test_direct_ingest_pdf_calls_chroma(ingest):
    with patch.object(ingest, "pdf_loader_pages") as loader_mock:
        loader_mock.return_value = AsyncMock()
        loader_mock.return_value.__aiter__.return_value = [
            Document("Test", metadata={"page_label": 1})
        ]
        await ingest.direct_ingest_pdf(pdf_path="dummy.pdf")
        ingest.chroma.aadd_documents.assert_awaited()

@pytest.mark.asyncio
async def test_chunking_ingest_pdf_calls_chroma(ingest):
    with patch.object(ingest, "pdf_loader_overlapping_pages") as loader_mock:
        loader_mock.return_value = AsyncMock()
        loader_mock.return_value.__aiter__.return_value = [
            (Document("Test", metadata={"page_label": 1}), None)
        ]
        await ingest.chunking_ingest_pdf(pdf_path="dummy.pdf")
        ingest.chroma.aadd_documents.assert_awaited()