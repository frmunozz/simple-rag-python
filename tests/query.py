import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.query import Query

@pytest.fixture
def mock_ingest():
    ingest = MagicMock()
    ingest.similarity_search = AsyncMock(return_value=[
        {"text": "Python is a programming language.", "page": "5"}
    ])
    return ingest

@pytest.mark.asyncio
async def test_query_returns_answer_and_sources(mock_ingest):
    # Patch ChatOllama and ChatOpenAI to avoid real LLM calls
    with patch("app.query.ChatOllama") as ollama_mock, \
         patch("app.query.ChatOpenAI") as openai_mock, \
         patch("app.query.Langfuse"), \
         patch("app.query.CallbackHandler"):
        chat_instance = ollama_mock.return_value
        chat_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Python is a programming language."))
        query = Query(mock_ingest)
        result = await query.query("What is Python?", k=1)
        assert "answer" in result
        assert "sources" in result
        assert result["sources"][0]["page"] == "5"

@pytest.mark.asyncio
async def test_query_with_openai_provider(mock_ingest):
    with patch("app.query.ChatOpenAI") as openai_mock, \
         patch("app.query.Langfuse"), \
         patch("app.query.CallbackHandler"):
        chat_instance = openai_mock.return_value
        chat_instance.ainvoke = AsyncMock(return_value=MagicMock(content="OpenAI answer"))
        # Patch settings to use openai provider
        with patch("app.query.SETTINGS") as settings_mock:
            settings_mock.provider = "openai"
            settings_mock.openai.chat_model = "gpt-4"
            settings_mock.openai.temperature = 0.5
            settings_mock.openai.api_key = "sk-test"
            query = Query(mock_ingest)
            result = await query.query("Test?", k=1)
            assert result["answer"] == "OpenAI answer"

@pytest.mark.asyncio
async def test_query_with_ollama_provider(mock_ingest):
    with patch("app.query.ChatOllama") as ollama_mock, \
         patch("app.query.Langfuse"), \
         patch("app.query.CallbackHandler"):
        chat_instance = ollama_mock.return_value
        chat_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Ollama answer"))
        with patch("app.query.SETTINGS") as settings_mock:
            settings_mock.provider = "ollama"
            settings_mock.ollama.chat_model = "llama3:8b"
            settings_mock.ollama.temperature = 0.5
            settings_mock.ollama.host = "http://localhost:11535"
            query = Query(mock_ingest)
            result = await query.query("Test?", k=1)
            assert result["answer"] == "Ollama answer"