from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langfuse import Langfuse
import logging
from .settings import SETTINGS
from .ingestion import Ingest
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler


class Query:
    def __init__(self, ingest: Ingest):
        self.logger = logging.getLogger("app.query")
        self.ingest = ingest
        self.chat = self._get_chat()

    def _get_chat(self) -> BaseChatModel:
        langfuse = Langfuse(
            public_key=SETTINGS.langfuse.public_key,
            secret_key=SETTINGS.langfuse.secret_key,
        )
        langfuse_callback = CallbackHandler()
        if SETTINGS.llm.provider == "openai":
            self.logger.info("Initializing OpenAI LLM")
            return ChatOpenAI(
                model=SETTINGS.llm.model,
                temperature=SETTINGS.llm.temperature,
                streaming=True,
                api_key=SETTINGS.openai_api_key,  # type: ignore
                callbacks=[langfuse_callback],
            )
        elif SETTINGS.llm.provider == "ollama":
            self.logger.info("Initializing Ollama LLM")
            return ChatOllama(
                model=SETTINGS.llm.model,
                temperature=SETTINGS.llm.temperature,
                base_url=SETTINGS.ollama.host,
                callbacks=[langfuse_callback],
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {SETTINGS.llm.provider}")

    async def _query_with_sources(self, query: str, k=5) -> dict:
        search_results = await self.ingest.similarity_search(query, k=5)

        sources_formatted = "\n".join(
            f"- [Page {match['page']}]: {match['text']}..." for match in search_results
        )

        instructions = SystemMessage(
            content=f"""
        You MUST answer using ONLY these sources:
        {sources_formatted}
        
        Rules:
        1. ALWAYS cite the exact page number like [Page X]
        1. If unsure, say "I couldn't find definitive information"
        2. Never invent facts outside the sources
        """
        )
        response = await self.chat.ainvoke([instructions, HumanMessage(content=query)])

        return {"answer": response.content, "sources": search_results}

    async def query(self, query: str, k: int = 5) -> dict:
        return await self._query_with_sources(query, k=k)
