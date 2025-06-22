from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings  # if doing local embedding with ollama
from langchain_openai import OpenAIEmbeddings  # if using openai for embeddings
from typing import List
from .settings import SETTINGS
import os
from langfuse import Langfuse
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from hashlib import md5
import json
import logging



class Ingest:
    def __init__(self):
        self.langfuse = Langfuse(
            public_key=SETTINGS.langfuse.public_key,
            secret_key=SETTINGS.langfuse.secret_key,
            host=SETTINGS.langfuse.host,
        )
        self.logger = logging.getLogger("app.ingest")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SETTINGS.text_splitter.chunk_size,
            chunk_overlap=SETTINGS.text_splitter.chunk_overlap,
        )
        self.embeddings = OllamaEmbeddings(
            model=SETTINGS.ollama.embedding_model, base_url=SETTINGS.ollama.host
        )
        # self.embeddings = OpenAIEmbeddings(
        #     api_key=SETTINGS.openai_api_key # type: ignore
        # )

        self.chroma = Chroma(
            embedding_function=self.embeddings,
            persist_directory=os.path.join(
                SETTINGS.package_root_directory, "chroma_data"
            ),
        )

    def _get_document_id(self, document: Document):
        content_hash = md5(document.page_content.encode("utf-8")).hexdigest()
        meta_hash = md5(
            json.dumps(document.metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{content_hash}-{meta_hash}"

    async def pdf_loader_pages(self, pdf_path: str|None = None, cleanup: bool = False):
        self.logger.debug(f"input pdf_path: {pdf_path}", extra={"pdf_path": pdf_path})
        pdf_path = pdf_path if pdf_path is not None else SETTINGS.pdf_path
        self.logger.info(f"loading pdf: {pdf_path}", extra={"pdf_path": pdf_path})
        with self.langfuse.start_as_current_span(name="pdf loading") as span:
            loader = PyPDFLoader(pdf_path)
            async for page in loader.alazy_load():
                yield page
        
        if cleanup and pdf_path != SETTINGS.pdf_path:
            try:
                os.remove(pdf_path)
            except Exception as e:
                # catch it and log the error
                self.logger.error(f"Failed to remove file: {e}", extra={"pdf_path": pdf_path})

    async def pdf_loader_overlapping_pages(self, pdf_path: str|None = None, cleanup: bool = False):
        prev_page: Document | None = None
        async for page in self.pdf_loader_pages(pdf_path=pdf_path, cleanup=cleanup):
            _prev_page_label: str | None = None
            if prev_page is not None:
                _page_content = (
                    prev_page.page_content[-SETTINGS.text_splitter.chunk_overlap :]
                    + page.page_content
                )
                _prev_page_label = str(prev_page.metadata["page_label"])
            else:
                _page_content = page.page_content
            yield Document(_page_content, metadata=page.metadata), _prev_page_label
            prev_page = page

    async def chunking_document(
        self, document: Document, prev_page_label: str | None = None
    ):
        # if page content has more than 'chunk_size' characters, do split
        if len(document.page_content) > SETTINGS.text_splitter.chunk_size:
            page_content_chunks = self.text_splitter.split_text(document.page_content)
        else:
            page_content_chunks = [document.page_content]
        chunks = []
        for i, content_chunk in enumerate(page_content_chunks):
            _metadata = document.metadata.copy()
            _metadata["chunk"] = i + 1
            _metadata["total_chunks"] = len(page_content_chunks)
            if i == 0 and prev_page_label is not None:
                # add this metadata only to the first chunk, and only if prev_page existed
                # so we have overlapped text
                _metadata["overlapped_prev_page"] = True
                _metadata["prev_page_label"] = prev_page_label
            chunks.append(Document(content_chunk, metadata=_metadata))
        return chunks

    async def direct_ingest_pdf(self, pdf_path: str|None = None, cleanup: bool = False):
        with self.langfuse.start_as_current_span(
            name="ingest", input="direct_ingest_pdf"
        ) as span:
            try:
                documents = [page async for page in self.pdf_loader_pages(pdf_path=pdf_path, cleanup=cleanup)]
                # use ids to prevent repeating embeddings for same pages
                document_ids = [self._get_document_id(doc) for doc in documents]
                self.logger.info(
                    f"ingesting {len(documents)} chunks", extra={"n_chunks": len(documents)})
                with self.langfuse.start_as_current_generation(
                    name="documment embedding",
                    input={"num_documents": len(documents)},
                ) as gen:
                    await self.chroma.aadd_documents(documents, ids=document_ids)
                self.logger.info(
                    f"ingested {len(documents)} chunks", extra={"n_chunks": len(documents)})
                span.update(output="OK")
            except Exception as e:
                span.update(output=f"Error: {e}")
                raise e

    async def chunking_ingest_pdf(self, pdf_path: str|None = None, cleanup: bool = False):
        with self.langfuse.start_as_current_span(
            name="ingest", input="chunking_ingest_pdf"
        ) as span:
            chunks: List[Document] = []
            try:
                async for page, prev_page_label in self.pdf_loader_overlapping_pages(
                    pdf_path=pdf_path, cleanup=cleanup
                ):
                    page_chunks = await self.chunking_document(page, prev_page_label)
                    chunks.extend(page_chunks)
                chunk_ids = [self._get_document_id(chunk) for chunk in chunks]
                self.logger.info(
                    f"ingesting {len(chunks)} chunks", extra={"n_chunks": len(chunks)})
                with self.langfuse.start_as_current_generation(
                    name="documment embedding",
                    input={
                        "num_chunks": len(chunks),
                        "chunk_size": SETTINGS.text_splitter.chunk_size,
                    },
                ) as gen:
                    await self.chroma.aadd_documents(chunks, ids=chunk_ids)
                self.logger.info(
                    f"ingested {len(chunks)} chunks", extra={"n_chunks": len(chunks)}
                )
                span.update(output="OK")
            except Exception as e:
                span.update(output=f"Error: {e}")
                self.logger.error(f"Error: {e}", extra={"pdf_path": pdf_path, "cleanup": cleanup})
                raise e

    async def streaming_chunking_ingest_pdf(self, pdf_path: str):
        """
        idea: we may want to upload a very large pdf which may require too much memory
                so we can stream the pdf to the vector db in batches, optimizing memory usage.
        """
        raise NotImplementedError()
