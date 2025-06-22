from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings  # if doing local embedding with ollama
from langchain_openai import OpenAIEmbeddings  # if using openai for embeddings
from typing import List, TypedDict
from .settings import SETTINGS, Provider
import os
from langfuse import Langfuse
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from hashlib import md5
import logging


class SimilaritySearchResult(TypedDict):
    text: str
    page: str


class Ingest:
    def __init__(self):
        """
        Initialize the ingestion service with the correct settings, embeddings, and chroma index.
        
        This method initializes the ingestion service with the following:
        
        - Langfuse instance for observability
        - TextSplitter for splitting documents into chunks
        - Embeddings model for generating document embeddings
        - Chroma index for storing and querying document embeddings
        
        It also sets up the logger for the ingestion service.
        """
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
        self.embeddings = self._get_embeddings_model()

        self.chroma = Chroma(
            embedding_function=self.embeddings,
            persist_directory=os.path.join(
                SETTINGS.package_root_directory, "chroma_data"
            ),
        )

    def _get_embeddings_model(self):
        """
        Select and initialize the appropriate embeddings model based on the configured provider.

        :return: An instance of an embeddings model (either OpenAIEmbeddings or OllamaEmbeddings)
        :raises ValueError: If the provider specified in the settings is not supported
        """
        if SETTINGS.provider == Provider.openai:
            return OpenAIEmbeddings(
                model=SETTINGS.openai.embedding_model,
                api_key=SETTINGS.openai.api_key,
            )
        elif SETTINGS.provider == Provider.ollama:
            return OllamaEmbeddings(
                model=SETTINGS.ollama.embedding_model, 
                base_url=SETTINGS.ollama.host
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {SETTINGS.provider}")

    def _get_document_id(self, document: Document):
        """
        Generate a unique identifier for a document based on its content.

        This method calculates the MD5 hash of the page content of the provided
        document to create a unique identifier. The metadata hash is commented out
        and not used in the current implementation.

        :param document: A Document object whose ID needs to be generated.
        :return: A string representing the MD5 hash of the document's content.
        """
        content_hash = md5(document.page_content.encode("utf-8")).hexdigest()
        # meta_hash = md5(
        #     json.dumps(document.metadata, sort_keys=True).encode("utf-8")
        # ).hexdigest()
        return content_hash

    async def pdf_loader_pages(
        self, pdf_path: str | None = None, cleanup: bool = False
    ):
        """
        Loads a PDF file, pages it, and yields each page as a Document
        object.

        If `cleanup` is True, the PDF file is removed after it is
        successfully loaded.

        :param pdf_path: The path to the PDF file to load. If None, the
            `pdf_path` specified in the settings is used.
        :type pdf_path: str | None
        :param cleanup: Whether to remove the PDF file after loading.
        :type cleanup: bool
        :yield: A Document object representing a page in the PDF.
        :rtype: Iterator[Document]
        """
       
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
                self.logger.error(
                    f"Failed to remove file: {e}", extra={"pdf_path": pdf_path}
                )

    async def pdf_loader_overlapping_pages(
        self, pdf_path: str | None = None, cleanup: bool = False
    ):
        """
        Loads a PDF file, pages it, and yields each page as a Document
        object with possible overlap with the previous page.

        The overlap is specified by the `chunk_overlap` setting in the
        `text_splitter` section of the configuration file.

        If `cleanup` is True, the PDF file is removed after it is
        successfully loaded.

        :param pdf_path: The path to the PDF file to load. If None, the
            `pdf_path` specified in the settings is used.
        :type pdf_path: str | None
        :param cleanup: Whether to remove the PDF file after loading.
        :type cleanup: bool
        :yield: A tuple of a Document object representing a page in the PDF
            and the label of the previous page.
        :rtype: Iterator[Tuple[Document, str | None]]
        """
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
        """
        Splits a document into chunks of approximately `chunk_size` characters.

        If the document has less than `chunk_size` characters, it is returned as a single chunk.
        Otherwise, the document is split into chunks of `chunk_size` characters.
        The first chunk is special: if the previous page existed, it will contain the last
        `chunk_overlap` characters of the previous page.

        :param document: The document to split.
        :type document: Document
        :param prev_page_label: The label of the previous page, if any.
        :type prev_page_label: str | None
        :return: A list of Document objects, each representing a chunk of the input document.
        :rtype: List[Document]
        """
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

    async def direct_ingest_pdf(
        self, pdf_path: str | None = None, cleanup: bool = False
    ):
        """
        Loads a PDF file and adds it to the ChromaDB database.

        If `cleanup` is True, the PDF file is removed after it is
        successfully loaded.

        :param pdf_path: The path to the PDF file to load. If None, the
            `pdf_path` specified in the settings is used.
        :type pdf_path: str | None
        :param cleanup: Whether to remove the PDF file after loading.
        :type cleanup: bool
        """
        with self.langfuse.start_as_current_span(
            name="ingest", input="direct_ingest_pdf"
        ) as span:
            try:
                documents = [
                    page
                    async for page in self.pdf_loader_pages(
                        pdf_path=pdf_path, cleanup=cleanup
                    )
                ]
                # use ids to prevent repeating embeddings for same pages
                document_ids = [self._get_document_id(doc) for doc in documents]
                self.logger.info(
                    f"ingesting {len(documents)} chunks",
                    extra={"n_chunks": len(documents)},
                )
                with self.langfuse.start_as_current_generation(
                    name="documment embedding",
                    input={"num_documents": len(documents)},
                ) as gen:
                    await self.chroma.aadd_documents(documents, ids=document_ids)
                self.logger.info(
                    f"ingested {len(documents)} chunks",
                    extra={"n_chunks": len(documents)},
                )
                span.update(output="OK")
            except Exception as e:
                span.update(output=f"Error: {e}")
                raise e

    async def chunking_ingest_pdf(
        self, pdf_path: str | None = None, cleanup: bool = False
    ):
        """
        Loads a PDF file, chunks it into overlapping pages, and ingests each chunk
        into the Chroma index.

        The overlap is specified by the `chunk_overlap` setting in the
        `text_splitter` section of the configuration file.

        If `cleanup` is True, the PDF file is removed after it is
        successfully loaded.

        :param pdf_path: The path to the PDF file to load. If None, the
            `pdf_path` specified in the settings is used.
        :type pdf_path: str | None
        :param cleanup: Whether to remove the PDF file after loading.
        :type cleanup: bool
        """
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
                    f"ingesting {len(chunks)} chunks", extra={"n_chunks": len(chunks)}
                )
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
                self.logger.error(
                    f"Error: {e}", extra={"pdf_path": pdf_path, "cleanup": cleanup}
                )
                raise e

    async def streaming_chunking_ingest_pdf(self, pdf_path: str):
        """
        Lazy loads a PDF file and chunks it into overlapping pages, streaming the chunks
        into the Chroma index as they are readed and processed to reduce memory usage.

        The overlap is specified by the `chunk_overlap` setting in the
        `text_splitter` section of the configuration file.

        NOTE: This method is not implemented yet.

        :param pdf_path: The path to the PDF file to load.
        :type pdf_path: str
        """
        raise NotImplementedError()

    async def similarity_search(
        self, query: str, k: int = 3
    ) -> List[SimilaritySearchResult]:
        """
        Perform a similarity search on the chromaDB database.

        :param query: The query text to search for
        :type query: str
        :param k: The number of results to return (default is 3)
        :type k: int
        :return: A list of SimilaritySearchResult objects
        :rtype: List[SimilaritySearchResult]
        """
        with self.langfuse.start_as_current_span(
            name="similarity search", input=query
        ) as span:
            results = await self.chroma.asimilarity_search(query, k=k)
            result_formatted = [
                SimilaritySearchResult(
                    text=doc.page_content,
                    page=str(
                        doc.metadata.get("page_label", doc.metadata.get("page", "N/A"))
                    ),
                )
                for doc in results
            ]
            span.update(output=result_formatted)
            return result_formatted
