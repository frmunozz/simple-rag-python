import logging
import logging.config
from .settings import SETTINGS
import os

# load the logging config before doing any other import
logging.config.fileConfig(
    os.path.join(SETTINGS.package_root_directory, "logging_config.ini"),
    disable_existing_loggers=True,
)

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from .ingestion import Ingest
from .query import Query
import asyncio
from tempfile import NamedTemporaryFile
import shutil
from pydantic import BaseModel

logger = logging.getLogger("app.api")


class QueryRequest(BaseModel):
    question: str
    k: int = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    # expose general fastAPI metrics
    global instrumentator, ingestion, ingestion_lock, ingestion_task
    instrumentator.expose(app)
    logger.info("ingestion on startup")
    ok, pdf_path = ingestion.validate_file()
    if not ok:
        logger.info("ingestion on startup skipped, file not changed")
    else:
        await ingestion.chunking_ingest_pdf(pdf_path, cleanup=False)
    yield
    # close stuff
    #
    async with ingestion_lock:
        if ingestion_task is not None:
            ingestion_task.cancel()
            try:
                await ingestion_task
            except:
                pass
            ingestion_task = None


app = FastAPI(lifespan=lifespan, title="RAG API", version="0.0.1")

# CORS allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator = Instrumentator().instrument(app)
ingestion = Ingest()
query_llm = Query(ingestion)

# define an ingestion lock to prevent duplicated ingests
# but if we want to run the API with more than 1 worker, we may need
# to adapt this logic with a shared cache memory like redis.
ingestion_lock = asyncio.Lock()
ingestion_task: asyncio.Task | None = None


@app.get("/health")
async def health():
    return "OK"


@app.post("/ingest")
async def ingest(file: UploadFile | None = None):
    global ingestion_task, ingestion_lock, ingestion

    if ingestion_lock.locked():
        logger.warning("Ingestion already in progress")
        raise HTTPException(status_code=400, detail="Ingestion already in progress")

    async with ingestion_lock:
        if ingestion_task is not None and not ingestion_task.done():
            logger.warning("Ingestion already in progress")
            raise HTTPException(409, "Ingestion already in progress")
        pdf_path = None
        if file is not None:
            if file.filename is None or not file.filename.endswith(".pdf"):
                logger.warning("Only PDF files are supported")
                raise HTTPException(400, "Only PDF files are supported")

            try:
                # Save to temp file
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    pdf_path = tmp.name
            except Exception as e:
                logger.error(f"Failed to save file: {e}")
                raise HTTPException(500, f"Failed to save file: {e}")
            finally:
                # close file
                file.file.close()
        valid, pdf_path = ingestion.validate_file(pdf_path)
        if not valid:
            logger.warning("File has not changed since last ingestion, skipping...")
            raise HTTPException(304, "File has not changed since last ingestion")
        ingestion_task = asyncio.create_task(ingestion.chunking_ingest_pdf(pdf_path))
        logger.info("Ingestion started correctly")
    return "OK"


@app.post("/query")
async def query(query_request: QueryRequest):
    try:
        return await query_llm.query(query_request.question, k=query_request.k)
    except Exception as e:
        logger.error(f"Failed to query: {e}")
        raise HTTPException(500, f"Failed to query: {e}")
