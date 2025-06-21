from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from .ingestion import Ingest
from .settings import SETTINGS
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # expose general fastAPI metrics
    global instrumentator, ingestion
    instrumentator.expose(app)
    t = asyncio.create_task(ingestion.chunking_ingest_pdf(SETTINGS.pdf_path))
    yield
    # close stuff
    #
    t.cancel()
    await t

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

@app.get("/health")
async def health():
    return "OK"

@app.post("/ingest")
async def ingest():
    return "OK"

@app.post("/query")
async def query():
    return
