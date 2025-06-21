from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # expose general fastAPI metrics
    global instrumentator
    instrumentator.expose(app)
    yield
    # close stuff
    #

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

@app.get("/health")
async def health():
    return "OK"

@app.post("/ingest")
async def ingest():
    return "OK"

@app.post("/query")
async def query():
    return
