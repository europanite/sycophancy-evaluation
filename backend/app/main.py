from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import benchmark, evidence, performance, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="sycophancy-evaluation API",
    version="0.6.0",
    description="Auditable sycophancy, LLM capability, and RAG performance diagnostics.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "sycophancy-evaluation"}


app.include_router(benchmark.router)
app.include_router(evidence.router)
app.include_router(performance.router)
app.include_router(rag.router)
