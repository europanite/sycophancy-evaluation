from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import benchmark


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="sycophancy-evaluation API",
    version="0.4.2",
    description="Controlled benchmark for detecting and quantifying LLM sycophancy.",
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
