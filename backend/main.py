import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router 

# Configure unified logging format across the entire backend
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title = "Portfolio RAG API",
    description = "Backend API for document ingestion and vector search",
    version = '1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_origins=["*"],
    allow_headers=["*"],
)


app.include_router(router)

@app.get('/')
async def root():
    return {'status' : 'ok','message' : 'API is running.'}
