from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
import logging
import uuid
import json

from schemas.pydantic_models import QueryRequest, UploadResponse
from core.ingestion import DocumentProcessor
from core.vector_store import VectorDBManager
from core.llm_service import LLMService
from core.reranker import ReRanker
from core.hybrid_search import HybridRetriever


router = APIRouter(prefix = '/api/v1',tags = ['RAG'])
doc_processor = DocumentProcessor()
vector_db = VectorDBManager()
llm_service = LLMService()
reranker = ReRanker()
hybrid_retriever = HybridRetriever()

logger = logging.getLogger(__name__)

# Sync BM25 index with existing ChromaDB chunks on startup
initial_chunks = vector_db.get_all_chunks()
if initial_chunks:
    hybrid_retriever.update_index(initial_chunks)
    logger.info(f"Startup: Synchronized BM25 index with {len(initial_chunks)} existing chunks from ChromaDB.")


@router.post('/upload', response_model = UploadResponse) 
async def upload_document(file : UploadFile = File(..., description = "PDF to be uploaded")):
    logger.info(f"Received file upload request for: {file.filename}")

    if file.content_type != 'application/pdf':
        logger.warning(f"Rejected upload for {file.filename}: Invalid content-type '{file.content_type}'")
        raise HTTPException(status_code = 400, detail = 'Only PDF files are supported')
    
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Rejected upload for {file.filename}: File extension must be .pdf")
        raise HTTPException(status_code = 400, detail = 'Only PDF files are supported')
    
    file_bytes = await file.read()
    chunks = doc_processor.process_stream(file_bytes = file_bytes, filename = file.filename)
    
    if len(chunks) == 0:
        logger.error(f"Failed to extract any non-empty chunks from {file.filename}")
        raise HTTPException(status_code = 422, detail = 'Failed to extract text from PDF.')

    if not vector_db.add_chunks(chunks=chunks, source_file=file.filename):
        logger.error(f"Failed to store chunks for {file.filename} in ChromaDB")
        raise HTTPException(status_code=500, detail='Database storage failed.')
    
    # Update BM25 index with all active corpus chunks
    all_chunks = vector_db.get_all_chunks()
    hybrid_retriever.update_index(all_chunks)
    logger.info(f"Successfully processed {file.filename} with {len(chunks)} chunks and updated BM25 index.")
    
    return {
        "filename": file.filename,
        "message": "Upload success (previous versions overwritten)",
        "document_id": str(uuid.uuid4())
    }

@router.post('/query')
async def query_rag(payload: QueryRequest):
    logger.info(f"Received query request: '{payload.query}' (top_k={payload.top_k}, history_len={len(payload.chat_history or [])})")
    
    if not payload.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code = 400, detail = 'Query cannot be empty.')
    
    CANDIDATE_K = 15
    
    # Stage 1a: Dense Vector Search
    dense_results = vector_db.search(query = payload.query, n_results = CANDIDATE_K)
    logger.info(f"Stage 1a (Vector Search): Retrieved {len(dense_results)} candidate chunks from ChromaDB")

    # Stage 1b: Sparse BM25 Keyword Search
    sparse_results = hybrid_retriever.bm25_search(query = payload.query, top_n = CANDIDATE_K)
    logger.info(f"Stage 1b (BM25 Search): Retrieved {len(sparse_results)} candidate chunks from BM25 index")

    # Stage 1c: Reciprocal Rank Fusion (RRF)
    fused_candidates = hybrid_retriever.reciprocal_rank_fusion(
        dense_results = dense_results,
        sparse_results = sparse_results,
        top_n = CANDIDATE_K
    )
    logger.info(f"Stage 1c (RRF Fusion): Merged and deduplicated to {len(fused_candidates)} fused candidates")

    # Stage 2: Cross-Encoder Re-Ranking
    reranked_results = reranker.rerank(
        query = payload.query, 
        chunks = fused_candidates, 
        top_k = payload.top_k
    )
    logger.info(f"Stage 2 (Cross-Encoder): Re-ranked candidates down to top {len(reranked_results)} results")
    
    # Build citations list from re-ranked results
    citations_list = []
    for res in reranked_results:
        citations_list.append({
            'source_file': res['metadata'].get('source_file', 'Unknown'),
            'page_number': res['metadata'].get('page_number', 0),
            'text_snippet': res["text"] 
        })
    logger.info(f"Prepared {len(citations_list)} citations for LLM streaming response")
    
    # Asynchronous NDJSON Stream Generator
    async def response_generator():
        logger.info("Starting NDJSON stream response...")
        # Yield citations first as a single JSON line
        yield json.dumps({
            "type": "citations",
            "data": citations_list
        }) + '\n'

        # Stream the LLM tokens as they are generated
        try:
            token_count = 0
            async for token in llm_service.stream_answer(
                query = payload.query,
                citations = citations_list,
                chat_history = payload.chat_history
            ): 
                token_count += 1
                yield json.dumps({
                    "type": "token",
                    "data": token
                }) + '\n'
            logger.info(f"Finished NDJSON stream successfully ({token_count} token chunks emitted).")

        except Exception as e:
            logger.error(f"Streaming error occurred during generation: {e}")
            yield json.dumps({"type": "token", "data": "\n[Stream interrupted due to an error.]"}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")