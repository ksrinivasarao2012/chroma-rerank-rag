import logging
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25: BM25Okapi = None
        self.corpus_chunks: List[Dict[str, Any]] = []

    def tokenize(self, text: str) -> List[str]:
        """Simple lower-case alphanumeric tokenizer for BM25."""
        return re.findall(r'\w+', text.lower())
    
    def update_index(self, all_chunks: List[Dict[str, Any]]):
        """Rebuilds the BM25 index whenever new document chunks are ingested."""
        if not all_chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return
        
        self.corpus_chunks = all_chunks
        tokenized_corpus = [self.tokenize(chunk["text"]) for chunk in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index updated with {len(all_chunks)} chunks.")

    def bm25_search(self, query: str, top_n: int = 15) -> List[Dict[str, Any]]:
        """Performs sparse keyword search using BM25."""
        if not self.bm25 or not self.corpus_chunks:
            logger.warning("BM25 index is empty. Returning empty list.")
            return []
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair chunks with scores and sort in descending order
        scored_chunks = list(zip(self.corpus_chunks, scores))
        sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)

        # Filter out 0-score matches and return up to top_n results
        results = [chunk for chunk, score in sorted_chunks if score > 0][:top_n]
        return results

    def reciprocal_rank_fusion(self,dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """Merges vector search and BM25 search results using RRF scoring."""

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        def process_rank_list(results: List[Dict[str, Any]]):
            for rank, item in enumerate(results, start=1):
                # Use snippet text as document fingerprint key
                doc_id = item["text"]
                doc_map[doc_id] = item
                
                # Calculate RRF score
                score = 1.0 / (self.rrf_k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        # Score dense and sparse results
        process_rank_list(dense_results)
        process_rank_list(sparse_results)

        # Sort documents by accumulated RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_n]]
        return fused_results