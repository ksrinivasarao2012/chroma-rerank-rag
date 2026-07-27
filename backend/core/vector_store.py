import os
import uuid
import chromadb
import logging
from dotenv import load_dotenv
from chromadb.api.types import EmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Any, Dict

load_dotenv()

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "portfolio_rag_docs"):
        """
        Initializes the local embedding model and the persistent ChromaDB connection.
        """
        logger.info(f"Initializing VectorDBManager at path={db_path}, collection_name={collection_name}")
        os.makedirs(db_path, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=db_path)
            logger.info("ChromaDB persistent client successfully created.")
            
            logger.info("Loading HuggingFaceEmbeddings model (BAAI/bge-base-en-v1.5)...")
            self.embedding = HuggingFaceEmbeddings(
                model_name='BAAI/bge-base-en-v1.5',
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB client or embedding model.")
            raise e

        # Create a lightweight adapter so ChromaDB can use the LangChain embedder
        class ChromaAdapter(EmbeddingFunction):
            def __init__(self, embedder):
                self.embedder = embedder
                
            def __call__(self, input: list[str]) -> list[list[float]]:
                logger.debug(f"ChromaAdapter embedding {len(input)} texts...")
                return self.embedder.embed_documents(input)
                
        chroma_embed_fn = ChromaAdapter(self.embedding)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_embed_fn
        )
        logger.info(f"Connected to collection: {collection_name}")

    def delete_by_source(self, source_file: str) -> bool:
        """Deletes all existing vectors originating from a specific source file."""
        try:
            self.collection.delete(where={"source_file": source_file})
            logger.info(f"Cleared existing vectors for source file: {source_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete existing vectors for {source_file}: {e}")
            return False

    def add_chunks(self, chunks: List[Dict[str, Any]], source_file: str = None) -> bool:
        if not chunks:
            logger.warning("add_chunks called with empty chunks list")
            return False
        
        try:
            # Step 1: Clean up any previously ingested version of this file
            target_source = source_file or chunks[0]["metadata"].get("source_file")
            if target_source:
                self.delete_by_source(target_source)

            logger.info(f"Preparing to add {len(chunks)} chunks to ChromaDB...")
            # Step 2: Prepare documents, metadatas, and deterministic IDs
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [
                f"{chunk['metadata'].get('source_file', 'doc')}_{chunk['metadata'].get('page_number', 0)}_{idx}"
                for idx, chunk in enumerate(chunks)
            ]

            # Step 3: Upsert into ChromaDB
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB.")
            return True
        except Exception as e:
            logger.exception("Failed to add chunks to ChromaDB.")
            return False

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for relevant document chunks and returns a clean list of results.
        """
        logger.info(f"Performing search query: '{query}' (limit={n_results})")
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
        except Exception as e:
            logger.exception(f"Query execution failed for query: '{query}'")
            return []
        
        clean_results = []
        
        # Safely extract batches
        docs_batch = results.get("documents")
        metas_batch = results.get("metadatas")

        # Defensive safety check before unpacking index [0]
        if docs_batch and len(docs_batch) > 0 and docs_batch[0]:
            docs = docs_batch[0]
            metas = metas_batch[0] if metas_batch and len(metas_batch) > 0 else [{}] * len(docs)
            
            logger.info(f"Found {len(docs)} matching documents.")
            for doc, meta in zip(docs, metas):
                clean_results.append({
                    "text": doc,
                    "metadata": meta
                })
        else:
            logger.info("No matching documents found.")
                
        return clean_results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all stored chunks from ChromaDB for sparse BM25 indexing."""
        try:
            results = self.collection.get(include=["documents", "metadatas"])
            chunks = []

            if results and results.get("documents"):
                for text, meta in zip(results["documents"], results["metadatas"]):
                    chunks.append({"text": text, "metadata": meta})
                logger.info(f"Retrieved {len(chunks)} chunks from ChromaDB.")
            return chunks
        except Exception as e:
            logger.exception("Failed to retrieve all chunks from ChromaDB.")
            return []
