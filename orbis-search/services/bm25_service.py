"""
BM25 keyword search service for hybrid retrieval using bm25s (fast implementation).
"""
import logging
from typing import List, Dict, Any, Optional
import bm25s
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BM25Service:
    """Service for BM25 keyword-based search using bm25s"""

    def __init__(self, persist_dir: str = "./data/bm25_index"):
        self.retriever = None
        self.corpus_data: List[Dict[str, Any]] = []  # Store original documents with metadata
        self.persist_dir = persist_dir
        self._ensure_persist_dir()

    def _ensure_persist_dir(self):
        """Ensure persistence directory exists"""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

    def index_documents(self, documents: List[Dict[str, Any]], save_to_disk: bool = True) -> None:
        """
        Index documents for BM25 search using bm25s.

        Args:
            documents: List of dicts with 'ticket' and 'concatenated_text' keys
            save_to_disk: Whether to save the index to disk after indexing
        """
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return

        self.corpus_data = documents

        # Extract text corpus
        corpus = [doc.get('concatenated_text', '') for doc in documents]

        # Tokenize corpus using bm25s
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        # Create and index the retriever
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

        logger.info(f"Indexed {len(documents)} documents for BM25 search using bm25s")

        if save_to_disk:
            self.save_index()

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Search using BM25 and return scored results.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of dicts with 'ticket', 'bm25_score', and 'concatenated_text'
        """
        if not self.retriever or not self.corpus_data:
            logger.warning("BM25 index not initialized")
            return []

        # Tokenize query using bm25s
        query_tokens = bm25s.tokenize(query, stopwords="en")

        if not query_tokens or len(query_tokens[0]) == 0:
            logger.warning("Empty query after tokenization")
            return []

        # Retrieve top_k documents and scores
        # bm25s returns (docs, scores) where docs are indices and scores are relevance scores
        docs_indices, scores = self.retriever.retrieve(query_tokens, k=min(top_k, len(self.corpus_data)))

        # Create results with scores
        results = []
        # docs_indices and scores are 2D arrays (batch_size x k), we take first batch
        if len(docs_indices) > 0 and len(scores) > 0:
            for idx, score in zip(docs_indices[0], scores[0]):
                if idx < len(self.corpus_data):
                    results.append({
                        'ticket': self.corpus_data[idx]['ticket'],
                        'bm25_score': float(score),
                        'concatenated_text': self.corpus_data[idx].get('concatenated_text', ''),
                        'similarity_score': self.corpus_data[idx].get('similarity_score', 0.0)
                    })

        # Results are already sorted by BM25 score descending from bm25s
        return results

    def is_initialized(self) -> bool:
        """Check if BM25 index is initialized"""
        return self.retriever is not None and len(self.corpus_data) > 0

    def get_corpus_size(self) -> int:
        """Get the number of documents in the corpus"""
        return len(self.corpus_data)

    def save_index(self) -> None:
        """Save BM25 index and corpus data to disk"""
        try:
            self._ensure_persist_dir()

            # Save the BM25 retriever
            retriever_path = os.path.join(self.persist_dir, "bm25_retriever")
            self.retriever.save(retriever_path, corpus=None)

            # Save corpus data separately (tickets + metadata)
            corpus_path = os.path.join(self.persist_dir, "corpus_data.json")
            # Serialize corpus data (convert ticket objects to dicts for JSON serialization)
            serialized_corpus = []
            for doc in self.corpus_data:
                serialized_doc = {
                    'ticket': doc['ticket'].model_dump() if hasattr(doc['ticket'], 'model_dump') else doc['ticket'],
                    'concatenated_text': doc.get('concatenated_text', ''),
                    'similarity_score': doc.get('similarity_score', 0.0)
                }
                serialized_corpus.append(serialized_doc)

            with open(corpus_path, 'w', encoding='utf-8') as f:
                json.dump(serialized_corpus, f, default=str)

            logger.info(f"BM25 index saved to {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            raise

    def load_index(self) -> bool:
        """
        Load BM25 index and corpus data from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            retriever_path = os.path.join(self.persist_dir, "bm25_retriever")
            corpus_path = os.path.join(self.persist_dir, "corpus_data.json")

            # Check if files exist
            if not os.path.exists(retriever_path) or not os.path.exists(corpus_path):
                logger.info("BM25 index files not found on disk")
                return False

            # Load the BM25 retriever
            self.retriever = bm25s.BM25.load(retriever_path, mmap=True)

            # Load corpus data
            with open(corpus_path, 'r', encoding='utf-8') as f:
                serialized_corpus = json.load(f)

            # Deserialize corpus data (keep as dicts, convert back to Ticket objects when needed)
            from models.schemas import Ticket
            self.corpus_data = []
            for doc in serialized_corpus:
                self.corpus_data.append({
                    'ticket': Ticket(**doc['ticket']),
                    'concatenated_text': doc.get('concatenated_text', ''),
                    'similarity_score': doc.get('similarity_score', 0.0)
                })

            logger.info(f"BM25 index loaded from {self.persist_dir} with {len(self.corpus_data)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def clear_index(self) -> None:
        """Clear the in-memory index and delete persisted files"""
        try:
            self.retriever = None
            self.corpus_data = []

            # Delete persisted files
            retriever_path = os.path.join(self.persist_dir, "bm25_retriever")
            corpus_path = os.path.join(self.persist_dir, "corpus_data.json")

            if os.path.exists(corpus_path):
                os.remove(corpus_path)
            if os.path.exists(retriever_path):
                # BM25 saves as a directory with multiple files
                import shutil
                if os.path.isdir(retriever_path):
                    shutil.rmtree(retriever_path)
                else:
                    os.remove(retriever_path)

            logger.info("BM25 index cleared from memory and disk")
        except Exception as e:
            logger.error(f"Failed to clear BM25 index: {e}")
            raise