"""
BM25 keyword search service for hybrid retrieval using bm25s (fast implementation).
"""
import logging
from typing import Any
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BM25Service:
    """Service for BM25 keyword-based search using bm25s"""

    def __init__(self, persist_dir: str = "./data/bm25_index"):
        try:
            import bm25s
            self.bm25s = bm25s
        except ImportError:
            logger.warning("bm25s package not available; BM25Service will be disabled")
            self.bm25s = None

        self.retriever = None
        self.corpus_data: list[dict[str, Any]] = []
        self.persist_dir = persist_dir
        self._ensure_persist_dir()

    def _ensure_persist_dir(self):
        """Ensure persistence directory exists"""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

    def index_documents(self, documents: list[dict[str, Any]], save_to_disk: bool = True) -> None:
        """
        Index documents for BM25 search.

        Args:
            documents: List of dicts with document data and 'concatenated_text' key
            save_to_disk: Whether to save the index to disk after indexing
        """
        if self.bm25s is None:
            logger.warning("bm25s not available, skipping indexing")
            return

        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return

        self.corpus_data = documents

        # Extract text corpus
        corpus = [doc.get('concatenated_text', '') for doc in documents]

        # Tokenize corpus using bm25s
        corpus_tokens = self.bm25s.tokenize(corpus, stopwords="en")

        # Create and index the retriever
        self.retriever = self.bm25s.BM25()
        self.retriever.index(corpus_tokens)

        logger.info(f"Indexed {len(documents)} documents for BM25 search")

        if save_to_disk:
            self.save_index()

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """
        Search using BM25 and return scored results.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of dicts with document data and 'bm25_score'
        """
        if self.bm25s is None or not self.retriever or not self.corpus_data:
            logger.warning("BM25 index not initialized")
            return []

        # Tokenize query
        query_tokens = self.bm25s.tokenize(query, stopwords="en")

        if not query_tokens or len(query_tokens[0]) == 0:
            logger.warning("Empty query after tokenization")
            return []

        # Retrieve top_k documents and scores
        docs_indices, scores = self.retriever.retrieve(query_tokens, k=min(top_k, len(self.corpus_data)))

        # Create results with scores
        results = []
        if len(docs_indices) > 0 and len(scores) > 0:
            for idx, score in zip(docs_indices[0], scores[0]):
                if idx < len(self.corpus_data):
                    result = self.corpus_data[idx].copy()
                    result['bm25_score'] = float(score)
                    results.append(result)

        return results

    def is_initialized(self) -> bool:
        """Check if BM25 index is initialized"""
        return self.retriever is not None and len(self.corpus_data) > 0

    def get_corpus_size(self) -> int:
        """Get the number of documents in the corpus"""
        return len(self.corpus_data)

    def save_index(self) -> None:
        """Save BM25 index and corpus data to disk"""
        if self.bm25s is None:
            return

        try:
            self._ensure_persist_dir()

            # Save the BM25 retriever
            retriever_path = os.path.join(self.persist_dir, "bm25_retriever")
            self.retriever.save(retriever_path, corpus=None)

            # Save corpus data separately
            corpus_path = os.path.join(self.persist_dir, "corpus_data.json")
            with open(corpus_path, 'w', encoding='utf-8') as f:
                json.dump(self.corpus_data, f, default=str)

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
        if self.bm25s is None:
            return False

        try:
            retriever_path = os.path.join(self.persist_dir, "bm25_retriever")
            corpus_path = os.path.join(self.persist_dir, "corpus_data.json")

            # Check if files exist
            if not os.path.exists(retriever_path) or not os.path.exists(corpus_path):
                logger.info("BM25 index files not found on disk")
                return False

            # Load the BM25 retriever
            self.retriever = self.bm25s.BM25.load(retriever_path, mmap=True)

            # Load corpus data
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.corpus_data = json.load(f)

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
                import shutil
                if os.path.isdir(retriever_path):
                    shutil.rmtree(retriever_path)
                else:
                    os.remove(retriever_path)

            logger.info("BM25 index cleared from memory and disk")
        except Exception as e:
            logger.error(f"Failed to clear BM25 index: {e}")
            raise
