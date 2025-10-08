"""
Generic Vector Service - Configuration-driven vector storage.
Handles vector storage for ANY content type without hardcoded assumptions.
Replaces hardcoded content type logic with dynamic collection management.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from engine.config.data_sources import (
    get_all_collection_names,
    get_collection_name,
    is_valid_source_type,
)

logger = logging.getLogger(__name__)


class GenericVectorService:
    """Handles vector storage for ANY content type via configuration"""

    def __init__(self):
        self.client: chromadb.Client | None = None
        self.collections: dict[str, chromadb.Collection] = {}
        self.db_path = settings.CHROMA_DB_PATH

        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and all configured collections"""
        try:
            # Ensure the directory exists
            os.makedirs(self.db_path, exist_ok=True)

            logger.info(f"Initializing ChromaDB client at path: {self.db_path}")

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Initialize all configured collections
            self._initialize_collections()

            logger.info("Generic ChromaDB service initialized successfully")
            logger.info(f"Database path: {self.db_path}")

            # Log collection info
            collections_summary = {name: coll.count() for name, coll in self.collections.items()}
            total_docs = sum(collections_summary.values())
            logger.info(f"{len(self.collections)} collections, {total_docs} total documents")
            for name, count in collections_summary.items():
                logger.debug(f"  {name}: {count} documents")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _initialize_collections(self):
        """Initialize all configured collections from data source registry"""
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")

        try:
            # Get all collection names from configuration
            collection_names = get_all_collection_names()

            # Create collections
            for collection_name in collection_names:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.collections[collection_name] = collection
                logger.debug(f"Initialized collection: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise

    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get or create a collection by name"""
        if collection_name not in self.collections:
            if not self.client:
                raise RuntimeError("ChromaDB client not initialized")

            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.collections[collection_name] = collection
            logger.debug(f"Created new collection: {collection_name}")

        return self.collections[collection_name]

    async def store_content(self, content_items: list[dict[str, Any]],
                          embeddings: list[list[float]],
                          collection_name: str,
                          clear_existing: bool = False) -> bool:
        """
        Store content and embeddings in specified collection.

        Args:
            content_items: List of content dictionaries with id, text, metadata
            embeddings: List of embedding vectors
            collection_name: Target collection name
            clear_existing: Whether to clear existing collection first

        Returns:
            True if successful
        """
        if len(content_items) != len(embeddings):
            raise ValueError("Number of content items and embeddings must match")

        if not content_items:
            return True

        try:
            # Get or create collection
            collection = self.get_or_create_collection(collection_name)

            # Clear existing collection if requested
            if clear_existing:
                self._recreate_collection(collection_name)
                collection = self.get_or_create_collection(collection_name)

            # Prepare data for storage
            # Deduplicate items within the batch by ID (keep last occurrence with its embedding)
            items_by_id = {}
            embeddings_by_id = {}
            for i, item in enumerate(content_items):
                item_id = str(item["id"])
                items_by_id[item_id] = item
                embeddings_by_id[item_id] = embeddings[i]

            # Extract deduplicated items and their embeddings
            unique_items = list(items_by_id.values())
            unique_embeddings = [embeddings_by_id[str(item["id"])] for item in unique_items]

            if len(unique_items) < len(content_items):
                logger.warning(f"Deduplication: {len(content_items)} items reduced to {len(unique_items)} unique items")

            ids = [str(item["id"]) for item in unique_items]
            documents = [item["text"] for item in unique_items]
            metadatas = []

            # Process metadata - ensure all values are JSON serializable
            for item in unique_items:
                metadata = item.get("metadata", {}).copy()

                # Add system metadata
                metadata.update({
                    "content_id": str(item["id"]),
                    "original_id": str(item["id"]),
                    "concatenated_text": item["text"],
                    "source_type": item.get("source_type", "unknown"),
                    "source_name": item.get("source_name", "unknown"),
                    "ingestion_timestamp": datetime.now().isoformat()
                })

                # Ensure all metadata values are JSON serializable strings
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if value is None:
                        cleaned_metadata[key] = ""
                    elif isinstance(value, str | int | float | bool):
                        cleaned_metadata[key] = str(value)
                    elif isinstance(value, list | dict):
                        cleaned_metadata[key] = json.dumps(value)
                    else:
                        cleaned_metadata[key] = str(value)

                metadatas.append(cleaned_metadata)

            # Store in ChromaDB - use upsert to handle duplicate IDs
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Stored {len(content_items)} items in collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to store content in collection '{collection_name}': {e}")
            raise

    async def search_content(self, query_embedding: list[float],
                           collection_names: list[str],
                           top_k: int = 10,
                           where_filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Search across specified collections.

        Args:
            query_embedding: Query embedding vector
            collection_names: List of collection names to search
            top_k: Number of results per collection
            where_filters: Optional metadata filters

        Returns:
            List of search results with normalized scores
        """
        all_results = []

        for collection_name in collection_names:
            if collection_name not in self.collections:
                logger.warning(f"Collection '{collection_name}' not found, skipping")
                continue

            try:
                collection = self.collections[collection_name]

                query_params = {
                    "query_embeddings": [query_embedding],
                    "n_results": top_k,
                    "include": ["metadatas", "documents", "distances"]
                }

                if where_filters:
                    query_params["where"] = where_filters

                results = collection.query(**query_params)

                # Process results from this collection
                if results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        result_item = {
                            "id": results['ids'][0][i],
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "distance": results['distances'][0][i],
                            "collection": collection_name,
                            "similarity": 1.0 - results['distances'][0][i]  # Convert distance to similarity
                        }
                        all_results.append(result_item)

            except Exception as e:
                logger.error(f"Error searching collection '{collection_name}': {e}")
                continue

        # Sort all results by similarity (highest first)
        all_results.sort(key=lambda x: x["similarity"], reverse=True)

        return all_results

    async def search_by_source_type(self, query_embedding: list[float],
                                   source_types: list[str],
                                   top_k: int = 10,
                                   where_filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Search collections by data source types.

        Args:
            query_embedding: Query embedding vector
            source_types: List of data source types to search
            top_k: Number of results per collection
            where_filters: Optional metadata filters

        Returns:
            List of search results
        """
        collection_names = []

        for source_type in source_types:
            if is_valid_source_type(source_type):
                collection_name = get_collection_name(source_type)
                if collection_name not in collection_names:
                    collection_names.append(collection_name)
            else:
                logger.warning(f"Invalid source type: {source_type}")

        return await self.search_content(
            query_embedding, collection_names, top_k, where_filters
        )

    async def search_all_collections(self, query_embedding: list[float],
                                   top_k: int = 10,
                                   where_filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search across all available collections"""
        collection_names = list(self.collections.keys())
        return await self.search_content(
            query_embedding, collection_names, top_k, where_filters
        )

    def get_collection_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all collections"""
        stats = {}

        for collection_name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[collection_name] = {
                    "document_count": count,
                    "collection_name": collection_name
                }
            except Exception as e:
                logger.error(f"Error getting stats for collection '{collection_name}': {e}")
                stats[collection_name] = {
                    "document_count": 0,
                    "error": str(e)
                }

        return stats

    def _recreate_collection(self, collection_name: str):
        """Drop and recreate a collection"""
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")

        try:
            # Delete existing collection
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass  # Ignore if collection doesn't exist

            # Create new collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.collections[collection_name] = collection
            logger.info(f"Recreated collection: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to recreate collection '{collection_name}': {e}")
            raise

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection completely"""
        try:
            if collection_name in self.collections:
                del self.collections[collection_name]

            if self.client:
                self.client.delete_collection(name=collection_name)

            logger.info(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    async def add_content_batch(self, content_items: list[dict[str, Any]],
                              collection_name: str,
                              embeddings: list[list[float]] | None = None) -> bool:
        """
        Add a batch of content to a collection.
        This method provides compatibility with existing vector service interface.
        """
        if embeddings is None:
            raise ValueError("Embeddings must be provided")

        return await self.store_content(content_items, embeddings, collection_name)

    def get_collection_names(self) -> list[str]:
        """Get all available collection names"""
        return list(self.collections.keys())

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        return collection_name in self.collections

    async def get_documents_by_metadata(self, collection_name: str, where_filters: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve documents from a collection by metadata filter without semantic search.

        Args:
            collection_name: Name of the collection to search
            where_filters: Optional metadata filters
            limit: Optional limit on number of results

        Returns:
            List of documents with metadata
        """
        try:
            if collection_name not in self.collections:
                logger.warning(f"Collection '{collection_name}' not found")
                return []

            collection = self.collections[collection_name]

            # Use ChromaDB's get method to retrieve documents by metadata filter
            query_params = {
                "include": ["metadatas", "documents"]
            }

            if where_filters:
                query_params["where"] = where_filters

            if limit:
                query_params["limit"] = limit

            results = collection.get(**query_params)

            # Process results
            documents = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    document = {
                        "id": results['ids'][i],
                        "content": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    documents.append(document)

            logger.debug(f"Retrieved {len(documents)} documents from collection '{collection_name}' with filters: {where_filters}")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents from collection '{collection_name}': {e}")
            return []

    def store_embeddings(self, content_items: list[Any],
                        embeddings: list[list[float]],
                        clear_existing: bool = False) -> bool:
        """
        Store embeddings for content items.
        This is a synchronous version that directly calls ChromaDB without async.

        Args:
            content_items: List of content items (can be dicts or Pydantic models)
            embeddings: List of embedding vectors
            clear_existing: Whether to clear existing collection first

        Returns:
            True if successful
        """
        if len(content_items) != len(embeddings):
            raise ValueError("Number of content items and embeddings must match")

        if not content_items:
            return True

        # Helper function to safely get attribute from dict or object
        def safe_get(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default)

        # Determine collection name from first item's source_type
        first_item = content_items[0]
        source_type = safe_get(first_item, "source_type")

        if not source_type:
            raise ValueError("Content items must have 'source_type' field")

        if not is_valid_source_type(source_type):
            raise ValueError(f"Invalid source type: {source_type}")

        collection_name = get_collection_name(source_type)

        try:
            # Get or create collection
            collection = self.get_or_create_collection(collection_name)

            # Clear existing collection if requested
            if clear_existing:
                self._recreate_collection(collection_name)
                collection = self.get_or_create_collection(collection_name)

            # Prepare data for storage
            # Deduplicate items within the batch by ID (keep last occurrence with its embedding)
            items_by_id = {}
            embeddings_by_id = {}
            for i, item in enumerate(content_items):
                item_id = safe_get(item, "id")
                if not item_id:
                    raise ValueError("Content item missing 'id' field")
                item_id = str(item_id)
                items_by_id[item_id] = item
                embeddings_by_id[item_id] = embeddings[i]

            # Extract deduplicated items and their embeddings
            unique_items = list(items_by_id.values())
            unique_embeddings = [embeddings_by_id[str(safe_get(item, "id"))] for item in unique_items]

            if len(unique_items) < len(content_items):
                logger.warning(f"Deduplication: {len(content_items)} items reduced to {len(unique_items)} unique items")

            ids = []
            documents = []
            metadatas = []

            for item in unique_items:
                item_id = str(safe_get(item, "id"))

                # Build text from content fields like embedding service does
                text_parts = []
                title = safe_get(item, "title")
                if title:
                    text_parts.append(title)

                # Handle different content types dynamically
                if hasattr(item, 'description') and safe_get(item, "description"):
                    text_parts.append(safe_get(item, "description"))
                if hasattr(item, 'content') and safe_get(item, "content"):
                    text_parts.append(safe_get(item, "content"))
                if hasattr(item, 'comments') and safe_get(item, "comments"):
                    text_parts.extend(safe_get(item, "comments"))
                if hasattr(item, 'extracted_text') and safe_get(item, "extracted_text"):
                    text_parts.append(safe_get(item, "extracted_text"))

                item_text = " ".join(text_parts) if text_parts else title or "No content"

                ids.append(item_id)
                documents.append(item_text)

                # Process metadata - handle both dict and object
                if isinstance(item, dict):
                    metadata = item.get("metadata", {}).copy()
                else:
                    # For Pydantic models, try to get metadata attribute or create from model fields
                    metadata = getattr(item, "metadata", {})
                    if not metadata:
                        # If no metadata attribute, create metadata from model attributes
                        metadata = {}
                        if hasattr(item, "source_name") and item.source_name:
                            metadata["source_name"] = item.source_name
                        if hasattr(item, "content_type") and item.content_type:
                            metadata["content_type"] = item.content_type
                        if hasattr(item, "title") and item.title:
                            metadata["title"] = item.title
                        if hasattr(item, "organization") and item.organization:
                            metadata["organization"] = item.organization
                        if hasattr(item, "project") and item.project:
                            metadata["project"] = item.project

                # Ensure mandatory fields from metadata are preserved
                # These come from the connector's get_metadata() and enrichment in ingestion
                if "title" not in metadata or not metadata["title"]:
                    metadata["title"] = safe_get(item, "title", "")
                if "content" not in metadata or not metadata["content"]:
                    metadata["content"] = safe_get(item, "content", "")
                if "source_reference" not in metadata or not metadata["source_reference"]:
                    metadata["source_reference"] = safe_get(item, "source_reference", "")

                # Add system metadata (without overwriting mandatory fields)
                system_metadata = {
                    "content_id": item_id,
                    "original_id": safe_get(item, "id"),
                    "concatenated_text": item_text,
                    "source_type": safe_get(item, "source_type", "unknown"),
                    "source_name": safe_get(item, "source_name", "unknown"),
                    "ingestion_timestamp": datetime.now().isoformat()
                }
                # Only add system metadata if it doesn't overwrite existing fields
                for key, value in system_metadata.items():
                    if key not in metadata:
                        metadata[key] = value

                # Ensure all metadata values are JSON serializable strings
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if value is None:
                        cleaned_metadata[key] = ""
                    elif isinstance(value, str | int | float | bool):
                        cleaned_metadata[key] = str(value)
                    elif isinstance(value, list | dict):
                        cleaned_metadata[key] = json.dumps(value)
                    else:
                        cleaned_metadata[key] = str(value)

                metadatas.append(cleaned_metadata)

            # Store in ChromaDB - use upsert to handle updates
            collection.upsert(
                ids=ids,
                embeddings=unique_embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Stored {len(unique_items)} items in collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to store content in collection '{collection_name}': {e}")
            raise

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection information in the expected format for the API endpoints"""
        stats = self.get_collection_stats()

        # Calculate total documents across all collections
        total_tickets = 0
        for collection_stats in stats.values():
            total_tickets += collection_stats.get("document_count", 0)

        return {
            "total_tickets": total_tickets,
            "collections": stats,
            "collection_count": len(stats)
        }
