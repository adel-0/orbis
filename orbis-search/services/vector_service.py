import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import os
from config import settings
from models.schemas import Ticket, SearchRequest
from orbis_core.utils.similarity import normalize_cosine_similarity

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self.db_path = settings.CHROMA_DB_PATH
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure the directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized successfully. Collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _recreate_collection(self):
        """Drop and recreate the collection to ensure a clean state."""
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        try:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                # Ignore if collection does not exist
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"Could not recreate collection: {e}")
    
    def _concatenate_ticket_text(self, ticket: Ticket) -> str:
        """Concatenate ticket text (title + description + comments)"""
        parts = [ticket.title]
        if ticket.description:
            parts.append(ticket.description)
        if ticket.comments:
            parts.extend(ticket.comments)
        return " ".join(parts)
    
    def _prepare_tickets_for_embedding(self, tickets: List[Ticket]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Prepare tickets for embedding by concatenating text and creating metadata"""
        texts = []
        metadatas = []
        
        for ticket in tickets:
            concatenated_text = self._concatenate_ticket_text(ticket)
            texts.append(concatenated_text)
            
            # Create metadata
            metadata = {
                "id": ticket.id,
                "title": ticket.title,
                "description": ticket.description or "",
                "comments": json.dumps(ticket.comments),
                "concatenated_text": concatenated_text,
                "source_name": ticket.source_name or "",
                "organization": ticket.organization or "",
                "project": ticket.project or "",
                "area_path": ticket.area_path or "",
                "iteration_path": ticket.iteration_path or "",
                "created_date": ticket.created_date.isoformat() if ticket.created_date else ""
            }
            metadatas.append(metadata)
        
        return texts, metadatas

    def store_embeddings(self, tickets: List[Ticket], embeddings: List[List[float]], clear_existing: bool = False) -> bool:
        """Store ticket embeddings in ChromaDB"""
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")
        
        if len(tickets) != len(embeddings):
            raise ValueError("Number of tickets and embeddings must match")
        
        try:
            # Prepare data
            texts, metadatas = self._prepare_tickets_for_embedding(tickets)
            ids = [f"ticket_{ticket.id}" for ticket in tickets]
            
            # Clear existing collection only if requested
            if clear_existing:
                self._recreate_collection()
            
            # Add embeddings directly (they're already in small batches)
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            # Demote per-batch log to debug to reduce noise; aggregated progress is logged elsewhere
            logger.debug(f"Stored {len(tickets)} ticket embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 3, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar tickets using cosine similarity"""
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")
        
        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["metadatas", "documents", "distances"]
            }
            if where:
                query_params["where"] = where
            
            results = self.collection.query(**query_params)
            
            # Process results
            processed_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    document = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    # Convert cosine distance to similarity score using orbis-core utility
                    similarity_score = normalize_cosine_similarity(distance, clamp=True)
                    
                    # Parse comments from JSON
                    comments = json.loads(metadata.get('comments', '[]'))

                    # Parse created_date from ISO format
                    from datetime import datetime
                    created_date = None
                    if metadata.get('created_date'):
                        try:
                            created_date = datetime.fromisoformat(metadata['created_date'])
                        except Exception:
                            pass

                    ticket = Ticket(
                        id=metadata['id'],
                        title=metadata['title'],
                        description=metadata.get('description', ''),
                        comments=comments,
                        source_name=metadata.get('source_name'),
                        organization=metadata.get('organization'),
                        project=metadata.get('project'),
                        area_path=metadata.get('area_path'),
                        iteration_path=metadata.get('iteration_path'),
                        created_date=created_date,
                        additional_fields={}  # Additional fields not stored in metadata for performance
                    )
                    
                    processed_results.append({
                        'ticket': ticket,
                        'similarity_score': similarity_score,
                        'concatenated_text': document
                    })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search similar tickets: {e}")
            raise
    
    def search_candidates(self, query_embedding: List[float], n_results: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for candidate results (no post-limit other than n_results).

        Mirrors search_similar but returns exactly up to n_results items for downstream reranking.
        """
        return self.search_similar(query_embedding, top_k=n_results, where=where)
    
    def filter_results(self, results: List[Dict[str, Any]], search_request: SearchRequest) -> List[Dict[str, Any]]:
        """Filter results based on search request filters (for BM25 results)"""
        if not results:
            return results

        filtered = []
        for result in results:
            ticket = result['ticket']

            # Apply filters (handle None values)
            if search_request.source_names and ticket.source_name not in search_request.source_names:
                continue
            if search_request.organizations and ticket.organization not in search_request.organizations:
                continue
            if search_request.projects and ticket.project not in search_request.projects:
                continue
            if search_request.area_path_prefix and (not ticket.area_path or not ticket.area_path.startswith(search_request.area_path_prefix)):
                continue
            if search_request.area_path_contains and (not ticket.area_path or search_request.area_path_contains not in ticket.area_path):
                continue
            if search_request.iteration_path_prefix and (not ticket.iteration_path or not ticket.iteration_path.startswith(search_request.iteration_path_prefix)):
                continue
            if search_request.iteration_path_contains and (not ticket.iteration_path or search_request.iteration_path_contains not in ticket.iteration_path):
                continue

            filtered.append(result)

        return filtered

    def build_where_clause(self, search_request: SearchRequest) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from search request filters"""
        filters = []
        
        # Source name filtering
        if search_request.source_names:
            if len(search_request.source_names) == 1:
                filters.append({"source_name": {"$eq": search_request.source_names[0]}})
            else:
                filters.append({"source_name": {"$in": search_request.source_names}})
        
        # Organization filtering
        if search_request.organizations:
            if len(search_request.organizations) == 1:
                filters.append({"organization": {"$eq": search_request.organizations[0]}})
            else:
                filters.append({"organization": {"$in": search_request.organizations}})
        
        # Project filtering
        if search_request.projects:
            if len(search_request.projects) == 1:
                filters.append({"project": {"$eq": search_request.projects[0]}})
            else:
                filters.append({"project": {"$in": search_request.projects}})
        
        # Area path filtering
        if search_request.area_path_prefix:
            # Use regex for prefix matching
            filters.append({"area_path": {"$regex": f"^{search_request.area_path_prefix}"}})
        elif search_request.area_path_contains:
            # Use regex for contains matching
            filters.append({"area_path": {"$regex": search_request.area_path_contains}})
        
        # Iteration path filtering
        if search_request.iteration_path_prefix:
            # Use regex for prefix matching
            filters.append({"iteration_path": {"$regex": f"^{search_request.iteration_path_prefix}"}})
        elif search_request.iteration_path_contains:
            # Use regex for contains matching
            filters.append({"iteration_path": {"$regex": search_request.iteration_path_contains}})
        
        # Combine filters with AND logic if multiple filters exist
        if not filters:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.collection:
            return {"initialized": False}
        
        try:
            count = self.collection.count()
            return {
                "initialized": True,
                "collection_name": self.collection_name,
                "total_tickets": count,
                "db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"initialized": False, "error": str(e)}
    
    def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        return self.collection is not None