"""
Service container for managing application dependencies.
Simplifies service initialization and dependency injection.
"""

import logging
from typing import Optional
from datetime import datetime

from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from services.summary_service import SummaryService
from services.rerank_service import RerankService
from services.bm25_service import BM25Service
from services.hybrid_search_service import HybridSearchService
from services.data_ingestion_service import DataIngestionService
from services.scheduler_service import SchedulerService
from models.schemas import HealthResponse
from config import settings

logger = logging.getLogger(__name__)

class ServiceContainer:
    """Simple service container to manage dependencies"""
    
    def __init__(self):
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_service: Optional[VectorService] = None
        self.summary_service: Optional[SummaryService] = None
        self.rerank_service: Optional[RerankService] = None
        self.bm25_service: Optional[BM25Service] = None
        self.hybrid_search_service: Optional[HybridSearchService] = None
        self.data_ingestion_service: Optional[DataIngestionService] = None
        self.scheduler_service: Optional[SchedulerService] = None
        self._initialized = False
    
    def initialize_services(self):
        """Initialize all services with proper dependency injection"""
        try:
            logger.info("Initializing OnCall Copilot services...")
            
            # Initialize core services first
            self.embedding_service = EmbeddingService()
            logger.info("Embedding service initialized")
            
            self.vector_service = VectorService()
            logger.info("Vector service initialized")
            
            self.summary_service = SummaryService()
            logger.info("Summary service initialized")

            # Initialize rerank service (always on) with chunking configuration
            self.rerank_service = RerankService(
                model_name=settings.RERANK_MODEL,
                device=settings.RERANK_DEVICE,
                max_length=settings.RERANK_MAX_LENGTH,
                max_chunk_size=settings.RERANK_MAX_CHUNK_SIZE
            )
            logger.info("Rerank service initialized with chunking support")

            # Initialize BM25 and hybrid search services
            self.bm25_service = BM25Service()
            # Try to load existing BM25 index from disk
            if self.bm25_service.load_index():
                logger.info(f"BM25 service initialized with {self.bm25_service.get_corpus_size()} documents loaded from disk")
            else:
                logger.info("BM25 service initialized (no existing index found)")

            self.hybrid_search_service = HybridSearchService()
            logger.info("Hybrid search service initialized")

            # Initialize services that depend on core services
            self.data_ingestion_service = DataIngestionService(
                embedding_service=self.embedding_service,
                vector_service=self.vector_service
            )
            logger.info("Data ingestion service initialized")

            # Attach vector service to embedding service for convenience methods
            self.embedding_service.vector_service = self.vector_service
            
            self.scheduler_service = SchedulerService(self.data_ingestion_service)
            
            # Set up scheduler callbacks for logging
            def log_scheduler_event(event_data):
                logger.info(f"Scheduler event: {event_data}")
            
            self.scheduler_service.set_callbacks(
                on_start=log_scheduler_event,
                on_complete=log_scheduler_event,
                on_error=log_scheduler_event
            )
            
            # Start the scheduler only if enabled
            if settings.SCHEDULER_ENABLED:
                self.scheduler_service.start()
                self.scheduler_service.ensure_task_running()
                logger.info("Scheduler service initialized and started")
            else:
                logger.info("Scheduler disabled by configuration; not starting scheduler service")
            
            self._initialized = True
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    def shutdown_services(self):
        """Cleanup services on shutdown"""
        try:
            logger.info("Shutting down OnCall Copilot services...")
            
            # Stop the scheduler
            if self.scheduler_service:
                self.scheduler_service.stop()
                logger.info("Scheduler service stopped")
            
            logger.info("All services shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_health_status(self) -> HealthResponse:
        """Get comprehensive health status for all services"""
        try:
            # Get collection info
            collection_info = self.vector_service.get_collection_info() if self.vector_service else {"initialized": False, "total_tickets": 0}
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                model_loaded=self.embedding_service.is_loaded() if self.embedding_service else False,
                chroma_connected=self.vector_service.is_initialized() if self.vector_service else False,
                azure_openai_configured=self.summary_service.is_configured() if self.summary_service else False,
                total_tickets=collection_info.get('total_tickets', 0)
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now(),
                model_loaded=False,
                chroma_connected=False,
                azure_openai_configured=False,
                total_tickets=0
            )
    
    def is_initialized(self) -> bool:
        """Check if all services are initialized"""
        return self._initialized and all([
            self.embedding_service,
            self.vector_service,
            self.summary_service,
            self.rerank_service,
            self.bm25_service,
            self.hybrid_search_service,
            self.data_ingestion_service,
            self.scheduler_service
        ])
    
    def get_service(self, service_name: str):
        """Get a specific service by name"""
        services = {
            'embedding': self.embedding_service,
            'vector': self.vector_service,
            'summary': self.summary_service,
            'rerank': self.rerank_service,
            'bm25': self.bm25_service,
            'hybrid_search': self.hybrid_search_service,
            'data_ingestion': self.data_ingestion_service,
            'scheduler': self.scheduler_service
        }
        return services.get(service_name) 