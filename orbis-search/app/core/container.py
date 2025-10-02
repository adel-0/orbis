import logging
from services.service_container import ServiceContainer

logger = logging.getLogger(__name__)


def create_container() -> ServiceContainer:
    container = ServiceContainer()
    container.initialize_services()
    return container


