import logging
from fastapi import APIRouter, HTTPException, Depends

from config import settings
from app.api.dependencies import get_data_ingestion_service, require_api_key
from models.schemas import (
    DataSourceCreateRequest,
    DataSourceUpdateRequest,
    DataSourceResponse,
    DataSourceListResponse,
)
from services.data_ingestion_service import DataIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["datasources"])


def _mask_pat(pat: str) -> str:
    return f"{pat[:4]}{'*' * (len(pat) - 8)}{pat[-4:]}" if pat and len(pat) > 8 else "***"

def _mask_client_id(client_id: str) -> str:
    return f"{client_id[:4]}{'*' * (len(client_id) - 8)}{client_id[-4:]}" if client_id and len(client_id) > 8 else "***"


@router.get("/datasources", response_model=DataSourceListResponse, dependencies=[Depends(require_api_key)])
async def list_data_sources(ingestion_service: DataIngestionService = Depends(get_data_ingestion_service)):
    try:
        data_sources = settings.get_data_sources()
        source_responses = []

        for source in data_sources:
            total_workitems = 0
            workitems = ingestion_service.get_work_items_for_source(source.name)
            total_workitems = len(workitems)

            # Handle masking based on auth type
            pat_masked = None
            client_id_masked = None
            if source.auth_type == "pat":
                pat_masked = _mask_pat(source.pat)
            elif source.auth_type == "oauth2":
                client_id_masked = _mask_client_id(source.client_id)

            source_responses.append(
                DataSourceResponse(
                    name=source.name,
                    organization=source.organization,
                    project=source.project,
                    auth_type=source.auth_type,
                    pat_masked=pat_masked,
                    client_id_masked=client_id_masked,
                    query_ids=source.query_ids,
                    fields=source.fields,
                    enabled=source.enabled,
                    total_workitems=total_workitems,
                )
            )

        return DataSourceListResponse(
            data_sources=source_responses,
            total_sources=len(source_responses),
            enabled_sources=sum(1 for s in source_responses if s.enabled),
        )
    except Exception as exc:
        logger.error(f"Failed to list data sources: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to list data sources: {str(exc)}")


@router.post("/datasources", response_model=DataSourceResponse, dependencies=[Depends(require_api_key)])
async def create_data_source(request: DataSourceCreateRequest):
    try:
        from services.data_source_service import DataSource

        # Validate request based on auth type
        if request.auth_type == "pat":
            if not request.pat:
                raise HTTPException(status_code=400, detail="PAT is required for PAT authentication")
        elif request.auth_type == "oauth2":
            if not all([request.client_id, request.client_secret, request.tenant_id]):
                raise HTTPException(status_code=400, detail="client_id, client_secret, and tenant_id are required for OAuth2 authentication")
        else:
            raise HTTPException(status_code=400, detail="auth_type must be 'pat' or 'oauth2'")

        data_source = DataSource(
            name=request.name,
            organization=request.organization,
            project=request.project,
            auth_type=request.auth_type,
            pat=request.pat,
            client_id=request.client_id,
            client_secret=request.client_secret,
            tenant_id=request.tenant_id,
            query_ids=request.query_ids,
            fields=request.fields,
            enabled=request.enabled,
        )

        settings.add_data_source(data_source)
        
        # Handle masking based on auth type
        pat_masked = None
        client_id_masked = None
        if request.auth_type == "pat":
            pat_masked = _mask_pat(request.pat)
        elif request.auth_type == "oauth2":
            client_id_masked = _mask_client_id(request.client_id)

        return DataSourceResponse(
            name=data_source.name,
            organization=data_source.organization,
            project=data_source.project,
            auth_type=data_source.auth_type,
            pat_masked=pat_masked,
            client_id_masked=client_id_masked,
            query_ids=data_source.query_ids,
            fields=data_source.fields,
            enabled=data_source.enabled,
            total_workitems=0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as exc:
        logger.error(f"Failed to create data source: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to create data source: {str(exc)}")


@router.put("/datasources/{source_name}", response_model=DataSourceResponse, dependencies=[Depends(require_api_key)])
async def update_data_source(source_name: str, request: DataSourceUpdateRequest, ingestion_service: DataIngestionService = Depends(get_data_ingestion_service)):
    try:

        data_sources = settings.load_data_sources()
        existing_source = None
        for source in data_sources:
            if source.name == source_name:
                existing_source = source
                break

        if not existing_source:
            raise HTTPException(status_code=404, detail=f"Data source '{source_name}' not found")

        if request.organization is not None:
            existing_source.organization = request.organization
        if request.project is not None:
            existing_source.project = request.project
        if request.auth_type is not None:
            existing_source.auth_type = request.auth_type
        if request.pat is not None:
            existing_source.pat = request.pat
        if request.client_id is not None:
            existing_source.client_id = request.client_id
        if request.client_secret is not None:
            existing_source.client_secret = request.client_secret
        if request.tenant_id is not None:
            existing_source.tenant_id = request.tenant_id
        if request.query_ids is not None:
            existing_source.query_ids = request.query_ids
        if request.fields is not None:
            existing_source.fields = request.fields
        if request.enabled is not None:
            existing_source.enabled = request.enabled

        settings.update_data_source(existing_source)
        
        # Handle masking based on auth type
        pat_masked = None
        client_id_masked = None
        if existing_source.auth_type == "pat":
            pat_masked = _mask_pat(existing_source.pat)
        elif existing_source.auth_type == "oauth2":
            client_id_masked = _mask_client_id(existing_source.client_id)

        workitems = ingestion_service.get_work_items_for_source(source_name)
        total_workitems = len(workitems)

        return DataSourceResponse(
            name=existing_source.name,
            organization=existing_source.organization,
            project=existing_source.project,
            auth_type=existing_source.auth_type,
            pat_masked=pat_masked,
            client_id_masked=client_id_masked,
            query_ids=existing_source.query_ids,
            fields=existing_source.fields,
            enabled=existing_source.enabled,
            total_workitems=total_workitems,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to update data source: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to update data source: {str(exc)}")


@router.delete("/datasources/{source_name}", dependencies=[Depends(require_api_key)])
async def delete_data_source(source_name: str):
    try:
        settings.remove_data_source(source_name)
        return {"message": f"Data source '{source_name}' deleted successfully"}
    except Exception as exc:
        logger.error(f"Failed to delete data source: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data source: {str(exc)}")
