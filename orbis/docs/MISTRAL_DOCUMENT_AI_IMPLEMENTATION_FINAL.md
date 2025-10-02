# Mistral Document AI Integration - Final Implementation Guide

## Final Requirements

Based on latest clarifications:
1. ✅ **Process all image types** (markdown, HTML, attachments)
2. ✅ **Use document annotation** (not custom summaries)
3. ✅ **Synchronous processing** (process images before text summarization)
4. ✅ **Store image names + hash** for reference and deduplication

## Revised Architecture

### Processing Flow
```
Wiki Content Ingestion → Image Processing & Annotation → Enhanced Batch Summarization
```

**Key Change**: Back to synchronous processing - images must be processed and their annotations included BEFORE the batch summarization step.

## Implementation Components

### 1. Updated Database Schema

**New Table**: `image_processing_cache`

```sql
CREATE TABLE image_processing_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash VARCHAR(64) NOT NULL UNIQUE,
    image_name VARCHAR(255) NOT NULL,  -- For reference/debugging
    image_source VARCHAR(500) NOT NULL,
    document_annotation TEXT NOT NULL,  -- JSON from Mistral Document AI
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Mistral Document AI Service

```python
class MistralDocumentAIService:
    """Service for processing images using Mistral Document AI on Azure"""
    
    def __init__(self):
        self.endpoint = settings.AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT
        self.api_key = settings.AZURE_MISTRAL_DOCUMENT_AI_KEY
        self.model = "mistral-document-ai-2505"
        
    async def process_images(self, image_references: list[dict]) -> list[dict]:
        """Process multiple images and return document annotations"""
        
        if not self.is_configured():
            return []
            
        annotations = []
        for image_ref in image_references:
            try:
                annotation = await self._process_single_image(image_ref)
                if annotation:
                    annotations.append(annotation)
            except Exception as e:
                logger.error(f"Failed to process image {image_ref.get('src', 'unknown')}: {e}")
                continue
                
        return annotations
        
    async def _process_single_image(self, image_ref: dict) -> dict | None:
        """Process single image with hash-based deduplication"""
        
        # Download image and generate hash
        image_data = await self._fetch_image_content(image_ref['src'])
        if not image_data:
            return None
            
        image_hash = self._hash_image_content(image_data)
        image_name = self._extract_image_name(image_ref['src'])
        
        # Check for existing annotation
        cached_annotation = await self._get_cached_annotation(image_hash)
        if cached_annotation:
            logger.debug(f"Using cached annotation for {image_name}")
            return {
                'image_name': image_name,
                'image_hash': image_hash,
                'annotation': cached_annotation
            }
        
        # Process with Mistral Document AI
        annotation = await self._call_mistral_document_ai(image_data)
        if annotation:
            # Cache the result
            await self._cache_annotation(image_hash, image_name, image_ref['src'], annotation)
            
        return {
            'image_name': image_name,
            'image_hash': image_hash,
            'annotation': annotation
        } if annotation else None
        
    async def _call_mistral_document_ai(self, image_data: bytes) -> dict | None:
        """Call Mistral Document AI for document annotation"""
        
        import base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": f"data:image/png;base64,{encoded_image}"
            },
            # Let Mistral decide the annotation structure - no custom schema
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('document_annotation', {})
                else:
                    logger.error(f"Mistral API failed: {response.status} - {await response.text()}")
                    return None
                    
    def _hash_image_content(self, image_data: bytes) -> str:
        """Generate hash for deduplication lookup"""
        import xxhash
        return xxhash.xxh64(image_data).hexdigest()
        
    def _extract_image_name(self, image_src: str) -> str:
        """Extract readable name from image source for reference"""
        from urllib.parse import unquote
        
        if image_src.startswith('[[attachment:') and image_src.endswith(']]'):
            # Azure DevOps attachment: [[attachment:filename.png]]
            return image_src[13:-2]  # Remove [[attachment: and ]]
        elif '/' in image_src:
            # URL path: extract filename
            return unquote(image_src.split('/')[-1])
        else:
            # Direct filename or other format
            return unquote(image_src)
            
    async def _fetch_image_content(self, image_src: str) -> bytes | None:
        """Download image from Azure DevOps wiki"""
        
        if image_src.startswith('[[attachment:') and image_src.endswith(']]'):
            # Azure DevOps attachment: [[attachment:filename.png]]
            attachment_name = image_src[13:-2]  # Remove [[attachment: and ]]
            return await self._fetch_wiki_attachment(attachment_name)
        
        elif image_src.startswith('http'):
            # Direct URL - Azure DevOps or external
            return await self._fetch_url_content(image_src)
        
        else:
            # Relative path - resolve against wiki base URL
            return await self._fetch_relative_image(image_src)
            
    async def _fetch_wiki_attachment(self, attachment_name: str) -> bytes | None:
        """Fetch Azure DevOps wiki attachment by name"""
        
        # Construct attachment API URL
        attachment_url = f"{self.base_url}/_apis/wiki/wikis/{self.wiki_id}/attachments"
        params = {
            "name": attachment_name,
            "api-version": "7.1-preview.1"
        }
        
        headers = await self._get_auth_headers()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment_url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to fetch attachment {attachment_name}: {response.status}")
                    return None
                    
    async def _fetch_url_content(self, url: str) -> bytes | None:
        """Fetch image from direct URL"""
        
        # Use authentication for Azure DevOps URLs
        headers = {}
        if 'dev.azure.com' in url:
            headers = await self._get_auth_headers()
            
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        if content_type.startswith('image/'):
                            return await response.read()
                        else:
                            logger.warning(f"URL {url} returned non-image content-type: {content_type}")
                            return None
                    else:
                        logger.error(f"Failed to fetch image URL {url}: {response.status}")
                        return None
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching image URL: {url}")
                return None
                
    async def _fetch_relative_image(self, relative_path: str) -> bytes | None:
        """Fetch image using relative path from wiki repository"""
        
        # For code wikis, construct Git API URL to fetch file content
        if hasattr(self, 'repository_id') and self.repository_id:
            file_url = f"https://dev.azure.com/{self.organization}/{self.project}/_apis/git/repositories/{self.repository_id}/items"
            params = {
                "path": relative_path,
                "api-version": "7.1",
                "includeContent": "true"
            }
            
            headers = await self._get_auth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(file_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        # Git API returns base64 encoded content
                        result = await response.json()
                        if result.get('content'):
                            import base64
                            return base64.b64decode(result['content'])
                    else:
                        logger.error(f"Failed to fetch relative image {relative_path}: {response.status}")
                        return None
        
        logger.warning(f"Cannot resolve relative path {relative_path} - no repository context")
        return None
        
    async def _get_cached_annotation(self, image_hash: str) -> dict | None:
        """Get cached annotation by hash"""
        from app.db.session import get_db_session
        from app.db.models import ImageProcessingCache  # New model
        
        with get_db_session() as db:
            cache_entry = db.query(ImageProcessingCache).filter(
                ImageProcessingCache.image_hash == image_hash
            ).first()
            
            if cache_entry:
                import json
                return json.loads(cache_entry.document_annotation)
        return None
        
    async def _cache_annotation(self, image_hash: str, image_name: str, 
                               image_source: str, annotation: dict) -> None:
        """Cache annotation by hash"""
        from app.db.session import get_db_session
        from app.db.models import ImageProcessingCache
        import json
        
        with get_db_session() as db:
            cache_entry = ImageProcessingCache(
                image_hash=image_hash,
                image_name=image_name,
                image_source=image_source,
                document_annotation=json.dumps(annotation)
            )
            db.add(cache_entry)
            db.commit()
            
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key)
```

### 3. Wiki Summarization Integration

**In `wiki_summarization.py:276` (`_aggregate_and_summarize_pages`)**:

```python
async def _aggregate_and_summarize_pages(self, wiki_pages: list[dict[str, str]], wiki_name: str) -> str | None:
    try:
        # NEW: Process images synchronously before batch summarization
        enhanced_pages = await self._enhance_pages_with_image_annotations(wiki_pages)
        
        # Continue with existing batching logic using enhanced pages
        page_batches = self._group_pages_into_batches(enhanced_pages)
        
        if len(page_batches) == 1:
            return await self._summarize_page_batch(page_batches[0], wiki_name, is_final=True)
            
        # Multiple batches logic continues...
        
async def _enhance_pages_with_image_annotations(self, wiki_pages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Enhance wiki pages with Mistral Document AI image annotations"""
    
    if not hasattr(self, 'mistral_service'):
        from core.services.mistral_document_ai_service import MistralDocumentAIService
        self.mistral_service = MistralDocumentAIService()
    
    if not self.mistral_service.is_configured():
        logger.info("Mistral Document AI not configured - skipping image processing")
        return wiki_pages
    
    enhanced_pages = []
    total_images_processed = 0
    
    for page in wiki_pages:
        image_refs = page.get('image_references', [])
        if not image_refs:
            enhanced_pages.append(page)
            continue
            
        logger.info(f"Processing {len(image_refs)} images for page: {page.get('title', 'Unknown')}")
        
        # Process images for this page
        annotations = await self.mistral_service.process_images(image_refs)
        total_images_processed += len(annotations)
        
        # Merge annotations into page content
        enhanced_content = self._merge_content_with_annotations(page['content'], annotations)
        enhanced_page = {**page, 'content': enhanced_content}
        enhanced_pages.append(enhanced_page)
    
    logger.info(f"Enhanced {len(enhanced_pages)} pages with {total_images_processed} image annotations")
    return enhanced_pages

def _merge_content_with_annotations(self, page_content: str, annotations: list[dict]) -> str:
    """Merge page text content with image annotations"""
    
    if not annotations:
        return page_content
    
    # Append image annotations section to page content
    annotations_section = "\n\n### Image Content Analysis\n"
    
    for annotation_data in annotations:
        image_name = annotation_data.get('image_name', 'Unknown')
        annotation = annotation_data.get('annotation', {})
        
        annotations_section += f"\n**Image: {image_name}**\n"
        
        # Format annotation content (structure depends on Mistral's output)
        if isinstance(annotation, dict):
            for key, value in annotation.items():
                if value:  # Only include non-empty values
                    annotations_section += f"- {key}: {value}\n"
        elif isinstance(annotation, str):
            annotations_section += f"- Content: {annotation}\n"
    
    return page_content + annotations_section
```

### 4. Database Model

**New model in `app/db/models.py`**:

```python
class ImageProcessingCache(Base):
    __tablename__ = "image_processing_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_hash = Column(String(64), unique=True, nullable=False)
    image_name = Column(String(255), nullable=False)
    image_source = Column(String(500), nullable=False)  
    document_annotation = Column(Text, nullable=False)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 5. Configuration

**Settings** (`config/settings.py`):

```python
# Mistral Document AI Configuration
AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT: str = ""
AZURE_MISTRAL_DOCUMENT_AI_KEY: str = ""
MISTRAL_DOCUMENT_AI_ENABLED: bool = False

# Image processing settings
IMAGE_PROCESSING_TIMEOUT_SECONDS: int = 30
MAX_IMAGE_SIZE_MB: int = 30  # Mistral limit
MAX_PAGES_PER_IMAGE: int = 30  # Mistral limit
```

## Key Changes from Previous Versions

### 1. **Synchronous Processing**
- Images are processed **before** batch summarization begins
- Page content is enhanced with image annotations first
- Text summarization includes the image content

### 2. **Document Annotation Focus**  
- No custom prompt schema - let Mistral decide annotation structure
- Store complete `document_annotation` response from Mistral
- Format annotations for inclusion in page content

### 3. **Image Reference Storage**
- Store `image_name` for human reference/debugging
- Store `image_hash` for deduplication  
- Store `image_source` for traceability

### 4. **Content Integration**
- Append "Image Content Analysis" section to each page
- Include image name and extracted annotation details
- LLM sees both text and image content together

## Processing Flow Example

```
Page: "Database Setup Guide"
Images: ["setup-wizard.png", "config-screen.png"]

1. Process setup-wizard.png → Document annotation: {text: "Configuration wizard showing database connection settings...", elements: [...]}

2. Process config-screen.png → Document annotation: {text: "Settings panel with authentication options...", tables: [...]}

3. Enhanced page content:
   "# Database Setup Guide
   Original markdown content...
   
   ### Image Content Analysis
   **Image: setup-wizard.png**
   - text: Configuration wizard showing database connection settings...
   - elements: [button: "Next", field: "Server Name"]
   
   **Image: config-screen.png**  
   - text: Settings panel with authentication options...
   - tables: [Authentication Methods table]"

4. Feed enhanced content to batch summarization
```

## Implementation Steps

### Phase 1: Core Service Development
1. Create `MistralDocumentAIService` with basic image processing
2. **Implement complete image download functionality** (currently missing):
   - Azure DevOps attachment API (`_fetch_wiki_attachment`)
   - Direct URL downloads with authentication (`_fetch_url_content`) 
   - Relative path resolution via Git API (`_fetch_relative_image`)
3. Implement image hashing and deduplication logic
4. Add database migration for cache table
5. Create configuration settings

### Phase 2: Integration  
1. Add `ImageProcessingCache` model to database schema
2. Modify `WikiSummarizationService` to call synchronous image processing
3. Update content merging logic to include image annotations
4. Enhance error handling and logging for image download failures
5. Add monitoring for image processing metrics

### Phase 3: Testing & Optimization
1. Test all image source types (attachments, URLs, relative paths)
2. Verify JPEG/PNG/PDF format support with Mistral Document AI
3. Test authentication with Azure DevOps image APIs
4. Performance monitoring and optimization
5. Handle edge cases (missing images, large files, timeouts)

## Critical Gap Identified

**⚠️ Missing Image Download Implementation**: The current wiki system only extracts image references but does not download actual image data. The implementation above provides complete image fetching logic for:

- **Azure DevOps Attachments**: `[[attachment:filename.jpg]]` → Azure DevOps Wiki Attachments API
- **Direct URLs**: `https://...` → HTTP download with authentication
- **Relative Paths**: `./images/screenshot.png` → Git API file retrieval

This approach ensures image content is fully integrated into the wiki summarization process while maintaining hash-based deduplication and proper reference tracking.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Revise implementation for synchronous processing with image names and document annotation", "status": "completed", "activeForm": "Revising implementation for synchronous processing with image names and document annotation"}]