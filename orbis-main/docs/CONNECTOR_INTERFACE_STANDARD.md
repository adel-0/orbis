# Connector Interface Standard

Defines the standard interface for connectors compatible with the configuration-driven architecture.

## Connector Interface Requirements

All connectors must implement the following standard interface to be compatible with the generic ingestion system:

### Required Methods

```python
class ConnectorInterface:
    def __init__(self):
        """Initialize connector (usually lightweight, no external connections)"""
        pass
        
    async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
        """
        Fetch data from external system using provided configuration.
        
        Args:
            config: Configuration dictionary containing all connection/query parameters
            incremental: Whether to fetch only new/updated items since last sync
            
        Returns:
            List of raw data items from external system
        """
        pass
        
    def get_content_id(self, item: dict) -> str:
        """
        Extract unique identifier from a data item.
        
        Args:
            item: Raw data item from external system
            
        Returns:
            Unique string identifier for the item
        """
        pass
```

### Optional Methods

```python
    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, ""
        
    def get_last_modified(self, item: dict) -> datetime | None:
        """
        Extract last modified timestamp from data item.
        
        Args:
            item: Raw data item from external system
            
        Returns:
            Last modified datetime or None if not available
        """
        return None
```

## Standard Methods

### `fetch_data()` Method

This is the core method that the generic ingestion system calls to retrieve data:

```python
async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
    """
    Fetch data from external system.
    
    The config parameter contains ALL configuration needed:
    - Connection details (URLs, credentials)
    - Query parameters (filters, limits)
    - Authentication settings
    - Any connector-specific settings
    """
    # 1. Extract connection parameters from config
    server_url = config['server_url']
    credentials = config['credentials']
    
    # 2. Create/configure external client
    client = ExternalSystemClient(server_url, credentials)
    
    # 3. Build query from config parameters
    query = self._build_query(config.get('query_config', {}))
    
    # 4. Fetch data
    raw_data = await client.fetch(query)
    
    # 5. Return raw data items as list of dicts
    return raw_data
```

### `get_content_id()` Method

Extracts a unique identifier from each data item:

```python
def get_content_id(self, item: dict) -> str:
    """
    Extract unique ID that will be used as external_id in the Content table.
    This ID should be stable across fetches of the same item.
    """
    # For work items: return work item ID
    return str(item['id'])
    
    # For wiki pages: return path or page ID
    return item['path']
    
    # For JIRA issues: return issue key
    return item['key']
```

## Configuration Handling

### Configuration Structure

Connectors should expect a configuration dictionary with these standard sections:

```python
config = {
    # Connection settings
    "connection": {
        "server_url": "https://api.example.com",
        "credentials": {...}
    },
    
    # Query/filter settings  
    "query_config": {
        "filters": {...},
        "limits": {...}
    },
    
    # Connector-specific settings
    "options": {
        "batch_size": 100,
        "timeout": 30
    }
}
```

### Authentication Patterns

Support common authentication methods:

```python
def _create_client(self, config):
    auth_type = config.get('auth_type', 'token')
    
    if auth_type == 'oauth2':
        return Client(
            config['server_url'],
            client_id=config['client_id'],
            client_secret=config['client_secret']
        )
    elif auth_type == 'pat':
        return Client(
            config['server_url'], 
            token=config['pat']
        )
    elif auth_type == 'basic':
        return Client(
            config['server_url'],
            username=config['username'],
            password=config['password']  
        )
```

## Current Connector Analysis

### Azure DevOps Work Item Service

**Location**: `infrastructure/connectors/azure_devops/work_item_service.py`

**Interface Compliance**: ✅ **COMPLIANT**

```python
class WorkItemService:
    def __init__(self, db: Session | None = None):  # ✅ Lightweight init
        
    async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:  # ✅ Standard interface
        # ✅ Handles both oauth2 and PAT authentication
        # ✅ Uses config parameters to create Azure DevOps client
        # ✅ Returns list of raw work item dictionaries
        
    def get_content_id(self, work_item: dict) -> str:  # ✅ Standard interface
        return str(work_item.get('id', ''))
```

**Configuration Support**: ✅ **FULL YAML SUPPORT**
- Accepts generic config dictionary
- Supports multiple auth types (oauth2, PAT)
- Handles all connection parameters from config
- No hardcoded values

### Azure DevOps Wiki Service  

**Location**: `infrastructure/connectors/azure_devops/wiki_service.py`

**Interface Compliance**: ✅ **COMPLIANT**

```python
class WikiService:
    def __init__(self):  # ✅ Lightweight init
        
    async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:  # ✅ Standard interface
        # ✅ Handles both oauth2 and PAT authentication  
        # ✅ Uses config parameters to create wiki client
        # ✅ Returns list of raw wiki page dictionaries
        
    def get_content_id(self, wiki_item: dict) -> str:  # ✅ Standard interface
        return wiki_item.get('path', '')
```

**Configuration Support**: ✅ **FULL YAML SUPPORT**
- Accepts generic config dictionary
- Supports multiple auth types (oauth2, PAT)  
- Handles wiki-specific parameters (included_wikis, excluded_wikis)
- No hardcoded values

### Summary

Both existing connectors fully comply with the standard interface, support YAML configuration, handle authentication from config parameters, and return standardized data formats.

## Connector Template

Use this template when creating new connectors:

```python
"""
{SYSTEM_NAME} Connector for OnCall Copilot
Implements standard connector interface for {DATA_TYPE} data.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class {ConnectorName}Service:
    """
    {SYSTEM_NAME} connector implementing standard interface.
    
    Fetches {DATA_TYPE} from {SYSTEM_NAME} using configuration-driven approach.
    """

    def __init__(self):
        """Lightweight initialization - no external connections."""
        self.client = None

    async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
        """
        Fetch {DATA_TYPE} from {SYSTEM_NAME} using provided configuration.
        
        Expected config structure:
        {
            "connection": {
                "server_url": "https://your-system.com",
                "auth_type": "token|oauth2|basic",
                "credentials": {...}
            },
            "query_config": {
                "filters": {...},
                "project": "...",
                "query": "..."
            },
            "options": {
                "batch_size": 100,
                "max_items": 1000
            }
        }
        
        Args:
            config: All configuration parameters as dictionary
            incremental: Whether to fetch only new/updated items
            
        Returns:
            List of raw {DATA_TYPE} dictionaries from {SYSTEM_NAME}
        """
        try:
            # 1. Validate required config
            self._validate_required_config(config)
            
            # 2. Create client from config
            client = self._create_client(config)
            
            # 3. Build query from config
            query = self._build_query(config.get('query_config', {}))
            
            # 4. Fetch data with error handling
            logger.info(f"Fetching {DATA_TYPE} from {SYSTEM_NAME}...")
            raw_data = await client.fetch_items(query)
            
            logger.info(f"Retrieved {len(raw_data)} {DATA_TYPE} from {SYSTEM_NAME}")
            return raw_data
            
        except Exception as e:
            logger.error(f"Failed to fetch {DATA_TYPE} from {SYSTEM_NAME}: {e}")
            raise

    def get_content_id(self, item: dict) -> str:
        """
        Extract unique identifier from {DATA_TYPE} item.
        
        Args:
            item: Raw {DATA_TYPE} dictionary from {SYSTEM_NAME}
            
        Returns:
            Unique string identifier for the item
        """
        # Return the primary identifier field for your data type
        return str(item.get('id', item.get('key', item.get('path', ''))))

    def get_last_modified(self, item: dict) -> datetime | None:
        """
        Extract last modified timestamp from {DATA_TYPE} item.
        
        Args:
            item: Raw {DATA_TYPE} dictionary from {SYSTEM_NAME}
            
        Returns:
            Last modified datetime or None if not available
        """
        # Extract timestamp field if available
        modified_str = item.get('updated', item.get('modified', item.get('lastModified')))
        if modified_str:
            try:
                return datetime.fromisoformat(modified_str.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Could not parse timestamp: {modified_str}")
        return None

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self._validate_required_config(config)
            return True, ""
        except ValueError as e:
            return False, str(e)

    def _validate_required_config(self, config: dict[str, Any]):
        """Validate that all required configuration is present."""
        connection = config.get('connection', {})
        
        required_fields = ['server_url']  # Add your required fields
        for field in required_fields:
            if not connection.get(field):
                raise ValueError(f"Missing required connection field: {field}")
        
        # Validate auth configuration
        auth_type = connection.get('auth_type', 'token')
        if auth_type == 'oauth2':
            if not all(connection.get(f) for f in ['client_id', 'client_secret']):
                raise ValueError("OAuth2 requires client_id and client_secret")
        elif auth_type == 'token':
            if not connection.get('token'):
                raise ValueError("Token auth requires token field")

    def _create_client(self, config: dict[str, Any]):
        """Create and configure client from config parameters."""
        connection = config.get('connection', {})
        
        # Import your client class
        from .your_system_client import YourSystemClient
        
        auth_type = connection.get('auth_type', 'token')
        
        if auth_type == 'oauth2':
            return YourSystemClient(
                connection['server_url'],
                client_id=connection['client_id'],
                client_secret=connection['client_secret']
            )
        elif auth_type == 'token':
            return YourSystemClient(
                connection['server_url'],
                token=connection['token']
            )
        else:
            raise ValueError(f"Unsupported auth_type: {auth_type}")

    def _build_query(self, query_config: dict[str, Any]) -> str:
        """Build system-specific query from config parameters."""
        # Build query string/object for your system
        # This is system-specific logic
        
        filters = query_config.get('filters', {})
        project = query_config.get('project')
        custom_query = query_config.get('query')
        
        if custom_query:
            return custom_query
        
        # Build default query from filters
        query_parts = []
        if project:
            query_parts.append(f"project = {project}")
        
        return " AND ".join(query_parts)
```

## Implementation Guidelines

### 1. Configuration-Driven Design

- **Never hardcode** connection details, credentials, or queries
- **Accept all parameters** through the config dictionary
- **Support multiple auth methods** commonly used by your target system
- **Provide reasonable defaults** for optional parameters

### 2. Error Handling

```python
async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
    try:
        # Implementation
        pass
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        raise ValueError(f"Authentication failed - check credentials: {e}")
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        raise ValueError(f"Could not connect to {SYSTEM_NAME}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 3. Logging

```python
logger.info(f"Connecting to {SYSTEM_NAME} at {server_url}")
logger.info(f"Fetching {DATA_TYPE} with query: {query}")
logger.info(f"Retrieved {len(items)} {DATA_TYPE}")
logger.warning(f"Skipping invalid item: {item_id}")
```

### 4. Data Transformation

Keep the `fetch_data()` method focused on data retrieval. Return raw data as-is:

```python
async def fetch_data(self, config: dict[str, Any], incremental: bool = True) -> list[dict[str, Any]]:
    # Good: Return raw data
    return await client.get_issues(query)
    
    # Bad: Don't transform data here
    # return [self.transform_issue(issue) for issue in issues]
```

Data transformation happens in the generic ingestion service, not in connectors.

### 5. Testing

Create a simple test configuration for your connector:

```python
test_config = {
    "connection": {
        "server_url": "https://test-system.com",
        "auth_type": "token", 
        "token": "test-token"
    },
    "query_config": {
        "project": "TEST",
        "filters": {"status": "open"}
    },
    "options": {
        "max_items": 10
    }
}

# Test the connector
connector = YourConnectorService()
items = await connector.fetch_data(test_config)
assert len(items) > 0
assert connector.get_content_id(items[0]) is not None
```

---

This standard ensures all connectors work seamlessly with the configuration-driven ingestion system and can be easily configured through YAML templates.