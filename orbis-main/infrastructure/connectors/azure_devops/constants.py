"""
Azure DevOps specific constants.
These constants are specific to Azure DevOps connector implementation.
"""


# Azure DevOps API versions
API_VERSIONS: dict[str, str] = {
    'query': '6.1-preview.2',
    'workitem': '7.1-preview.3',
    'comments': '7.1-preview.3',
    'wiki': '7.1-preview.1',
    'git': '7.1',
    'reporting': '4.1'
}

# Azure DevOps specific processing settings
DEFAULT_BATCH_SIZE: int = 200  # Azure DevOps API maximum
DEFAULT_DB_BATCH_SIZE: int = 500  # Database batch commit size

# Core work item fields (Azure DevOps System fields)
CORE_WORKITEM_FIELDS: list[str] = [
    "System.Id", "System.Title", "System.State", "System.WorkItemType",
    "System.ChangedDate", "System.CreatedDate", "System.Description",
    "System.AreaPath", "System.Tags",
    "System.AssignedTo", "System.CreatedBy"
]

# Azure DevOps specific field mappings
AZDO_FIELD_MAPPINGS: dict[str, str] = {
    "id": "System.Id",
    "title": "System.Title",
    "state": "System.State",
    "work_item_type": "System.WorkItemType",
    "changed_date": "System.ChangedDate",
    "created_date": "System.CreatedDate",
    "description": "System.Description",
    "area_path": "System.AreaPath",
    "tags": "System.Tags",
    "assigned_to": "System.AssignedTo",
    "created_by": "System.CreatedBy"
}
