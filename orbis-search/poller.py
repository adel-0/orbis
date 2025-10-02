r"""
Automated poller for Azure DevOps work items.

Behavior:
- Polls Azure DevOps for newly created/changed work items (since last run) using the Reporting API
- Processes ALL work items in the defined area path (no tag filtering)
- For each work item, searches for top 3 related tickets using the local Orbis Search API (/search)
- Posts a comment with the top 3 related tickets and hyperlinks (no summary generation)
- For existing work items, detects and updates existing AI-generated comments instead of creating new ones
- Includes AI attribution stamp with timestamp and version

Configuration via in-file constants (edit below):
- AZDO_ORG (required)
- AZDO_PROJECT (required)
- Authentication (choose one):
  - OAuth2 (recommended): Set USE_OAUTH2=True, CLIENT_ID, CLIENT_SECRET, TENANT_ID
  - PAT: Set USE_OAUTH2=False, AZDO_PAT (scopes: Work Items (Read & Write))
- ITERATION_PREFIX (optional)  e.g., "MyProject\Iteration 1"
- AREA_PREFIX (optional)       e.g., "MyProject\Area\SubArea"
- API_BASE_URL (default: "http://api:7887")
- POLL_SECONDS (default: 10)
- TOPK (default: 3)
- STATE_PATH (default: ".ado_related/state.json")

Run locally in your .venv:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements-local.txt
  python poller.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tomllib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from html import escape
from urllib.parse import quote
from pathlib import Path

import aiohttp

from services.azure_devops_client import AzureDevOpsClient
from models.schemas import Ticket


# -----------------------------
# Configuration (defaults with env overrides)
# -----------------------------
DEFAULT_AZDO_ORG: str = "HXGN-SI-ZUR"
DEFAULT_AZDO_PROJECT: str = "HxGN OnCall"
DEFAULT_AZDO_PAT: str = ""

# OAuth2 Configuration
DEFAULT_USE_OAUTH2: bool = True
DEFAULT_CLIENT_ID: str = "33ddc298-e780-41ea-85b5-451f6aa10cd9"
DEFAULT_CLIENT_SECRET: str = "REDACTED_CLIENT_SECRET"
DEFAULT_TENANT_ID: str = "1b16ab3e-b8f6-4fe3-9f3e-2db7fe549f6a"

DEFAULT_ITERATION_PREFIX: str = "HxGN OnCall"
DEFAULT_AREA_PREFIX: str = "HxGN OnCall\\SG\\SG NEZ Operations"

DEFAULT_POLL_SECONDS: int = 30
DEFAULT_TOPK: int = 3
DEFAULT_STATE_PATH: str = ".ado_related/state.json"

# Use local Orbis Search API /search to get related tickets and summary
# The poller will call your FastAPI (main.py) at the address below.
# Note: "api" is the Docker Compose service name
DEFAULT_USE_LOCAL_SEARCH: bool = True
DEFAULT_API_BASE_URL: str = "http://api:7887"
# If your API has API key protection enabled, set these accordingly
DEFAULT_API_KEY_ENABLED: bool = False
DEFAULT_API_KEY: str = ""

# Optional override for hyperlink organization/project used in rendered comments.
# In some setups, the organization/project used for API polling differs from the
# organization/project that should be used for web hyperlinks. If not set, the
# values below are used.
DEFAULT_LINK_ORG: str = "hexagon-si-gpc"
DEFAULT_LINK_PROJECT: str = "GPC Support"
DEFAULT_BOT_IDENTITY_NAME: str = "HxGN-SI-ZUR-orbis-search"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


AZDO_ORG: str = os.getenv("AZDO_ORG", DEFAULT_AZDO_ORG)
AZDO_PROJECT: str = os.getenv("AZDO_PROJECT", DEFAULT_AZDO_PROJECT)
AZDO_PAT: str = os.getenv("AZDO_PAT", DEFAULT_AZDO_PAT)

# OAuth2 Configuration
USE_OAUTH2: bool = _env_bool("USE_OAUTH2", DEFAULT_USE_OAUTH2)
CLIENT_ID: str = os.getenv("CLIENT_ID", DEFAULT_CLIENT_ID)
CLIENT_SECRET: str = os.getenv("CLIENT_SECRET", DEFAULT_CLIENT_SECRET)
TENANT_ID: str = os.getenv("TENANT_ID", DEFAULT_TENANT_ID)

ITERATION_PREFIX: str = os.getenv("ITERATION_PREFIX", DEFAULT_ITERATION_PREFIX)
AREA_PREFIX: str = os.getenv("AREA_PREFIX", DEFAULT_AREA_PREFIX)

POLL_SECONDS: int = _env_int("POLL_SECONDS", DEFAULT_POLL_SECONDS)
TOPK: int = _env_int("TOPK", DEFAULT_TOPK)
STATE_PATH: str = os.getenv("STATE_PATH", DEFAULT_STATE_PATH)

USE_LOCAL_SEARCH: bool = _env_bool("USE_LOCAL_SEARCH", DEFAULT_USE_LOCAL_SEARCH)
API_BASE_URL: str = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
API_KEY_ENABLED: bool = _env_bool("API_KEY_ENABLED", DEFAULT_API_KEY_ENABLED)
API_KEY: str = os.getenv("API_KEY", DEFAULT_API_KEY)

LINK_ORG: str = os.getenv("LINK_ORG", DEFAULT_LINK_ORG)
LINK_PROJECT: str = os.getenv("LINK_PROJECT", DEFAULT_LINK_PROJECT)
BOT_IDENTITY_NAME: str = os.getenv("BOT_IDENTITY_NAME", DEFAULT_BOT_IDENTITY_NAME)


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def get_version() -> str:
    """Get the version from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "unknown")
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


async def _auth_headers_oauth2(client_id: str, client_secret: str, tenant_id: str) -> Dict[str, str]:
    """Get OAuth2 authentication headers"""
    import msal
    
    oauth2_client = msal.ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=f"https://login.microsoftonline.com/{tenant_id}"
    )
    
    # The scope for Azure DevOps API
    scope = ["499b84ac-1321-427f-aa17-267ca6975798/.default"]
    
    result = oauth2_client.acquire_token_for_client(scopes=scope)
    
    if "access_token" not in result:
        error_description = result.get("error_description", result.get("error", "Unknown error"))
        raise Exception(f"Failed to acquire OAuth2 token: {error_description}")
    
    return {
        "Authorization": f"Bearer {result['access_token']}",
        "Content-Type": "application/json",
    }

def _auth_headers_pat(pat: str) -> Dict[str, str]:
    """Get PAT authentication headers"""
    token = base64.b64encode(f":{pat}".encode("ascii")).decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }


def _parse_ado_datetime(value: Any) -> Optional[datetime]:
    """Parse ADO ISO-ish datetime values, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _ado_work_item_link(org: str, project: str, workitem_id: Any) -> str:
    return f"https://dev.azure.com/{org}/{quote(project)}/_workitems/edit/{workitem_id}"


def _ado_work_item_api_url(org: str, project: str, workitem_id: int) -> str:
    project_path = quote(project)
    return f"https://dev.azure.com/{org}/{project_path}/_apis/wit/workItems/{workitem_id}?api-version=7.1"


def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_run": None}


def _save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


async def ado_search(
    session: aiohttp.ClientSession, *, org: str, project: str, query_text: str, area: str, topk: int
) -> List[Dict[str, Any]]:
    """Call Azure DevOps Work Item Search API and return results list."""
    project_path = quote(project)
    url = (
        f"https://almsearch.dev.azure.com/{org}/{project_path}/_apis/search/"
        f"workitemsearchresults?api-version=7.1-preview.1"
    )
    body: Dict[str, Any] = {
        "searchText": query_text,
        "top": topk,
        "filters": {"System.TeamProject": [project]},
    }
    if area:
        body["filters"]["System.AreaPath"] = [area]

    # Get appropriate auth headers based on configuration
    if USE_OAUTH2:
        headers = await _auth_headers_oauth2(CLIENT_ID, CLIENT_SECRET, TENANT_ID)
    else:
        headers = _auth_headers_pat(AZDO_PAT)

    async with session.post(url, json=body, headers=headers) as r:
        r.raise_for_status()
        data = await r.json()
        return data.get("results", [])


async def local_search(
    session: aiohttp.ClientSession, *, base_url: str, query_text: str, topk: int
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Call local FastAPI /search and return pseudo-ADO hits and summary.

    Returns a tuple of (hits_like_ado, summary). Each hit includes fields.system.id/title
    so that existing rendering can be reused.
    """
    url = f"{base_url.rstrip('/')}/search"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY_ENABLED and API_KEY:
        headers["X-API-Key"] = API_KEY
    body = {"query": query_text, "top_k": topk}
    log(f"Local search POST {url} (top_k={topk})")
    async with session.post(url, json=body, headers=headers) as r:
        status = r.status
        r.raise_for_status()
        data = await r.json()
        results = data.get("results", [])
        summary = data.get("summary")
        log(
            f"Local search response {status}: results={len(results)}, summary={'yes' if summary else 'no'}"
        )
        hits: List[Dict[str, Any]] = []
        for item in results:
            ticket = item.get("ticket", {})
            rid = ticket.get("id")
            rtitle = ticket.get("title", "")
            similarity_score = item.get("similarity_score")
            org_value = ticket.get("organization")
            project_value = ticket.get("project")
            hit: Dict[str, Any] = {"fields": {"system.id": rid, "system.title": rtitle}}
            if similarity_score is not None:
                hit["similarity_score"] = similarity_score
            if org_value:
                hit["organization"] = org_value
            if project_value:
                hit["project"] = project_value
            hits.append(hit)
        return hits, summary

def render_comment_html(*, org: str, project: str, new_title: str, hits: List[Dict[str, Any]]) -> str:
    """Render an HTML-formatted comment with top 3 related tickets and hyperlinks.
    
    Shows "Top 3 related tickets" with all tickets listed together with hyperlinks and relevancy scores.
    Always includes AI attribution stamp.
    """
    parts: List[str] = []
    
    if not hits:
        parts.append("<div>No clearly related prior tickets were found.</div>")
    else:
        # Limit to top 3 tickets
        top_hits = hits[:3]
        ticket_count = len(top_hits)
        
        parts.append(
            f"<div><strong>Top {ticket_count} related ticket{'' if ticket_count == 1 else 's'}:</strong></div>"
        )
        parts.append("<ul>")
        
        for r in top_hits:
            fields = r.get("fields", {})
            rid = fields.get("system.id") or fields.get("System.Id")
            rtitle = fields.get("system.title") or fields.get("System.Title") or ""
            if rid is None:
                continue
            r_org = r.get("organization") or org
            r_project = r.get("project") or project
            link = _ado_work_item_link(r_org, r_project, rid)
            score_suffix = ""
            try:
                score_val = r.get("similarity_score")
                score = float(score_val) if score_val is not None else None
                if score is not None:
                    score_suffix = f" — Relevancy: {score:.2f}"
            except Exception:
                pass
            parts.append(
                f"<li><a href=\"{link}\">#{escape(str(rid))} {escape(str(rtitle))}</a>{score_suffix}</li>"
            )
        parts.append("</ul>")
    
    # Add AI attribution footer (always included)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    version = get_version()
    footer = (
        f"<div style=\"margin-top:16px; padding-top:8px; border-top:1px solid #e0e0e0; "
        f"font-size:11px; color:#888; font-style:italic;\">"
        f"Bip Bop, I'm a robot • {timestamp} • Orbis Search v{version}"
        f"</div>"
    )
    parts.append(footer)
    
    return "".join(parts)


async def post_comment(
    session: aiohttp.ClientSession, *, org: str, project: str, workitem_id: int, text: str
) -> None:
    """Post a formatted comment by updating System.History with HTML.
    
    Using JSON Patch ensures Azure DevOps renders the content as rich text.
    """
    url = _ado_work_item_api_url(org, project, workitem_id)
    
    # Get appropriate auth headers based on configuration
    if USE_OAUTH2:
        headers = await _auth_headers_oauth2(CLIENT_ID, CLIENT_SECRET, TENANT_ID)
    else:
        headers = _auth_headers_pat(AZDO_PAT)
    
    headers["Content-Type"] = "application/json-patch+json"
    body = [
        {"op": "add", "path": "/fields/System.History", "value": text},
    ]
    async with session.patch(url, json=body, headers=headers) as r:
        r.raise_for_status()


async def get_workitem_comments_detailed(
    session: aiohttp.ClientSession, *, org: str, project: str, workitem_id: int
) -> List[Dict[str, Any]]:
    """Get detailed comment information including comment IDs for a work item."""
    project_path = quote(project)
    url = f"https://dev.azure.com/{org}/{project_path}/_apis/wit/workItems/{workitem_id}/comments?api-version=7.1-preview.3"
    
    # Get appropriate auth headers based on configuration
    if USE_OAUTH2:
        headers = await _auth_headers_oauth2(CLIENT_ID, CLIENT_SECRET, TENANT_ID)
    else:
        headers = _auth_headers_pat(AZDO_PAT)
    
    async with session.get(url, headers=headers) as r:
        r.raise_for_status()
        data = await r.json()
        return data.get("comments", [])


async def find_ai_comment(
    session: aiohttp.ClientSession, *, org: str, project: str, workitem_id: int
) -> Optional[Dict[str, Any]]:
    """Find existing AI-generated comment on a work item.
    
    Returns the comment dict with 'id' and 'text' fields if found, None otherwise.
    Looks for AI-generated comments by checking for the AI attribution stamp.
    Supports both old ("AI-generated response") and new ("Bip Bop, I'm a robot") formats.
    """
    try:
        comments = await get_workitem_comments_detailed(session, org=org, project=project, workitem_id=workitem_id)
        
        # Look for AI-generated comments by checking for the AI attribution stamp
        for comment in comments:
            comment_text = comment.get("text", "")
            # Check for our AI attribution stamp that's always added
            if ("AI-generated response" in comment_text or "Bip Bop, I'm a robot" in comment_text) and ("OnCall Copilot v" in comment_text or "Orbis Search v" in comment_text):
                log(f"Found existing AI comment on WI #{workitem_id} (ID: {comment.get('id')})")
                return comment
        
        log(f"No existing AI comment found on WI #{workitem_id}")
        return None
    except Exception as e:
        log(f"Failed to get comments for WI #{workitem_id}: {e}")
        return None


async def update_comment(
    session: aiohttp.ClientSession, *, org: str, project: str, workitem_id: int, comment_id: int, text: str
) -> None:
    """Update an existing comment on a work item."""
    project_path = quote(project)
    url = f"https://dev.azure.com/{org}/{project_path}/_apis/wit/workItems/{workitem_id}/comments/{comment_id}?format=html&api-version=7.1-preview.4"
    
    # Get appropriate auth headers based on configuration
    if USE_OAUTH2:
        headers = await _auth_headers_oauth2(CLIENT_ID, CLIENT_SECRET, TENANT_ID)
    else:
        headers = _auth_headers_pat(AZDO_PAT)
    
    headers["Content-Type"] = "application/json"
    body = {"text": text}
    
    async with session.patch(url, json=body, headers=headers) as r:
        r.raise_for_status()


def _match_filters(fields: Dict[str, Any]) -> bool:
    """Return True if work item matches iteration/area filters (if provided)."""
    if ITERATION_PREFIX:
        iteration = str(fields.get("System.IterationPath", ""))
        if not iteration.startswith(ITERATION_PREFIX):
            return False
    if AREA_PREFIX:
        area = str(fields.get("System.AreaPath", ""))
        if not area.startswith(AREA_PREFIX):
            return False
    return True


def _build_query_text(fields: Dict[str, Any], client: Any, comments: List[str]) -> str:
    """Build query text based on work item type with appropriate fields."""
    title = str(fields.get("System.Title", ""))
    description = str(fields.get("System.Description", "") or "")
    work_item_type = str(fields.get("System.WorkItemType", ""))
    
    # Clean HTML from description
    try:
        description = client.clean_html_string(description)
    except Exception:
        pass
    
    # Start with title and description
    parts = [title]
    if description:
        parts.append(description)
    
    # Add type-specific fields
    if work_item_type.lower() == "bug":
        # For Bugs: Title, Repo, Steps, Symptom, System Info, Fix
        repo_steps = str(fields.get("Microsoft.VSTS.TCM.ReproSteps", "") or "")
        symptom = str(fields.get("Microsoft.VSTS.Common.Symptom", "") or "")
        system_info = str(fields.get("Microsoft.VSTS.TCM.SystemInfo", "") or "")
        fix = str(fields.get("Microsoft.VSTS.Common.Fix", "") or "")
        
        # Clean HTML and add non-empty fields
        try:
            if repo_steps:
                repo_steps = client.clean_html_string(repo_steps)
                parts.append(f"Repro Steps: {repo_steps}")
            if symptom:
                symptom = client.clean_html_string(symptom)
                parts.append(f"Symptom: {symptom}")
            if system_info:
                system_info = client.clean_html_string(system_info)
                parts.append(f"System Info: {system_info}")
            if fix:
                fix = client.clean_html_string(fix)
                parts.append(f"Fix: {fix}")
        except Exception:
            pass
            
    elif work_item_type.lower() in ["requirement", "user story", "feature"]:
        # For Requirements: Title, Description, User Story, Acceptance Criteria
        user_story = str(fields.get("Microsoft.VSTS.Common.UserStory", "") or "")
        acceptance_criteria = str(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", "") or "")
        
        # Clean HTML and add non-empty fields
        try:
            if user_story:
                user_story = client.clean_html_string(user_story)
                parts.append(f"User Story: {user_story}")
            if acceptance_criteria:
                acceptance_criteria = client.clean_html_string(acceptance_criteria)
                parts.append(f"Acceptance Criteria: {acceptance_criteria}")
        except Exception:
            pass
    
    # Add comments if any
    if comments:
        for i, comment in enumerate(comments[:5]):  # Limit to first 5 comments
            try:
                clean_comment = client.clean_html_string(comment)
                if clean_comment.strip():
                    parts.append(f"Comment {i+1}: {clean_comment}")
            except Exception:
                pass
    
    return "\n\n".join(parts).strip()


async def poll_once(state: Dict[str, Any]) -> None:
    if not AZDO_ORG or not AZDO_PROJECT:
        raise SystemExit("Please set AZDO_ORG and AZDO_PROJECT constants at the top of poller.py.")
    
    if USE_OAUTH2:
        if not CLIENT_ID or not CLIENT_SECRET or not TENANT_ID:
            raise SystemExit("OAuth2 requires CLIENT_ID, CLIENT_SECRET, and TENANT_ID to be set.")
        client = AzureDevOpsClient(
            AZDO_ORG, 
            AZDO_PROJECT, 
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            tenant_id=TENANT_ID,
            use_oauth2=True
        )
    else:
        if not AZDO_PAT:
            raise SystemExit("Please set AZDO_PAT constant when not using OAuth2.")
        client = AzureDevOpsClient(AZDO_ORG, AZDO_PROJECT, auth_token=AZDO_PAT, use_oauth2=False)
    last_run: Optional[str] = state.get("last_run")
    now_iso = _now_iso()
    log(f"Polling for changes (last_run={last_run})")

    timeout = aiohttp.ClientTimeout(total=180)
    connector = aiohttp.TCPConnector(limit=50)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 1) Discover changed IDs since last run
        ids: List[int] = []
        if last_run:
            try:
                ids = await client.get_workitems_from_reporting_api(session, last_run)
            except Exception:
                ids = []

        # First run: seed timestamp and exit without retroactive processing
        if not last_run:
            state["last_run"] = now_iso
            _save_state(state)
            log("First run detected; seeding last_run and waiting for next cycle")
            return

        if not ids:
            state["last_run"] = now_iso
            _save_state(state)
            log("No changes since last run")
            return

        # 2) Fetch work item details in batch
        workitems = await client.get_workitem_details_batch(session, ids)
        log(f"Fetched details for {len(workitems)} items")

        # 3) Filter: created after last_run and matches iteration/area (tag filtering happens later)
        # Also track work items that already have AI comments to avoid infinite loops

        last_dt = _parse_ado_datetime(last_run)

        candidate_items: List[Dict[str, Any]] = []
        for wi in workitems:
            wid = wi.get("id")
            fields: Dict[str, Any] = wi.get("fields", {})
            created = fields.get("System.CreatedDate")
            changed = fields.get("System.ChangedDate")
            
            if wid is None:
                log("Skipping item without id")
                continue
            if not _match_filters(fields):
                log(f"Skipping WI #{wid}: iteration/area filter not matched")
                continue
                
            created_dt = _parse_ado_datetime(created)
            changed_dt = _parse_ado_datetime(changed)

            include = False
            reason = ""
            
            changed_by_raw = fields.get("System.ChangedBy")
            changed_by = str(changed_by_raw.get("displayName")) if isinstance(changed_by_raw, dict) else str(changed_by_raw or "")
            
            if last_dt and created_dt and created_dt >= last_dt:
                include = True
                reason = "created after last run"
            elif last_dt and changed_dt and changed_dt >= last_dt:
                # Skip processing if this work item was likely changed by our own AI comment
                # Only skip if the last change was made by the bot itself
                try:
                    if BOT_IDENTITY_NAME and BOT_IDENTITY_NAME.lower() in changed_by.lower():
                        log(f"Skipping WI #{wid}: last change by bot '{changed_by}'")
                        include = False
                    else:
                        include = True
                        reason = "changed after last run by user"
                except Exception as e:
                    log(f"Error evaluating ChangedBy for WI #{wid}: {e}")

            if include:
                log(f"Candidate WI #{wid}: {reason}")
                candidate_items.append(wi)
            else:
                log(
                    f"Skipping WI #{wid}: not newly created/changed since last run (created={created}, changed={changed})"
                )

        # 4) Process all candidate items (no tag filtering)
        new_items: List[Dict[str, Any]] = []
        for wi in candidate_items:
            wid = wi.get("id")
            log(f"Processing WI #{wid}: in defined area path")
            new_items.append(wi)

        if not new_items:
            state["last_run"] = now_iso
            _save_state(state)
            log("No new items to process after filtering")
            return

        # 5) For each new item: search for related tickets and post/update comment
        for wi in new_items:
            wid = int(wi["id"])  # safe from filtering above
            fields: Dict[str, Any] = wi.get("fields", {})
            title = str(fields.get("System.Title", ""))
            area_for_filter = str(fields.get("System.AreaPath", ""))
            changed = fields.get("System.ChangedDate")
            changed_dt = _parse_ado_datetime(changed)
            
            # Skip processing if this work item was likely changed by our own AI comment
            # Only skip if the last change was made by the bot itself
            changed_by_raw = fields.get("System.ChangedBy")
            changed_by = str(changed_by_raw.get("displayName")) if isinstance(changed_by_raw, dict) else str(changed_by_raw or "")

            try:
                if BOT_IDENTITY_NAME and BOT_IDENTITY_NAME.lower() in changed_by.lower():
                    log(f"Skipping WI #{wid}: last change by bot '{changed_by}'")
                    continue
            except Exception as e:
                log(f"Error evaluating ChangedBy for WI #{wid}: {e}")
            
            # Fetch comments for this work item
            comments: List[str] = []
            try:
                comments = await client.get_workitem_comments(session, wid)
                log(f"WI #{wid}: fetched {len(comments)} comments")
            except Exception as e:
                log(f"Failed to fetch comments for #{wid}: {e}")

            # Build query text based on work item type and include comments
            query_text = _build_query_text(fields, client, comments)

            # Search: prefer local API /search (expected behavior for this poller), else ADO Search
            try:
                log(
                    f"Processing WI #{wid}: searching for related tickets (local={USE_LOCAL_SEARCH}); "
                    f"area='{area_for_filter}'"
                )
                if USE_LOCAL_SEARCH:
                    hits, api_summary = await local_search(
                        session,
                        base_url=API_BASE_URL,
                        query_text=query_text,
                        topk=TOPK + 1,  # Get one extra result to account for potential self-reference
                    )
                else:
                    hits = await ado_search(
                        session,
                        org=AZDO_ORG,
                        project=AZDO_PROJECT,
                        query_text=query_text,
                        area=area_for_filter,
                        topk=TOPK + 1,  # Get one extra result to account for potential self-reference
                    )
                    api_summary = None
                
                # Filter out the current work item to prevent self-reference
                filtered_hits = []
                current_wid_str = str(wid)
                for hit in hits:
                    hit_fields = hit.get("fields", {})
                    hit_id = hit_fields.get("system.id") or hit_fields.get("System.Id")
                    if hit_id and str(hit_id) != current_wid_str:
                        filtered_hits.append(hit)
                    elif hit_id and str(hit_id) == current_wid_str:
                        log(f"WI #{wid}: filtered out self-reference from search results")
                
                # Limit to requested number of results after filtering
                hits = filtered_hits[:TOPK]
                
                log(f"WI #{wid}: found {len(hits)} related hits (after self-filtering)")
            except Exception as e:
                log(f"Search failed for #{wid}: {e}")
                continue

            # Generate comment text (no summary, just top 3 related tickets)
            comment_text = render_comment_html(
                org=LINK_ORG,
                project=LINK_PROJECT,
                new_title=title,
                hits=hits,
            )

            # Check if there's already an AI-generated comment to update
            try:
                existing_comment = await find_ai_comment(
                    session,
                    org=AZDO_ORG,
                    project=AZDO_PROJECT,
                    workitem_id=wid,
                )
                
                if existing_comment:
                    # Update existing comment
                    comment_id = existing_comment.get("id")
                    if comment_id is None:
                        log(f"WI #{wid}: found AI comment but no ID, posting new comment instead")
                        await post_comment(
                            session,
                            org=AZDO_ORG,
                            project=AZDO_PROJECT,
                            workitem_id=wid,
                            text=comment_text,
                        )
                        log(f"Posted new comment on WI #{wid}")
                    else:
                        log(f"WI #{wid}: updating existing AI comment (ID: {comment_id})")
                        await update_comment(
                            session,
                            org=AZDO_ORG,
                            project=AZDO_PROJECT,
                            workitem_id=wid,
                            comment_id=comment_id,
                            text=comment_text,
                        )
                        log(f"Updated existing comment on WI #{wid}")
                else:
                    # Post new comment
                    log(f"WI #{wid}: posting new AI comment (hits={len(hits)})")
                    await post_comment(
                        session,
                        org=AZDO_ORG,
                        project=AZDO_PROJECT,
                        workitem_id=wid,
                        text=comment_text,
                    )
                    log(f"Posted new comment on WI #{wid}")
            except Exception as e:
                log(f"Failed to handle comment on #{wid}: {e}")
                continue

            # Work item successfully processed

        # Persist state
        state["last_run"] = now_iso
        _save_state(state)
        log(f"Processed {len(new_items)} new item(s)")


async def main() -> None:
    auth_method = "OAuth2" if USE_OAUTH2 else "PAT"
    log(f"Starting poller for org='{AZDO_ORG}', project='{AZDO_PROJECT}', auth={auth_method}, poll_seconds={POLL_SECONDS}")
    state = _load_state()
    try:
        while True:
            try:
                await poll_once(state)
            except Exception as e:
                log(f"Poll error: {e}")
            log(f"Sleeping {POLL_SECONDS} seconds...")
            await asyncio.sleep(POLL_SECONDS)
    except asyncio.CancelledError:
        log("Cancellation received; shutting down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.", flush=True)
