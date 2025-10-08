r"""
Minimal local poller for Azure DevOps work items.

Behavior:
- Polls Azure DevOps for newly created work items (since last run) using the Reporting API
- Filters by iteration path and/or area path prefix (configurable via constants at the top of this file)
- For each new item, queries the Work Item Search API with its title+description
- Optionally generates a short summary (if Azure OpenAI is configured)
- Posts a comment on the new item with the related tickets and the summary

Configuration via in-file constants (edit below):
- AZDO_ORG (required)
- AZDO_PROJECT (required)
- Authentication (choose one):
  - OAuth2 (recommended): Set USE_OAUTH2=True, CLIENT_ID, CLIENT_SECRET, TENANT_ID
  - PAT: Set USE_OAUTH2=False, AZDO_PAT (scopes: Work Items (Read & Write), Search (Read))
- ITERATION_PREFIX (optional)  e.g., "MyProject\Iteration 1"
- AREA_PREFIX (optional)       e.g., "MyProject\Area\SubArea"
- POLL_SECONDS (default: 60)
- TOPK (default: 5)
- STATE_PATH (default: ".ado_related/state.json")

Run locally in your .venv:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements-local.txt
  python poller.py
"""

import asyncio
import base64
import json
import os
from datetime import UTC, datetime
from html import escape
from typing import Any
from urllib.parse import quote

import aiohttp

from engine.agents.summary_agent import SearchResultsSummarizer
from engine.schemas import Ticket
from orbis_core.connectors.azure_devops import AzureDevOpsClient

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

# Use local Orbis API /search to get related tickets and summary
# The poller will call your FastAPI (main.py) at the address below.
DEFAULT_USE_LOCAL_SEARCH: bool = True
DEFAULT_API_BASE_URL: str = "http://127.0.0.1:7887"
# If your API has API key protection enabled, set these accordingly
DEFAULT_API_KEY_ENABLED: bool = False
DEFAULT_API_KEY: str = ""

# Optional override for hyperlink organization/project used in rendered comments.
# In some setups, the organization/project used for API polling differs from the
# organization/project that should be used for web hyperlinks. If not set, the
# values below are used.
DEFAULT_LINK_ORG: str = "hexagon-si-gpc"
DEFAULT_LINK_PROJECT: str = "GPC Support"


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


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


async def _auth_headers_oauth2(client_id: str, client_secret: str, tenant_id: str) -> dict[str, str]:
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

def _auth_headers_pat(pat: str) -> dict[str, str]:
    """Get PAT authentication headers"""
    token = base64.b64encode(f":{pat}".encode("ascii")).decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }


def _parse_ado_datetime(value: Any) -> datetime | None:
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


def _load_state() -> dict[str, Any]:
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_run": None, "processed_ids": []}


def _save_state(state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


async def ado_search(
    session: aiohttp.ClientSession, *, org: str, project: str, query_text: str, area: str, topk: int
) -> list[dict[str, Any]]:
    """Call Azure DevOps Work Item Search API and return results list."""
    project_path = quote(project)
    url = (
        f"https://almsearch.dev.azure.com/{org}/{project_path}/_apis/search/"
        f"workitemsearchresults?api-version=7.1-preview.1"
    )
    body: dict[str, Any] = {
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
) -> tuple[list[dict[str, Any]], str | None]:
    """Call local FastAPI /search and return pseudo-ADO hits and summary.

    Returns a tuple of (hits_like_ado, summary). Each hit includes fields.system.id/title
    so that existing rendering can be reused.
    """
    url = f"{base_url.rstrip('/')}/search"
    headers: dict[str, str] = {"Content-Type": "application/json"}
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
        hits: list[dict[str, Any]] = []
        for item in results:
            ticket = item.get("ticket", {})
            rid = ticket.get("id")
            rtitle = ticket.get("title", "")
            similarity_score = item.get("similarity_score")
            org_value = ticket.get("organization")
            project_value = ticket.get("project")
            hit: dict[str, Any] = {"fields": {"system.id": rid, "system.title": rtitle}}
            if similarity_score is not None:
                hit["similarity_score"] = similarity_score
            if org_value:
                hit["organization"] = org_value
            if project_value:
                hit["project"] = project_value
            hits.append(hit)
        return hits, summary

def render_comment_html(*, org: str, project: str, new_title: str, hits: list[dict[str, Any]], summary: str | None) -> str:
    """Render an HTML-formatted comment suitable for Azure DevOps rich text fields.

    Produces:
    - Top related ticket with hyperlink and optional relevancy
    - Summary block (with line breaks preserved)
    - Unordered list of other related tickets (with links and relevancy)
    """
    if not hits:
        return "<div>No clearly related prior tickets were found.</div>"

    # Identify top hit
    top_item: dict[str, Any] | None = None
    for r in hits:
        fields = r.get("fields", {})
        rid = fields.get("system.id") or fields.get("System.Id")
        if rid is not None:
            top_item = r
            break

    parts: list[str] = []
    if top_item is not None:
        fields = top_item.get("fields", {})
        rid = fields.get("system.id") or fields.get("System.Id")
        rtitle = fields.get("system.title") or fields.get("System.Title") or ""
        # Per-hit org/project override
        top_org = top_item.get("organization") or org
        top_project = top_item.get("project") or project
        link = _ado_work_item_link(top_org, top_project, rid)
        score_html = ""
        try:
            score_val = top_item.get("similarity_score")
            score = float(score_val) if score_val is not None else None
            if score is not None:
                score_html = f" — Relevancy: {score:.2f}"
        except Exception:
            pass
        parts.append(
            f"<div><strong>Top related ticket:</strong> <a href=\"{link}\">#{escape(str(rid))} {escape(str(rtitle))}</a>{score_html}</div>"
        )

    # Summary
    if summary:
        safe_summary = escape(summary).replace("\n", "<br/>")
        parts.append("<div style=\"margin-top:8px;\"><strong>Summary:</strong><br/>" + safe_summary + "</div>")

    # Other tickets
    other_items = hits[1:]
    if other_items:
        parts.append(
            f"<div style=\"margin-top:8px;\"><strong>Other related tickets for '{escape(new_title)}':</strong></div>"
        )
        parts.append("<ul>")
        for r in other_items:
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


async def update_workitem_tags(
    session: aiohttp.ClientSession, *, org: str, project: str, workitem_id: int, current_tags: str
) -> None:
    """Update work item tags by replacing 'Summon Orbis' with 'Orbis Summoned'.

    Uses direct field replacement approach. If the operation fails due to tag creation permissions,
    the calling code should handle the exception gracefully.
    """
    url = _ado_work_item_api_url(org, project, workitem_id)

    # Get appropriate auth headers based on configuration
    if USE_OAUTH2:
        headers = await _auth_headers_oauth2(CLIENT_ID, CLIENT_SECRET, TENANT_ID)
    else:
        headers = _auth_headers_pat(AZDO_PAT)

    headers["Content-Type"] = "application/json-patch+json"

    # Parse current tags into a list
    tag_list = []
    if current_tags:
        # Tags are semicolon-separated in Azure DevOps
        tag_list = [tag.strip() for tag in current_tags.split(";") if tag.strip()]

    # Only proceed if we actually need to remove the trigger tag
    if "Summon Orbis" not in tag_list:
        # Nothing to do - the trigger tag isn't present
        return

    # Remove the old tag and add the new tag
    tag_list.remove("Summon Orbis")
    if "Orbis Summoned" not in tag_list:
        tag_list.append("Orbis Summoned")

    # Convert back to semicolon-separated string
    updated_tags = "; ".join(tag_list) if tag_list else ""

    # Use "replace" operation to completely replace the tags field with our updated list
    # This ensures the old tag is removed and the new tag is added
    body = [
        {"op": "replace", "path": "/fields/System.Tags", "value": updated_tags},
    ]

    async with session.patch(url, json=body, headers=headers) as r:
        r.raise_for_status()


def _match_filters(fields: dict[str, Any]) -> bool:
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


def _build_query_text(fields: dict[str, Any], client: Any, comments: list[str]) -> str:
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


async def poll_once(state: dict[str, Any]) -> None:
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
    last_run: str | None = state.get("last_run")
    now_iso = _now_iso()
    log(f"Polling for changes (last_run={last_run})")

    timeout = aiohttp.ClientTimeout(total=180)
    connector = aiohttp.TCPConnector(limit=50)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 1) Discover changed IDs since last run
        ids: list[int] = []
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
        processed_ids: list[int] = state.get("processed_ids", []) or []
        processed_set = {int(x) for x in processed_ids}

        last_dt = _parse_ado_datetime(last_run)

        candidate_items: list[dict[str, Any]] = []
        for wi in workitems:
            wid = wi.get("id")
            fields: dict[str, Any] = wi.get("fields", {})
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
            if last_dt and created_dt and created_dt >= last_dt:
                include = True
                reason = "created after last run"
            elif last_dt and changed_dt and changed_dt >= last_dt:
                include = True
                reason = "changed after last run"

            if include:
                log(f"Candidate WI #{wid}: {reason}")
                candidate_items.append(wi)
            else:
                log(
                    f"Skipping WI #{wid}: not newly created/changed since last run (created={created}, changed={changed})"
                )

        # 4) Apply tag filter - process all items with the correct tag regardless of previous processing
        new_items: list[dict[str, Any]] = []
        for wi in candidate_items:
            wid = wi.get("id")
            fields: dict[str, Any] = wi.get("fields", {})
            tags = str(fields.get("System.Tags", ""))

            # Check for required tag - process any ticket with this tag
            if "Summon Orbis" not in tags:
                log(f"Skipping WI #{wid}: does not have 'Summon Orbis' tag")
                continue

            log(f"Processing WI #{wid}: has 'Summon Orbis' tag")
            new_items.append(wi)

        if not new_items:
            state["last_run"] = now_iso
            _save_state(state)
            log("No new items to process after filtering")
            return

        # 5) For each new item: search (either local API or ADO), (optional) summarize, and comment
        search_results_summarizer = SearchResultsSummarizer()

        for wi in new_items:
            wid = int(wi["id"])  # safe from filtering above
            fields: dict[str, Any] = wi.get("fields", {})
            title = str(fields.get("System.Title", ""))
            tags = str(fields.get("System.Tags", ""))
            area_for_filter = str(fields.get("System.AreaPath", ""))

            # Fetch comments for this work item
            comments: list[str] = []
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
                        topk=TOPK,
                    )
                else:
                    hits = await ado_search(
                        session,
                        org=AZDO_ORG,
                        project=AZDO_PROJECT,
                        query_text=query_text,
                        area=area_for_filter,
                        topk=TOPK,
                    )
                    api_summary = None
                log(f"WI #{wid}: found {len(hits)} related hits")
            except Exception as e:
                log(f"Search failed for #{wid}: {e}")
                continue

            # If local API provided a summary, use it directly. Otherwise, optionally generate one
            # using Azure OpenAI (if configured) based on minimal ticket data built from hits.
            summary: str | None = None
            try:
                # If local API provided a summary, use it; otherwise optionally use Azure OpenAI
                if USE_LOCAL_SEARCH and api_summary:
                    summary = api_summary
                elif search_results_summarizer.is_configured():
                    # Minimal tickets from search results
                    minimal_tickets: list[Ticket] = []
                    for r in hits:
                        fields_r = r.get("fields", {})
                        rid = fields_r.get("system.id") or fields_r.get("System.Id")
                        rtitle = fields_r.get("system.title") or fields_r.get("System.Title") or ""
                        minimal_tickets.append(Ticket(id=str(rid), title=rtitle, description=None, comments=[]))
                    summary = search_results_summarizer.generate_summary(query_text, minimal_tickets, similarity_scores=None)
            except Exception as e:
                log(f"Summary failed for #{wid}: {e}")
                summary = None

            comment_text = render_comment_html(
                org=LINK_ORG,
                project=LINK_PROJECT,
                new_title=title,
                hits=hits,
                summary=summary,
            )

            try:
                log(
                    f"WI #{wid}: posting comment (hits={len(hits)}, summary={'yes' if summary else 'no'})"
                )
                await post_comment(
                    session,
                    org=AZDO_ORG,
                    project=AZDO_PROJECT,
                    workitem_id=wid,
                    text=comment_text,
                )
                log(f"Comment posted on WI #{wid}")
            except Exception as e:
                log(f"Failed to comment on #{wid}: {e}")
                continue

            # Update tags: replace "Summon Orbis" with "Orbis Summoned"
            try:
                await update_workitem_tags(
                    session,
                    org=AZDO_ORG,
                    project=AZDO_PROJECT,
                    workitem_id=wid,
                    current_tags=tags,
                )
                log(f"Updated tags on WI #{wid}")
            except Exception as e:
                log(f"Failed to update tags on #{wid}: {e}")

            # Mark processed
            processed_set.add(wid)

        # Persist state
        state["processed_ids"] = sorted(processed_set)[-1000:]  # cap growth
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
