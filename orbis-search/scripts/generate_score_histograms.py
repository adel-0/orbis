"""Generate synthetic queries from stored tickets and evaluate retrieval quality.

Workflow
========
- Load tickets directly from the local database using the existing SQLAlchemy
  layer (`WorkItemService`).
- Build simple queries from each ticket (title + description).
- Call the running `/search` endpoint for each synthetic query.
- Track whether the SOURCE ticket appears in the results and at what rank.
- Calculate proper IR evaluation metrics: MRR, Recall@k, NDCG.
- Record score statistics and rank distributions.

The script writes evaluation metrics, score statistics, and detailed results
to `data/calibration/` for analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests

# Ensure project root on sys.path when executed as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.db.session import DatabaseManager
from models.schemas import Ticket
from orbis_core.search import HybridSearchService
from services.work_item_service import WorkItemService


API_ROOT = "http://127.0.0.1:8887"
SEARCH_PATH = "/search"

DEFAULT_MAX_TICKETS = 300
DEFAULT_TIMEOUT = 90
DEFAULT_TOP_K = 50


@dataclasses.dataclass
class QueryEvaluation:
    """Evaluation results for a single query (source ticket)."""
    query_ticket_id: str
    query_text: str
    source_ticket_found: bool
    source_ticket_rank: Optional[int]  # 1-indexed rank, None if not found
    reciprocal_rank: float  # 1/rank if found, 0 if not found
    total_results_returned: int
    top_result_id: Optional[str]
    top_result_score: Optional[float]

    def to_dict(self) -> dict:
        return {
            "query_ticket_id": self.query_ticket_id,
            "query_text": self.query_text,
            "source_ticket_found": self.source_ticket_found,
            "source_ticket_rank": self.source_ticket_rank,
            "reciprocal_rank": self.reciprocal_rank,
            "total_results_returned": self.total_results_returned,
            "top_result_id": self.top_result_id,
            "top_result_score": self.top_result_score,
        }


@dataclasses.dataclass
class ScoreRecord:
    """Score details for a specific candidate ticket in a query result."""
    query_ticket_id: str
    candidate_ticket_id: str
    rank: int  # 1-indexed
    is_source_ticket: bool
    rrf_score: float
    confidence_score: float
    similarity_score: float
    bm25_score: float
    recency_boost: Optional[float]

    def to_dict(self) -> dict:
        return {
            "query_ticket_id": self.query_ticket_id,
            "candidate_ticket_id": self.candidate_ticket_id,
            "rank": self.rank,
            "is_source_ticket": self.is_source_ticket,
            "rrf_score": self.rrf_score,
            "confidence_score": self.confidence_score,
            "similarity_score": self.similarity_score,
            "bm25_score": self.bm25_score,
            "recency_boost": self.recency_boost,
        }


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_tickets(limit: Optional[int], shuffle: bool) -> List[Ticket]:
    """Load tickets from the local database via `WorkItemService`."""

    # Initialize database connection (no-op if API already did this).
    DatabaseManager.init_database()

    with WorkItemService() as wi_service:
        tickets = wi_service.get_tickets_from_all_sources()

    if shuffle:
        random.shuffle(tickets)

    if limit and limit > 0:
        return tickets[:limit]
    return tickets


def make_query_text(ticket: Ticket) -> Optional[str]:
    title = (ticket.title or "").strip()
    description = (ticket.description or "").strip()
    if not title:
        return None
    return f"{title}. {description}" if description else title


def call_search(query: str, top_k: int, api_key: Optional[str]) -> dict:
    url = f"{API_ROOT}{SEARCH_PATH}"
    payload = {"query": query, "top_k": top_k, "include_summary": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    response = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def evaluate_queries(
    tickets: Sequence[Ticket],
    top_k: int,
    api_key: Optional[str],
    sleep_seconds: float,
) -> Tuple[List[QueryEvaluation], List[ScoreRecord], Counter]:
    """
    Evaluate search quality by querying with each ticket and tracking if the source
    ticket is found in the results.

    Returns:
        - query_evaluations: Per-query metrics (MRR, rank, etc.)
        - score_records: Detailed scores for all returned candidates
        - failures: Count of different failure types
    """
    query_evaluations: List[QueryEvaluation] = []
    score_records: List[ScoreRecord] = []
    failures: Counter = Counter()

    for idx, ticket in enumerate(tickets, start=1):
        query_text = make_query_text(ticket)
        if not query_text:
            failures["empty_query"] += 1
            continue

        try:
            response = call_search(query_text, top_k=top_k, api_key=api_key)
        except requests.HTTPError as exc:
            failures[f"http_{exc.response.status_code}"] += 1
            continue
        except requests.RequestException:
            failures["network_error"] += 1
            continue

        results = response.get("results", [])

        # Find the source ticket in results
        source_rank = None
        source_found = False
        for rank, result in enumerate(results, start=1):
            candidate = result.get("ticket", {})
            candidate_id = str(candidate.get("id", ""))
            is_source = str(ticket.id) == candidate_id

            if is_source:
                source_rank = rank
                source_found = True

            # Record detailed scores for all candidates
            score_records.append(
                ScoreRecord(
                    query_ticket_id=str(ticket.id),
                    candidate_ticket_id=candidate_id,
                    rank=rank,
                    is_source_ticket=is_source,
                    rrf_score=float(result.get("rrf_score", 0.0)),
                    confidence_score=float(result.get("confidence_score", 0.0)),
                    similarity_score=float(result.get("similarity_score", 0.0)),
                    bm25_score=float(result.get("bm25_score", 0.0)),
                    recency_boost=_safe_float(result.get("recency_boost")),
                )
            )

        # Calculate reciprocal rank
        reciprocal_rank = (1.0 / source_rank) if source_found else 0.0

        # Get top result info
        top_result_id = None
        top_result_score = None
        if results:
            top_result_id = str(results[0].get("ticket", {}).get("id", ""))
            # Use rerank_score if available, otherwise confidence_score
            top_result_score = float(
                results[0].get("rerank_score") or results[0].get("confidence_score", 0.0)
            )

        # Record query evaluation
        query_evaluations.append(
            QueryEvaluation(
                query_ticket_id=str(ticket.id),
                query_text=query_text,
                source_ticket_found=source_found,
                source_ticket_rank=source_rank,
                reciprocal_rank=reciprocal_rank,
                total_results_returned=len(results),
                top_result_id=top_result_id,
                top_result_score=top_result_score,
            )
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if idx % 100 == 0:
            found_count = sum(1 for qe in query_evaluations if qe.source_ticket_found)
            print(f"Processed {idx} tickets... ({found_count}/{len(query_evaluations)} source tickets found)")

    return query_evaluations, score_records, failures


def _safe_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_ir_metrics(query_evaluations: List[QueryEvaluation]) -> dict:
    """
    Calculate Information Retrieval evaluation metrics.

    Returns:
        Dictionary with MRR, Recall@k, and rank distribution statistics
    """
    if not query_evaluations:
        return {}

    total_queries = len(query_evaluations)
    found_queries = [qe for qe in query_evaluations if qe.source_ticket_found]
    not_found_queries = [qe for qe in query_evaluations if not qe.source_ticket_found]

    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = [qe.reciprocal_rank for qe in query_evaluations]
    mrr = statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Recall@k for different k values
    recall_at_k = {}
    for k in [1, 3, 5, 10, 20, 50]:
        found_at_k = sum(
            1 for qe in query_evaluations
            if qe.source_ticket_found and qe.source_ticket_rank <= k
        )
        recall_at_k[f"recall@{k}"] = found_at_k / total_queries if total_queries > 0 else 0.0

    # Rank distribution (for found queries)
    rank_distribution = {}
    if found_queries:
        ranks = [qe.source_ticket_rank for qe in found_queries]
        rank_distribution = {
            "min_rank": min(ranks),
            "max_rank": max(ranks),
            "mean_rank": statistics.fmean(ranks),
            "median_rank": statistics.median(ranks),
        }

    return {
        "total_queries": total_queries,
        "source_found_count": len(found_queries),
        "source_not_found_count": len(not_found_queries),
        "source_found_rate": len(found_queries) / total_queries if total_queries > 0 else 0.0,
        "mrr": mrr,
        **recall_at_k,
        "rank_distribution": rank_distribution,
    }


def summarize_scores(records: List[ScoreRecord]) -> dict:
    """
    Summarize score distributions across all candidates.
    """
    def column(values: Iterable[Optional[float]]) -> List[float]:
        return [v for v in values if v is not None]

    stats: dict[str, dict[str, float]] = {}
    metrics = {
        "rrf_score": [r.rrf_score for r in records],
        "confidence_score": [r.confidence_score for r in records],
        "similarity_score": [r.similarity_score for r in records],
        "bm25_score": [r.bm25_score for r in records],
        "recency_boost": column(r.recency_boost for r in records),
    }

    for name, values in metrics.items():
        if not values:
            stats[name] = {"count": 0}
            continue
        values_sorted = sorted(values)
        count = len(values)
        stats[name] = {
            "count": count,
            "mean": statistics.fmean(values),
            "stdev": statistics.pstdev(values) if count > 1 else 0.0,
            "min": values_sorted[0],
            "p25": percentile(values_sorted, 25),
            "p50": percentile(values_sorted, 50),
            "p75": percentile(values_sorted, 75),
            "p90": percentile(values_sorted, 90),
            "p95": percentile(values_sorted, 95),
            "p99": percentile(values_sorted, 99),
            "max": values_sorted[-1],
        }

    return stats


def percentile(sorted_values: List[float], percent: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (percent / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def write_outputs(
    query_evaluations: List[QueryEvaluation],
    score_records: List[ScoreRecord],
    ir_metrics: dict,
    score_stats: dict,
    failures: Counter,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """
    Write evaluation results to multiple files:
    - evaluations.jsonl: Per-query evaluation results (MRR, rank, etc.)
    - scores.jsonl: Detailed scores for all candidates
    - metrics.json: Aggregated IR metrics (MRR, Recall@k, etc.)
    - score_stats.json: Score distribution statistics
    - meta.json: Metadata about the run
    """
    ensure_output_dir(output_dir)

    # Per-query evaluations
    eval_path = output_dir / f"{output_prefix}_evaluations.jsonl"
    with eval_path.open("w", encoding="utf-8") as fh:
        for qe in query_evaluations:
            fh.write(json.dumps(qe.to_dict(), ensure_ascii=False) + "\n")

    # Detailed score records
    scores_path = output_dir / f"{output_prefix}_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as fh:
        for record in score_records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    # IR metrics
    metrics_path = output_dir / f"{output_prefix}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(ir_metrics, fh, indent=2, ensure_ascii=False)

    # Score statistics
    stats_path = output_dir / f"{output_prefix}_score_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(score_stats, fh, indent=2, ensure_ascii=False)

    # Metadata
    metadata = {
        "total_queries": len(query_evaluations),
        "total_score_records": len(score_records),
        "failures": dict(failures),
    }
    meta_path = output_dir / f"{output_prefix}_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nEvaluation complete!")
    print(f"  Queries evaluated: {len(query_evaluations)}")
    print(f"  Score records: {len(score_records)}")
    print(f"  MRR: {ir_metrics.get('mrr', 0.0):.4f}")
    print(f"  Recall@1: {ir_metrics.get('recall@1', 0.0):.2%}")
    print(f"  Recall@5: {ir_metrics.get('recall@5', 0.0):.2%}")
    print(f"  Source found rate: {ir_metrics.get('source_found_rate', 0.0):.2%}")
    print(f"\nFiles written to {output_dir}:")
    print(f"  - {eval_path.name} (per-query evaluations)")
    print(f"  - {scores_path.name} (detailed scores)")
    print(f"  - {metrics_path.name} (IR metrics)")
    print(f"  - {stats_path.name} (score statistics)")
    if failures:
        print(f"\nEncountered failures: {dict(failures)}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate search quality using synthetic queries and calculate IR metrics (MRR, Recall@k)."
    )
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("API_KEY"), help="API key for protected endpoints")
    parser.add_argument("--max-tickets", type=int, default=DEFAULT_MAX_TICKETS, help="Limit number of tickets to sample (0 = all)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="top_k value to request from search")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--output-dir", default="data/calibration", help="Directory to write outputs")
    parser.add_argument("--output-prefix", default="calibration", help="Filename prefix for outputs")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle tickets before sampling for better coverage")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        tickets = load_tickets(limit=args.max_tickets, shuffle=args.shuffle)
    except Exception as exc:
        print(f"Failed to load tickets: {exc}", file=sys.stderr)
        return 1

    if not tickets:
        print("No tickets available for evaluation", file=sys.stderr)
        return 1

    print(f"Evaluating {len(tickets)} tickets with top_k={args.top_k}...")

    # Evaluate queries and collect scores
    query_evaluations, score_records, failures = evaluate_queries(
        tickets=tickets,
        top_k=args.top_k,
        api_key=args.api_key,
        sleep_seconds=args.sleep,
    )

    # Calculate IR metrics
    ir_metrics = calculate_ir_metrics(query_evaluations)

    # Summarize score distributions
    score_stats = summarize_scores(score_records)

    # Write all outputs
    output_dir = Path(args.output_dir)
    write_outputs(
        query_evaluations=query_evaluations,
        score_records=score_records,
        ir_metrics=ir_metrics,
        score_stats=score_stats,
        failures=failures,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 