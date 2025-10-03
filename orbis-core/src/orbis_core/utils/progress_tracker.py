"""
Progress tracking utility for long-running operations.
Provides ETA calculation and periodic logging.
"""
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress and logs ETA for batch operations"""

    def __init__(self, total: int, log_interval_percent: int = 10, operation_name: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            log_interval_percent: Log progress every N percent (default: 10)
            operation_name: Name of the operation for logging (default: "Processing")
        """
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.last_percent_logged = -1
        self.log_interval = log_interval_percent
        self.operation_name = operation_name

    def update(self, count: int = 1, custom_message: Optional[str] = None):
        """
        Update progress and log if threshold reached.

        Args:
            count: Number of items processed in this update (default: 1)
            custom_message: Optional custom message to include in log
        """
        self.processed += count

        if self.total == 0:
            return

        percent = int((self.processed * 100) / self.total)

        # Log at intervals or on completion
        if percent >= self.last_percent_logged + self.log_interval or self.processed == self.total:
            elapsed = time.time() - self.start_time
            rate = (self.processed / elapsed) if elapsed > 0 else 0.0
            remaining = self.total - self.processed
            eta_seconds = (remaining / rate) if rate > 0 else 0.0
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)

            base_message = (
                f"{self.operation_name}: {self.processed}/{self.total} ({percent}%) - "
                f"elapsed {elapsed:.1f}s, ETA {eta_min:02d}:{eta_sec:02d}"
            )

            if custom_message:
                base_message = f"{base_message} - {custom_message}"

            logger.info(base_message)
            self.last_percent_logged = percent

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time

    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.processed >= self.total
