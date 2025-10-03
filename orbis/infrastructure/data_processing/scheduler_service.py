"""
Task scheduler service for automated data ingestion.
Handles nightly scheduling of work item data ingestion and embedding generation.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, time, timedelta
from typing import Any

from config.settings import settings
from core.services.generic_data_ingestion import GenericDataIngestionService

logger = logging.getLogger(__name__)

class SchedulerService:
    """Service for scheduling automated tasks"""

    def __init__(self, data_ingestion_service: GenericDataIngestionService | None = None):
        self.data_ingestion_service = data_ingestion_service
        self.is_running = False
        self.scheduler_task: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

        # Configuration
        self.schedule_enabled = getattr(settings, 'SCHEDULER_ENABLED', False)
        self.schedule_time = self._parse_schedule_time(getattr(settings, 'SCHEDULED_INGESTION_TIME', '02:00'))
        self.schedule_interval_hours = getattr(settings, 'SCHEDULER_INTERVAL_HOURS', 24)

        # Callbacks for notifications
        self.on_task_start: Callable | None = None
        self.on_task_complete: Callable | None = None
        self.on_task_error: Callable | None = None

        # Last run tracking
        self.last_run_time: datetime | None = None
        self.last_run_result: dict[str, Any] | None = None

    def _parse_schedule_time(self, time_str: str) -> time:
        """Parse time string in HH:MM format"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid schedule time '{time_str}', using default 02:00")
            return time(2, 0)  # Default to 2 AM

    def _calculate_next_run(self) -> datetime:
        """Calculate the next scheduled run time.
        If an interval is configured, run every N hours since last run; otherwise, daily at schedule_time.
        """
        now = datetime.now()
        interval_hours = max(0, int(self.schedule_interval_hours or 0))

        if interval_hours > 0:
            if self.last_run_time:
                next_run = self.last_run_time + timedelta(hours=interval_hours)
            else:
                # First run: align to today's schedule_time if provided, else 'now + interval'
                scheduled_today = datetime.combine(now.date(), self.schedule_time)
                next_run = scheduled_today if now < scheduled_today else now + timedelta(hours=interval_hours)

            if next_run <= now:
                next_run = now + timedelta(hours=interval_hours)
            return next_run

        # Fallback: daily at schedule_time
        scheduled_today = datetime.combine(now.date(), self.schedule_time)
        if now >= scheduled_today:
            return scheduled_today + timedelta(days=1)
        return scheduled_today

    async def _wait_until_next_run(self) -> bool:
        """Wait until the next scheduled run time. Returns False if cancelled."""
        next_run = self._calculate_next_run()
        now = datetime.now()
        wait_seconds = (next_run - now).total_seconds()

        logger.info(f"Next scheduled ingestion: {next_run.strftime('%Y-%m-%d %H:%M:%S')} (in {wait_seconds/3600:.1f} hours)")

        try:
            await asyncio.sleep(wait_seconds)
            return True
        except asyncio.CancelledError:
            logger.info("Scheduler wait cancelled")
            return False

    async def _run_scheduled_task(self):
        """Run the scheduled data ingestion task"""
        if not self.data_ingestion_service:
            logger.error("Data ingestion service not available for scheduled task")
            return

        logger.info("Starting scheduled data ingestion task...")

        # Notify task start
        if self.on_task_start:
            try:
                await self._call_callback(self.on_task_start, {"type": "scheduled_ingestion_start"})
            except Exception as e:
                logger.warning(f"Task start callback failed: {e}")

        try:
            # Run data ingestion (incremental by default)
            result = await self.data_ingestion_service.ingest_all_sources(
                force_full_sync=False,
                skip_embedding=False
            )

            self.last_run_time = datetime.now()
            self.last_run_result = result

            logger.info(f"Scheduled data ingestion completed: {result}")

            # Notify task completion
            if self.on_task_complete:
                try:
                    await self._call_callback(self.on_task_complete, {
                        "type": "scheduled_ingestion_complete",
                        "result": result
                    })
                except Exception as e:
                    logger.warning(f"Task complete callback failed: {e}")

        except Exception as e:
            logger.error(f"Scheduled data ingestion failed: {e}")

            self.last_run_time = datetime.now()
            self.last_run_result = {"success": False, "error": str(e)}

            # Notify task error
            if self.on_task_error:
                try:
                    await self._call_callback(self.on_task_error, {
                        "type": "scheduled_ingestion_error",
                        "error": str(e)
                    })
                except Exception as e:
                    logger.warning(f"Task error callback failed: {e}")

    async def _call_callback(self, callback: Callable, data: dict[str, Any]):
        """Call a callback function, handling both sync and async functions"""
        if asyncio.iscoroutinefunction(callback):
            await callback(data)
        else:
            callback(data)

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler loop started")

        while self.is_running:
            try:
                # Wait until next scheduled time
                if not await self._wait_until_next_run():
                    break  # Cancelled

                if not self.is_running:
                    break  # Stopped while waiting

                # Run the scheduled task
                await self._run_scheduled_task()

            except asyncio.CancelledError:
                logger.info("Scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                # Wait a bit before retrying to avoid tight error loops
                try:
                    await asyncio.sleep(300)  # 5 minutes
                except asyncio.CancelledError:
                    break

        logger.info("Scheduler loop stopped")

    def start(self):
        """Start the scheduler"""
        if not self.schedule_enabled:
            logger.info("Scheduled ingestion is disabled")
            return

        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        if not self.data_ingestion_service:
            logger.error("Cannot start scheduler: data ingestion service not available")
            return

        logger.info(f"Starting scheduler (runs daily at {self.schedule_time.strftime('%H:%M')})")
        self.is_running = True

        # Get or create event loop
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, we'll create the task when loop is available
            self.loop = None

        # Start scheduler task
        if self.loop and self.loop.is_running():
            self.scheduler_task = self.loop.create_task(self._scheduler_loop())
        else:
            # Defer task creation until loop is running
            logger.info("Event loop not running, scheduler task will be created later")

    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return

        logger.info("Stopping scheduler...")
        self.is_running = False

        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()

        self.scheduler_task = None
        logger.info("Scheduler stopped")

    def ensure_task_running(self):
        """Ensure the scheduler task is running (call this from startup)"""
        if self.is_running and (not self.scheduler_task or self.scheduler_task.done()):
            try:
                loop = asyncio.get_running_loop()
                self.scheduler_task = loop.create_task(self._scheduler_loop())
                logger.info("Scheduler task created and started")
            except RuntimeError:
                logger.warning("No running event loop for scheduler task")

    async def trigger_manual_ingestion(self, force_full_sync: bool = False) -> dict[str, Any]:
        """Manually trigger data ingestion outside of schedule"""
        if not self.data_ingestion_service:
            raise ValueError("Data ingestion service not available")

        logger.info(f"Manual data ingestion triggered (force_full_sync={force_full_sync})")

        try:
            result = await self.data_ingestion_service.ingest_all_sources(
                force_full_sync=force_full_sync,
                skip_embedding=False
            )

            logger.info(f"Manual data ingestion completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Manual data ingestion failed: {e}")
            raise

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status information"""
        next_run = self._calculate_next_run() if self.is_running else None

        return {
            "enabled": self.schedule_enabled,
            "running": self.is_running,
            "schedule_time": self.schedule_time.strftime('%H:%M'),
            "next_run": next_run.isoformat() if next_run else None,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "last_run_result": self.last_run_result,
            "task_status": "running" if (self.scheduler_task and not self.scheduler_task.done()) else "stopped"
        }

    def set_callbacks(self,
                     on_start: Callable | None = None,
                     on_complete: Callable | None = None,
                     on_error: Callable | None = None):
        """Set callback functions for scheduler events"""
        self.on_task_start = on_start
        self.on_task_complete = on_complete
        self.on_task_error = on_error
