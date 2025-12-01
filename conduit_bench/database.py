"""Database connection and models for benchmark result persistence.

Provides async PostgreSQL connection handling and result storage for streaming benchmark writes.
"""

import asyncio
import json
import math
import os
from datetime import datetime
from typing import Any, Callable, TypeVar

import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

T = TypeVar('T')


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Sanitize metadata by converting Infinity/NaN to JSON-safe values.

    PostgreSQL JSONB cannot store float Infinity or NaN values. This function
    converts them to string representations for database persistence.

    Args:
        metadata: Metadata dictionary that may contain Infinity/NaN values

    Returns:
        Sanitized metadata dictionary safe for JSON serialization
    """
    if not metadata:
        return metadata

    def sanitize_value(v: Any) -> Any:
        """Recursively sanitize a value."""
        if isinstance(v, float):
            if math.isinf(v):
                return "Infinity" if v > 0 else "-Infinity"
            if math.isnan(v):
                return "NaN"
        elif isinstance(v, dict):
            return {k: sanitize_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [sanitize_value(val) for val in v]
        return v

    return sanitize_value(metadata)


async def retry_with_exponential_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
) -> T:
    """Retry async operation with exponential backoff.

    Only retries on transient database errors (connection issues, timeouts).
    Does not retry on constraint violations or data errors.

    Args:
        func: Async function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries exhausted
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except (
            asyncpg.PostgresConnectionError,
            asyncpg.ConnectionDoesNotExistError,
            asyncpg.InterfaceError,
            asyncio.TimeoutError,
            ConnectionError,
        ) as e:
            # Transient errors - retry with backoff
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                raise
        except (
            asyncpg.UniqueViolationError,
            asyncpg.ForeignKeyViolationError,
            asyncpg.CheckViolationError,
            asyncpg.NotNullViolationError,
        ):
            # Data integrity errors - don't retry
            raise
        except Exception:
            # Unknown errors - don't retry
            raise

    # Should never reach here, but for type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error - no exception but no success")


class BenchmarkDatabase:
    """Async PostgreSQL connection manager for benchmark results.

    Handles streaming writes for benchmark runs, algorithm runs, and query evaluations
    to enable real-time analysis and prevent data loss on crashes.
    """

    def __init__(self, database_url: str | None = None) -> None:
        """Initialize database connection manager.

        Args:
            database_url: PostgreSQL connection URL. If None, reads from DATABASE_URL env var.
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create async connection pool."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=15,  # Support 11 parallel algorithms + overhead
            command_timeout=120,  # Increase for network latency
            statement_cache_size=0,  # Required for Supabase/pgbouncer
        )

    async def disconnect(self) -> None:
        """Close connection pool with timeout to prevent hanging."""
        if self.pool:
            import asyncio
            try:
                # Use 5 second timeout for pool.close() to prevent hanging
                await asyncio.wait_for(self.pool.close(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force terminate connections if graceful close times out (synchronous method)
                self.pool.terminate()
            finally:
                self.pool = None

    async def create_benchmark_run(
        self,
        benchmark_id: str,
        dataset_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new benchmark run record.

        Args:
            benchmark_id: Unique identifier for the benchmark run
            dataset_size: Number of queries in the dataset
            metadata: Optional metadata dictionary
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        await self.pool.execute(
            """
            INSERT INTO benchmark_runs (benchmark_id, dataset_size, metadata)
            VALUES ($1, $2, $3::jsonb)
            """,
            benchmark_id,
            dataset_size,
            json.dumps(sanitize_metadata(metadata)) if metadata else None,
        )

    async def create_algorithm_run(
        self,
        run_id: str,
        benchmark_id: str,
        algorithm_name: str,
        started_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new algorithm run record.

        Args:
            run_id: Unique identifier for the algorithm run
            benchmark_id: Parent benchmark ID
            algorithm_name: Name of the bandit algorithm
            started_at: Start timestamp
            metadata: Optional metadata dictionary
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        await self.pool.execute(
            """
            INSERT INTO algorithm_runs (
                run_id, benchmark_id, algorithm_name,
                total_cost, average_quality, total_queries,
                started_at, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            """,
            run_id,
            benchmark_id,
            algorithm_name,
            0.0,  # Initial values
            0.0,
            0,
            started_at,
            json.dumps(sanitize_metadata(metadata)) if metadata else None,
        )

    async def write_query_evaluation(
        self,
        run_id: str,
        query_id: str,
        model_id: str,
        quality_score: float,
        cost: float,
        latency: float,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a single query evaluation result (streaming) with retry.

        Args:
            run_id: Algorithm run ID
            query_id: Query identifier
            model_id: Selected model ID
            quality_score: Evaluated quality (0-1)
            cost: Query cost in dollars
            latency: Query latency in seconds
            success: Whether execution succeeded
            metadata: Optional metadata dictionary
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        async def _write():
            await self.pool.execute(
                """
                INSERT INTO query_evaluations (
                    run_id, query_id, model_id,
                    quality_score, cost, latency, success, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                ON CONFLICT (run_id, query_id) DO NOTHING
                """,
                run_id,
                query_id,
                model_id,
                quality_score,
                cost,
                latency,
                success,
                json.dumps(sanitize_metadata(metadata)) if metadata else None,
            )

        await retry_with_exponential_backoff(_write)

    async def update_algorithm_run(
        self,
        run_id: str,
        total_cost: float,
        average_quality: float,
        total_queries: int,
        completed_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update algorithm run with cumulative metrics.

        Args:
            run_id: Algorithm run ID
            total_cost: Cumulative cost
            average_quality: Average quality across all queries
            total_queries: Number of queries processed
            completed_at: Completion timestamp (if completed)
            metadata: Algorithm-specific metadata (e.g., learned parameters)
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        if completed_at:
            if metadata:
                await self.pool.execute(
                    """
                    UPDATE algorithm_runs
                    SET total_cost = $1,
                        average_quality = $2,
                        total_queries = $3,
                        completed_at = $4,
                        metadata = $5
                    WHERE run_id = $6
                    """,
                    total_cost,
                    average_quality,
                    total_queries,
                    completed_at,
                    json.dumps(sanitize_metadata(metadata)),
                    run_id,
                )
            else:
                await self.pool.execute(
                    """
                    UPDATE algorithm_runs
                    SET total_cost = $1,
                        average_quality = $2,
                        total_queries = $3,
                        completed_at = $4
                    WHERE run_id = $5
                    """,
                    total_cost,
                    average_quality,
                    total_queries,
                    completed_at,
                    run_id,
                )
        else:
            if metadata:
                await self.pool.execute(
                    """
                    UPDATE algorithm_runs
                    SET total_cost = $1,
                        average_quality = $2,
                        total_queries = $3,
                        metadata = $4
                    WHERE run_id = $5
                    """,
                    total_cost,
                    average_quality,
                    total_queries,
                    json.dumps(sanitize_metadata(metadata)),
                    run_id,
                )
            else:
                await self.pool.execute(
                    """
                    UPDATE algorithm_runs
                    SET total_cost = $1,
                        average_quality = $2,
                        total_queries = $3
                    WHERE run_id = $4
                    """,
                    total_cost,
                    average_quality,
                    total_queries,
                    run_id,
                )

    async def get_benchmark_run(self, benchmark_id: str) -> dict[str, Any] | None:
        """Retrieve benchmark run by ID.

        Args:
            benchmark_id: Benchmark identifier

        Returns:
            Benchmark run data or None if not found
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        row = await self.pool.fetchrow(
            """
            SELECT benchmark_id, dataset_size, created_at, metadata
            FROM benchmark_runs
            WHERE benchmark_id = $1
            """,
            benchmark_id,
        )

        if row:
            return dict(row)
        return None

    async def get_algorithm_runs(self, benchmark_id: str) -> list[dict[str, Any]]:
        """Retrieve all algorithm runs for a benchmark.

        Args:
            benchmark_id: Benchmark identifier

        Returns:
            List of algorithm run data dictionaries
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        rows = await self.pool.fetch(
            """
            SELECT run_id, benchmark_id, algorithm_name,
                   total_cost, average_quality, total_queries,
                   started_at, completed_at, metadata
            FROM algorithm_runs
            WHERE benchmark_id = $1
            ORDER BY started_at
            """,
            benchmark_id,
        )

        return [dict(row) for row in rows]

    async def get_query_evaluations(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all query evaluations for an algorithm run.

        Args:
            run_id: Algorithm run identifier

        Returns:
            List of query evaluation data dictionaries
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        rows = await self.pool.fetch(
            """
            SELECT evaluation_id, run_id, query_id, model_id,
                   quality_score, cost, latency, success,
                   created_at, metadata
            FROM query_evaluations
            WHERE run_id = $1
            ORDER BY created_at
            """,
            run_id,
        )

        return [dict(row) for row in rows]
