#!/usr/bin/env python3
"""
Check success field and metadata for failed queries.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from conduit_bench.database import BenchmarkDatabase


async def check_failure_details():
    """Check success field and metadata for failures."""

    db = BenchmarkDatabase()

    try:
        await db.connect()

        # Get latest benchmark
        row = await db.pool.fetchrow(
            """
            SELECT benchmark_id
            FROM benchmark_runs
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        benchmark_id = row['benchmark_id']

        # Get AlwaysBest run
        algo_runs = await db.get_algorithm_runs(benchmark_id)
        always_best_run = next(r for r in algo_runs if r['algorithm_name'] == 'always_best')
        run_id = always_best_run['run_id']

        # Get failed queries with all fields
        failed_queries = await db.pool.fetch(
            """
            SELECT query_id, model_id, quality_score, cost, latency, success, metadata
            FROM query_evaluations
            WHERE run_id = $1
              AND quality_score = 0.0
              AND cost = 0.0
            ORDER BY created_at DESC
            LIMIT 10
            """,
            run_id
        )

        print(f"Checking last {len(failed_queries)} failed queries:")
        print("=" * 100)

        for i, q in enumerate(failed_queries, 1):
            print(f"\n#{i}: {q['query_id']}")
            print(f"  Model: {q['model_id']}")
            print(f"  Success: {q['success']}")
            print(f"  Quality: {q['quality_score']}, Cost: ${q['cost']}, Latency: {q['latency']}s")

            if q['metadata']:
                metadata = q['metadata'] if isinstance(q['metadata'], dict) else json.loads(q['metadata'])
                print(f"  Metadata:")
                for key, value in metadata.items():
                    if key == 'response' and value:
                        print(f"    {key}: {str(value)[:200]}...")  # Truncate long responses
                    else:
                        print(f"    {key}: {value}")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(check_failure_details())
