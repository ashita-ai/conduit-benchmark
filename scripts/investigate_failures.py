#!/usr/bin/env python3
"""
Investigate queries with 0.000 quality and 0.000 cost.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from conduit_bench.database import BenchmarkDatabase


async def investigate_failures():
    """Find and analyze failed queries."""

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

        # Get all evaluations for AlwaysBest
        evals = await db.get_query_evaluations(run_id)

        # Filter for failures (0.000 quality and 0.000 cost)
        failures = [e for e in evals if e['quality_score'] == 0.0 and e['cost'] == 0.0]

        print(f"Total AlwaysBest queries: {len(evals)}")
        print(f"Failed queries (0.000 quality AND 0.000 cost): {len(failures)}")
        print()

        if failures:
            print("Sample failures:")
            print(f"{'Query ID':<25} {'Model':<25} {'Latency':<10} {'Quality':<10} {'Cost':<12}")
            print("-" * 90)
            for f in failures[-10:]:  # Show last 10 failures
                print(f"{f['query_id']:<25} {f['model_id']:<25} {f['latency']:<10.2f} {f['quality_score']:<10.3f} ${f['cost']:<11.6f}")

        # Look for pattern - are these consecutive?
        if failures:
            failure_indices = [evals.index(f) for f in failures]
            print()
            print(f"Failure positions: {failure_indices[-20:]}")  # Last 20 failure positions
            print()

            # Check if failures are recent
            if len(evals) > 0:
                recent_failures = [e for e in evals[-20:] if e['quality_score'] == 0.0 and e['cost'] == 0.0]
                print(f"Failures in last 20 queries: {len(recent_failures)}")

        # Get database schema to understand what fields are available
        print("\n" + "="*90)
        print("Checking for detailed error information...")

        # Check if there are error fields in the query_evaluations table
        columns_query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'query_evaluations'
            ORDER BY ordinal_position
        """
        columns = await db.pool.fetch(columns_query)
        print("\nquery_evaluations table columns:")
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']}")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(investigate_failures())
