#!/usr/bin/env python3
"""
Query the PostgreSQL database to monitor benchmark progress in real-time.

Usage:
    python scripts/query_database.py [benchmark_id]
    python scripts/query_database.py --dataset gsm8k
    python scripts/query_database.py --dataset mmlu
    python scripts/query_database.py --all
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from conduit_bench.database import BenchmarkDatabase


async def query_progress(benchmark_id: str | None = None, dataset: str | None = None, show_all: bool = False):
    """Query and display benchmark progress from database.

    Args:
        benchmark_id: Specific benchmark ID to query
        dataset: Filter by dataset name (e.g., 'gsm8k', 'mmlu')
        show_all: Show all running benchmarks grouped by dataset
    """

    # Initialize database
    db = BenchmarkDatabase()

    try:
        await db.connect()
        print("✅ Connected to database\n")

        # Get benchmark(s) to display
        benchmarks_to_show = []

        if benchmark_id:
            # Specific benchmark ID provided
            row = await db.pool.fetchrow(
                """
                SELECT benchmark_id, dataset_size, created_at, metadata
                FROM benchmark_runs
                WHERE benchmark_id = $1
                """,
                benchmark_id
            )
            if row:
                benchmarks_to_show.append(row)
            else:
                print(f"No benchmark found with ID: {benchmark_id}")
                return

        elif show_all:
            # Show all running benchmarks
            rows = await db.pool.fetch(
                """
                SELECT br.benchmark_id, br.dataset_size, br.created_at, br.metadata
                FROM benchmark_runs br
                WHERE EXISTS (
                    SELECT 1 FROM algorithm_runs ar
                    WHERE ar.benchmark_id = br.benchmark_id
                    AND ar.completed_at IS NULL
                )
                ORDER BY br.created_at DESC
                """
            )
            benchmarks_to_show = rows

        elif dataset:
            # Get latest benchmark for specific dataset
            # Note: We need to check if metadata contains the dataset info
            # The CLI stores algorithm names in metadata, not dataset name directly
            # We'll need to infer from the output filename or add dataset to metadata
            rows = await db.pool.fetch(
                """
                SELECT benchmark_id, dataset_size, created_at, metadata
                FROM benchmark_runs
                ORDER BY created_at DESC
                """
            )

            # Filter by checking algorithm runs for query_ids that match dataset pattern
            for row in rows:
                # Check first query evaluation to determine dataset
                algo_runs = await db.get_algorithm_runs(row['benchmark_id'])
                if algo_runs:
                    run_id = algo_runs[0]['run_id']
                    evals = await db.get_query_evaluations(run_id)
                    if evals and len(evals) > 0 and evals[0]['query_id'].startswith(dataset):
                        benchmarks_to_show.append(row)
                        break  # Only get the most recent one
        else:
            # Get latest benchmark overall
            row = await db.pool.fetchrow(
                """
                SELECT benchmark_id, dataset_size, created_at, metadata
                FROM benchmark_runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            if row:
                benchmarks_to_show.append(row)

        if not benchmarks_to_show:
            print("No benchmark runs found")
            return

        # Display each benchmark
        for idx, benchmark_row in enumerate(benchmarks_to_show):
            if idx > 0:
                print("\n" + "="*80 + "\n")

            benchmark_id = benchmark_row['benchmark_id']
            dataset_size = benchmark_row['dataset_size']
            created_at = benchmark_row['created_at']

            print(f"📊 Benchmark: {benchmark_id}")
            print(f"   Dataset Size: {dataset_size:,} queries")
            print(f"   Started: {created_at}")
            print()

            # Get algorithm runs
            algo_runs = await db.get_algorithm_runs(benchmark_id)

            if not algo_runs:
                print(f"No algorithm runs found for benchmark {benchmark_id}")
                continue

            print(f"{'='*80}")
            print(f"BENCHMARK PROGRESS: {benchmark_id}")
            print(f"{'='*80}\n")

            # Summary table
            print(f"{'Algorithm':<25} {'Queries':<12} {'Cost':<12} {'Quality':<10} {'Status':<10}")
            print("-" * 80)

            for algo in algo_runs:
                name = algo['algorithm_name']
                queries = algo['total_queries']
                cost = algo['total_cost']
                quality = algo['average_quality']
                completed = algo['completed_at']

                status = "✅ DONE" if completed else "🔄 RUNNING"

                print(f"{name:<25} {queries:>5} / {dataset_size:<5}  ${cost:>10.4f} {quality:>9.3f} {status:<10}")

            print()

            # Detailed per-algorithm analysis
            for algo in algo_runs:
                run_id = algo['run_id']
                name = algo['algorithm_name']
                queries = algo['total_queries']

                print(f"\n{'='*80}")
                print(f"📈 {name.upper()}")
                print(f"{'='*80}")

                # Get query evaluations
                evaluations = await db.get_query_evaluations(run_id)

                if not evaluations:
                    print("No evaluations yet")
                    continue

                # Calculate trends
                total_eval = len(evaluations)
                avg_cost = sum(e['cost'] for e in evaluations) / total_eval
                avg_quality = sum(e['quality_score'] for e in evaluations) / total_eval
                avg_latency = sum(e['latency'] for e in evaluations) / total_eval

                # Recent window (last 100)
                window_size = min(100, total_eval // 2)
                if window_size > 0:
                    recent = evaluations[-window_size:]
                    recent_cost = sum(e['cost'] for e in recent) / len(recent)
                    recent_quality = sum(e['quality_score'] for e in recent) / len(recent)
                    recent_latency = sum(e['latency'] for e in recent) / len(recent)
                else:
                    recent_cost = avg_cost
                    recent_quality = avg_quality
                    recent_latency = avg_latency

                # Model distribution
                model_counts = {}
                for e in evaluations:
                    model = e['model_id']
                    model_counts[model] = model_counts.get(model, 0) + 1

                print(f"\nQueries Processed: {total_eval:,}")
                print(f"\nCost:")
                print(f"  Total: ${algo['total_cost']:.4f}")
                print(f"  Avg per query: ${avg_cost:.6f}")
                print(f"  Recent avg (last {window_size}): ${recent_cost:.6f}")
                print(f"\nQuality:")
                print(f"  Overall avg: {avg_quality:.3f}")
                print(f"  Recent avg (last {window_size}): {recent_quality:.3f}")
                print(f"\nLatency:")
                print(f"  Avg: {avg_latency:.2f}s")
                print(f"  Recent avg: {recent_latency:.2f}s")
                print(f"\nModel Distribution:")
                for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = (count / total_eval * 100) if total_eval > 0 else 0
                    print(f"  {model}: {count:,} ({pct:.1f}%)")

                # Show last 5 evaluations
                print(f"\nRecent Evaluations:")
                print(f"  {'Query ID':<20} {'Model':<25} {'Quality':<8} {'Cost':<10}")
                print("  " + "-" * 70)
                for e in evaluations[-5:]:
                    print(f"  {e['query_id']:<20} {e['model_id']:<25} {e['quality_score']:<8.3f} ${e['cost']:<9.6f}")

            print(f"\n{'='*80}\n")

    finally:
        await db.disconnect()
        print("Disconnected from database")


async def main():
    parser = argparse.ArgumentParser(
        description="Query benchmark progress from PostgreSQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query_database.py                    # Latest benchmark
  python scripts/query_database.py --dataset gsm8k    # Latest GSM8K benchmark
  python scripts/query_database.py --dataset mmlu     # Latest MMLU benchmark
  python scripts/query_database.py --all              # All running benchmarks
  python scripts/query_database.py <benchmark-id>     # Specific benchmark by ID
        """
    )

    parser.add_argument(
        "benchmark_id",
        nargs="?",
        help="Specific benchmark ID to query"
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "mmlu"],
        help="Filter by dataset (shows latest benchmark for that dataset)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all running benchmarks"
    )

    args = parser.parse_args()

    await query_progress(
        benchmark_id=args.benchmark_id,
        dataset=args.dataset,
        show_all=args.all
    )


if __name__ == "__main__":
    asyncio.run(main())
