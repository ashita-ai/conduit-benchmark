#!/usr/bin/env python3
"""
Show quality and cost comparison across algorithms.
Calculates metrics directly from query_evaluations for accuracy.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from conduit_bench.database import BenchmarkDatabase


async def show_quality_comparison(benchmark_id: str | None = None):
    """Show concise quality and cost comparison."""

    db = BenchmarkDatabase()

    try:
        await db.connect()

        # Get latest benchmark if ID not provided
        if not benchmark_id:
            row = await db.pool.fetchrow(
                """
                SELECT benchmark_id, dataset_size, created_at
                FROM benchmark_runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            if not row:
                print("No benchmark runs found in database")
                return

            benchmark_id = row['benchmark_id']
            print(f"📊 Latest Benchmark: {benchmark_id}")
            print(f"   Dataset Size: {row['dataset_size']:,} queries")
            print(f"   Started: {row['created_at']}")
            print()

        # Get algorithm runs
        algo_runs = await db.get_algorithm_runs(benchmark_id)

        if not algo_runs:
            print(f"No algorithm runs found for benchmark {benchmark_id}")
            return

        print(f"{'='*90}")
        print(f"QUALITY & COST COMPARISON")
        print(f"{'='*90}")
        print(f"{'Algorithm':<22} {'Queries':<10} {'Quality':<10} {'Avg Cost':<12} {'Top Model Selection':<30}")
        print("-" * 90)

        results = []
        for algo in algo_runs:
            run_id = algo['run_id']
            name = algo['algorithm_name']

            # Get query evaluations to calculate metrics
            evaluations = await db.get_query_evaluations(run_id)

            if not evaluations:
                results.append({
                    'name': name,
                    'queries': 0,
                    'quality': 0.0,
                    'cost': 0.0,
                    'model_info': 'No data yet'
                })
                continue

            # Calculate metrics from evaluations
            total_queries = len(evaluations)
            avg_quality = sum(e['quality_score'] for e in evaluations) / total_queries
            avg_cost = sum(e['cost'] for e in evaluations) / total_queries

            # Get model distribution
            model_counts = {}
            for e in evaluations:
                model = e['model_id']
                model_counts[model] = model_counts.get(model, 0) + 1

            # Find top model
            top_model, top_count = max(model_counts.items(), key=lambda x: x[1])
            top_pct = (top_count / total_queries * 100) if total_queries > 0 else 0
            model_info = f"{top_model} ({top_count}/{total_queries} = {top_pct:.1f}%)"

            results.append({
                'name': name,
                'queries': total_queries,
                'quality': avg_quality,
                'cost': avg_cost,
                'model_info': model_info
            })

        # Sort by queries processed (descending)
        results.sort(key=lambda x: x['queries'], reverse=True)

        # Print results
        for r in results:
            queries_str = f"{r['queries']}"
            quality_str = f"{r['quality']:.3f}" if r['queries'] > 0 else "N/A"
            cost_str = f"${r['cost']:.6f}" if r['queries'] > 0 else "N/A"

            print(f"{r['name']:<22} {queries_str:<10} {quality_str:<10} {cost_str:<12} {r['model_info']:<30}")

        print()

        # Show key insights
        if results and results[0]['queries'] > 0:
            print("📈 Key Insights:")
            print()

            # Quality leader
            quality_leader = max((r for r in results if r['queries'] > 0), key=lambda x: x['quality'])
            print(f"   🏆 Highest Quality: {quality_leader['name']} ({quality_leader['quality']:.3f})")

            # Cost efficiency
            cost_leader = min((r for r in results if r['queries'] > 0), key=lambda x: x['cost'])
            print(f"   💰 Lowest Cost: {cost_leader['name']} (${cost_leader['cost']:.6f} per query)")

            # Model selection for baselines
            for r in results:
                if r['name'] == 'always_best' and r['queries'] > 0:
                    print(f"   ✅ AlwaysBest selecting: {r['model_info'].split('(')[0].strip()}")
                elif r['name'] == 'always_cheapest' and r['queries'] > 0:
                    print(f"   ✅ AlwaysCheapest selecting: {r['model_info'].split('(')[0].strip()}")

            print()

    finally:
        await db.disconnect()


async def main():
    benchmark_id = sys.argv[1] if len(sys.argv) > 1 else None
    await show_quality_comparison(benchmark_id)


if __name__ == "__main__":
    asyncio.run(main())
