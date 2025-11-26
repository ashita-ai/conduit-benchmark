"""Quick script to query and verify database results from the 5-query evaluation."""

import asyncio
import json
from conduit_bench.database import BenchmarkDatabase


async def main():
    db = BenchmarkDatabase()
    await db.connect()

    print("\n=== QUERYING DATABASE RESULTS ===\n")

    # Find the most recent benchmark run
    rows = await db.pool.fetch(
        """
        SELECT benchmark_id, dataset_size, created_at, metadata
        FROM benchmark_runs
        ORDER BY created_at DESC
        LIMIT 1
        """
    )

    if not rows:
        print("❌ No benchmark runs found in database")
        await db.disconnect()
        return

    benchmark = dict(rows[0])
    benchmark_id = benchmark["benchmark_id"]

    print(f"📊 BENCHMARK RUN")
    print(f"  ID: {benchmark_id}")
    print(f"  Dataset size: {benchmark['dataset_size']}")
    print(f"  Created: {benchmark['created_at']}")
    print(f"  Metadata: {benchmark['metadata']}")

    # Get algorithm runs for this benchmark
    algo_runs = await db.pool.fetch(
        """
        SELECT run_id, algorithm_name, total_cost, average_quality,
               total_queries, started_at, completed_at, metadata
        FROM algorithm_runs
        WHERE benchmark_id = $1
        ORDER BY started_at
        """,
        benchmark_id,
    )

    print(f"\n🤖 ALGORITHM RUNS ({len(algo_runs)} total)")
    for algo in algo_runs:
        print(f"\n  Algorithm: {algo['algorithm_name']}")
        print(f"  Run ID: {algo['run_id']}")
        print(f"  Total cost: ${algo['total_cost']:.4f}")
        print(f"  Avg quality: {algo['average_quality']:.3f}")
        print(f"  Total queries: {algo['total_queries']}")
        print(f"  Started: {algo['started_at']}")
        print(f"  Completed: {algo['completed_at']}")
        print(f"  Duration: {(algo['completed_at'] - algo['started_at']).total_seconds():.1f}s")

        # Get query evaluations for this algorithm run
        evals = await db.pool.fetch(
            """
            SELECT query_id, model_id, quality_score, cost, latency,
                   success, created_at, metadata
            FROM query_evaluations
            WHERE run_id = $1
            ORDER BY created_at
            """,
            algo["run_id"],
        )

        print(f"\n  📝 QUERY EVALUATIONS ({len(evals)} queries)")
        for i, eval in enumerate(evals, 1):
            print(f"\n    Query {i}:")
            print(f"      Query ID: {eval['query_id'][:8]}...")
            print(f"      Model: {eval['model_id']}")
            print(f"      Quality: {eval['quality_score']:.3f}")
            print(f"      Cost: ${eval['cost']:.6f}")
            print(f"      Latency: {eval['latency']:.2f}s")
            print(f"      Success: {eval['success']}")
            if eval["metadata"]:
                print(f"      Metadata: {eval['metadata']}")

    # Statistics
    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)

    stats = await db.pool.fetchrow(
        """
        SELECT
            COUNT(*) as total_evaluations,
            AVG(quality_score) as avg_quality,
            MIN(quality_score) as min_quality,
            MAX(quality_score) as max_quality,
            AVG(cost) as avg_cost,
            SUM(cost) as total_cost,
            AVG(latency) as avg_latency,
            COUNT(*) FILTER (WHERE success) as successful_queries
        FROM query_evaluations qe
        JOIN algorithm_runs ar ON qe.run_id = ar.run_id
        WHERE ar.benchmark_id = $1
        """,
        benchmark_id,
    )

    print(f"\nTotal evaluations: {stats['total_evaluations']}")
    print(f"Quality: avg={stats['avg_quality']:.3f}, min={stats['min_quality']:.3f}, max={stats['max_quality']:.3f}")
    print(f"Cost: total=${stats['total_cost']:.4f}, avg=${stats['avg_cost']:.6f}")
    print(f"Latency: avg={stats['avg_latency']:.2f}s")
    print(f"Success rate: {stats['successful_queries']}/{stats['total_evaluations']} ({100*stats['successful_queries']/stats['total_evaluations']:.1f}%)")

    # Check for quality score diversity (should NOT all be 0.500)
    quality_scores = await db.pool.fetch(
        """
        SELECT quality_score
        FROM query_evaluations qe
        JOIN algorithm_runs ar ON qe.run_id = ar.run_id
        WHERE ar.benchmark_id = $1
        ORDER BY quality_score
        """,
        benchmark_id,
    )

    unique_scores = set(row["quality_score"] for row in quality_scores)
    print(f"\nUnique quality scores: {len(unique_scores)}")
    print(f"Quality score distribution: {sorted([row['quality_score'] for row in quality_scores])}")

    if len(unique_scores) == 1 and 0.500 in unique_scores:
        print("⚠️  WARNING: All quality scores are 0.500 - reference answers may not be working")
    else:
        print("✅ Quality scores show variation - reference answers working correctly")

    await db.disconnect()
    print("\n✅ Database query complete\n")


if __name__ == "__main__":
    asyncio.run(main())
