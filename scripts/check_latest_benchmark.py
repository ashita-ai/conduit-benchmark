#!/usr/bin/env python3
"""Check status and data quality of the most recent benchmark run."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conduit_bench.database import BenchmarkDatabase


async def main():
    db = BenchmarkDatabase()
    await db.connect()

    try:
        # Get most recent benchmark by finding the most recent algorithm run
        latest_run = await db.pool.fetchrow("""
            SELECT ar.run_id, ar.algorithm_name, ar.total_queries, ar.total_cost,
                   ar.average_quality, ar.started_at, ar.benchmark_id
            FROM algorithm_runs ar
            ORDER BY ar.started_at DESC
            LIMIT 1
        """)

        if not latest_run:
            print("No benchmark runs found in database")
            return

        benchmark_id = latest_run['benchmark_id']

        print("=" * 80)
        print(f"LATEST BENCHMARK: {benchmark_id}")
        print(f"Started: {latest_run['started_at']}")
        print("=" * 80)
        print()

        # Get all algorithm runs for this benchmark
        algo_runs = await db.pool.fetch("""
            SELECT run_id, algorithm_name, total_queries, total_cost, average_quality
            FROM algorithm_runs
            WHERE benchmark_id = $1
            ORDER BY algorithm_name
        """, benchmark_id)

        print(f"📊 ALGORITHM PROGRESS ({len(algo_runs)} algorithms)")
        print("-" * 80)

        total_evals = 0
        for run in algo_runs:
            # Count actual evaluations
            eval_count = await db.pool.fetchval(
                "SELECT COUNT(*) FROM query_evaluations WHERE run_id = $1",
                run['run_id']
            )
            total_evals += eval_count

            # Get latest evaluation time
            latest_eval = await db.pool.fetchrow("""
                SELECT created_at, query_id, model_id, quality_score, cost
                FROM query_evaluations
                WHERE run_id = $1
                ORDER BY created_at DESC
                LIMIT 1
            """, run['run_id'])

            if latest_eval:
                elapsed = datetime.now(latest_eval['created_at'].tzinfo) - latest_run['started_at']
                elapsed_sec = elapsed.total_seconds()
                rate = eval_count / elapsed_sec if elapsed_sec > 0 else 0

                print(f"{run['algorithm_name']:<30} {eval_count:>4} evals "
                      f"({rate:.2f}/s) | Latest: {latest_eval['created_at'].strftime('%H:%M:%S')}")

        print(f"\nTotal evaluations across all algorithms: {total_evals}")
        print()

        # Data quality checks
        print("🔍 DATA QUALITY CHECKS")
        print("-" * 80)

        # Check for duplicates
        duplicates = await db.pool.fetch("""
            SELECT run_id, query_id, COUNT(*) as cnt
            FROM query_evaluations
            GROUP BY run_id, query_id
            HAVING COUNT(*) > 1
        """)

        if duplicates:
            print(f"⚠️  WARNING: Found {len(duplicates)} duplicate (run_id, query_id) pairs!")
            for dup in duplicates[:5]:
                print(f"   run_id={dup['run_id'][:12]}... query_id={dup['query_id']} count={dup['cnt']}")
        else:
            print("✅ No duplicates - composite PK working correctly")

        # Check for anomalies
        anomalies = await db.pool.fetch("""
            SELECT run_id, query_id, quality_score, cost, latency
            FROM query_evaluations
            WHERE quality_score < 0 OR quality_score > 1
               OR cost < 0 OR cost > 1
               OR latency < 0 OR latency > 300
            LIMIT 10
        """)

        if anomalies:
            print(f"⚠️  WARNING: Found {len(anomalies)} anomalies!")
            for a in anomalies:
                print(f"   {a['query_id']}: Q={a['quality_score']:.2f} Cost=${a['cost']:.4f} Latency={a['latency']:.2f}s")
        else:
            print("✅ No anomalous values detected")

        print()

        # Recent activity
        print("📝 RECENT EVALUATIONS (last 10)")
        print("-" * 80)

        recent = await db.pool.fetch("""
            SELECT qe.created_at, ar.algorithm_name, qe.query_id, qe.model_id,
                   qe.quality_score, qe.cost, qe.latency
            FROM query_evaluations qe
            JOIN algorithm_runs ar ON qe.run_id = ar.run_id
            WHERE ar.benchmark_id = $1
            ORDER BY qe.created_at DESC
            LIMIT 10
        """, benchmark_id)

        for r in recent:
            print(f"{r['created_at'].strftime('%H:%M:%S')} | "
                  f"{r['algorithm_name']:<25} | "
                  f"{r['query_id']:<18} | "
                  f"{r['model_id']:<20} | "
                  f"Q={r['quality_score']:.1f} "
                  f"${r['cost']:.4f} "
                  f"{r['latency']:.2f}s")

        print()

        # Model distribution
        print("🤖 MODEL USAGE")
        print("-" * 80)

        model_stats = await db.pool.fetch("""
            SELECT qe.model_id, COUNT(*) as count,
                   AVG(qe.quality_score) as avg_quality,
                   AVG(qe.cost) as avg_cost
            FROM query_evaluations qe
            JOIN algorithm_runs ar ON qe.run_id = ar.run_id
            WHERE ar.benchmark_id = $1
            GROUP BY qe.model_id
            ORDER BY count DESC
        """, benchmark_id)

        for m in model_stats:
            print(f"{m['model_id']:<25} {m['count']:>4} uses | "
                  f"Avg Q={m['avg_quality']:.3f} | "
                  f"Avg Cost=${m['avg_cost']:.5f}")

        print()
        print("=" * 80)

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
