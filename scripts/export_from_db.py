#!/usr/bin/env python3
"""Export benchmark from database to JSON format."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conduit_bench.database import BenchmarkDatabase
from conduit_bench.benchmark_models import AlgorithmRun, BenchmarkResult, QueryEvaluation


async def export_to_json(benchmark_id: str, output_file: str):
    db = BenchmarkDatabase()
    await db.connect()

    # Get benchmark info
    benchmark = await db.get_benchmark_run(benchmark_id)
    if not benchmark:
        print(f'Benchmark {benchmark_id} not found')
        return

    # Get algorithm runs
    algo_runs_data = await db.get_algorithm_runs(benchmark_id)

    algorithms = []
    for algo in algo_runs_data:
        # Get evaluations for this run
        evals_data = await db.get_query_evaluations(algo['run_id'])

        evaluations = []
        for e in evals_data:
            metadata = e.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            evaluations.append(QueryEvaluation(
                query_id=e['query_id'],
                model_id=e['model_id'],
                response_text=metadata.get('response_text', '') if metadata else '',
                quality_score=e['quality_score'],
                cost=e['cost'],
                latency=e['latency'],
                success=e['success'],
                error=metadata.get('error_details') if metadata else None
            ).model_dump())

        # Parse algorithm metadata (also stored as JSON string in database)
        algo_metadata = algo.get('metadata', {})
        if isinstance(algo_metadata, str):
            algo_metadata = json.loads(algo_metadata)

        algorithms.append(AlgorithmRun(
            algorithm_name=algo['algorithm_name'],
            run_id=algo['run_id'],
            total_queries=algo['total_queries'],
            total_cost=algo['total_cost'],
            average_quality=algo['average_quality'],
            selections=[(e['query_id'], e['model_id']) for e in evals_data],
            feedback=evaluations,
            cumulative_cost=[sum(eval['cost'] for eval in evals_data[:i+1]) for i in range(len(evals_data))],
            converged=False,
            convergence_point=None,
            started_at=algo['started_at'],
            completed_at=algo['completed_at'],
            metadata=algo_metadata
        ).model_dump())

    result = BenchmarkResult(
        benchmark_id=benchmark['benchmark_id'],
        dataset_size=benchmark['dataset_size'],
        algorithms=algorithms
    )

    with open(output_file, 'w') as f:
        json.dump(result.model_dump(), f, indent=2, default=str)

    print(f'Exported {len(algorithms)} algorithms to {output_file}')
    for algo in algorithms:
        print(f"  - {algo['algorithm_name']}: {algo['total_queries']} queries, cost=${algo['total_cost']:.4f}, quality={algo['average_quality']:.3f}")

    await db.disconnect()


if __name__ == "__main__":
    benchmark_id = sys.argv[1] if len(sys.argv) > 1 else '3cb61c4f-4a7e-48cf-b9a1-17a3a478e95b'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'results/mmlu_1000_from_db.json'

    asyncio.run(export_to_json(benchmark_id, output_file))
