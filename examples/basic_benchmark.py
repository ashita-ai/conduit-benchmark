"""Basic benchmark example demonstrating the conduit-bench system.

This example:
1. Creates a small synthetic dataset
2. Compares Thompson Sampling vs UCB1 algorithms
3. Generates a simple performance report
"""

import asyncio
import os
from conduit.engines.bandits import ModelArm, ThompsonSamplingBandit, UCB1Bandit
from conduit_bench.generators import SyntheticQueryGenerator
from conduit_bench.runners import BenchmarkRunner


# Define model arms (simplified set for demo)
DEMO_ARMS = [
    ModelArm(
        model_id="openai:gpt-4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
        cost_per_input_token=0.00015,
        cost_per_output_token=0.0006,
        expected_quality=0.85,
    ),
    ModelArm(
        model_id="anthropic:claude-3-haiku",
        provider="anthropic",
        model_name="claude-3-haiku-20240307",
        cost_per_input_token=0.00025,
        cost_per_output_token=0.00125,
        expected_quality=0.80,
    ),
]


async def main():
    """Run a basic benchmark comparison."""
    print("\n" + "=" * 60)
    print("CONDUIT BENCHMARK - Basic Example")
    print("=" * 60 + "\n")

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using mocked responses.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set. Using mocked responses.")

    print("\n📊 Step 1: Generating synthetic dataset...")
    generator = SyntheticQueryGenerator(seed=42)
    dataset = await generator.generate_simple(
        n_queries=10,
        categories=["technical", "math", "creative"],
    )
    print(f"✅ Generated {len(dataset)} queries across 3 categories\n")

    # Show sample queries
    print("Sample queries:")
    for i, query in enumerate(dataset[:3], 1):
        print(f"  {i}. [{query.category}] {query.query_text}")
    print()

    print("🤖 Step 2: Initializing bandit algorithms...")
    algorithms = [
        ThompsonSamplingBandit(arms=DEMO_ARMS),
        UCB1Bandit(arms=DEMO_ARMS, c=1.5),
    ]
    print(f"✅ Initialized {len(algorithms)} algorithms\n")

    print("🚀 Step 3: Running benchmark...")
    runner = BenchmarkRunner(algorithms=algorithms)
    result = await runner.run(dataset=dataset, show_progress=True)

    print("\n" + "=" * 60)
    print("📈 RESULTS SUMMARY")
    print("=" * 60 + "\n")

    for algo_run in result.algorithms:
        print(f"Algorithm: {algo_run.algorithm_name}")
        print(f"  Total Cost:       ${algo_run.total_cost:.4f}")
        print(f"  Average Quality:  {algo_run.average_quality:.3f}")
        print(f"  Total Queries:    {algo_run.total_queries}")
        print()

    # Show model selection distribution
    print("\n📊 Model Selection Distribution:")
    for algo_run in result.algorithms:
        print(f"\n{algo_run.algorithm_name}:")
        model_counts = {}
        for _, model_id in algo_run.selections:
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
        for model_id, count in model_counts.items():
            percentage = (count / algo_run.total_queries) * 100
            print(f"  {model_id}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ Benchmark complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
