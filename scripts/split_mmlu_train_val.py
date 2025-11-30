#!/usr/bin/env python3
"""Split MMLU dataset into train (1000) and validation (319) sets.

Creates reproducible splits using seed=42 for:
- Training: 1,000 queries for algorithm learning
- Validation: 319 queries for holdout evaluation

Saves to data/ directory as JSON files.
"""
import json
from pathlib import Path

from conduit_bench.datasets.mmlu import MMLULoader


def main():
    """Create train/validation split from MMLU test set."""
    print("Loading MMLU test set (1,319 total queries)...")

    # Load full 1,319 queries with seed=42 for reproducibility
    loader = MMLULoader()
    all_queries = loader.load(split="test", limit=1319, seed=42)

    print(f"Loaded {len(all_queries)} queries")

    # Split into train (1000) and validation (319)
    train_queries = all_queries[:1000]
    val_queries = all_queries[1000:]

    print(f"\nTrain set: {len(train_queries)} queries")
    print(f"Validation set: {len(val_queries)} queries")

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save train split
    train_path = data_dir / "mmlu_train_1000.json"
    with open(train_path, 'w') as f:
        json.dump(
            {
                "dataset": "mmlu",
                "split": "train",
                "size": len(train_queries),
                "seed": 42,
                "queries": [
                    {
                        "query_id": q.query_id,
                        "query_text": q.query_text,
                        "reference_answer": q.reference_answer,
                        "metadata": q.metadata,
                    }
                    for q in train_queries
                ]
            },
            f,
            indent=2
        )
    print(f"\n✅ Train set saved to: {train_path}")

    # Save validation split
    val_path = data_dir / "mmlu_validation_319.json"
    with open(val_path, 'w') as f:
        json.dump(
            {
                "dataset": "mmlu",
                "split": "validation",
                "size": len(val_queries),
                "seed": 42,
                "queries": [
                    {
                        "query_id": q.query_id,
                        "query_text": q.query_text,
                        "reference_answer": q.reference_answer,
                        "metadata": q.metadata,
                    }
                    for q in val_queries
                ]
            },
            f,
            indent=2
        )
    print(f"✅ Validation set saved to: {val_path}")

    # Verify no overlap
    train_ids = {q.query_id for q in train_queries}
    val_ids = {q.query_id for q in val_queries}
    overlap = train_ids & val_ids

    print(f"\n✅ Verification:")
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Validation IDs: {len(val_ids)}")
    print(f"  Overlap: {len(overlap)} (should be 0)")

    if overlap:
        print(f"  ❌ ERROR: Found overlapping IDs!")
    else:
        print(f"  ✅ No overlap - splits are valid!")

    print(f"\n📊 Subject distribution:")
    train_subjects = {}
    for q in train_queries:
        subject = q.metadata.get("subject", "unknown")
        train_subjects[subject] = train_subjects.get(subject, 0) + 1

    print(f"  Train set covers {len(train_subjects)} subjects")

    val_subjects = {}
    for q in val_queries:
        subject = q.metadata.get("subject", "unknown")
        val_subjects[subject] = val_subjects.get(subject, 0) + 1

    print(f"  Validation set covers {len(val_subjects)} subjects")

    print("\n🎯 Ready to run training benchmark on 1,000 train queries!")


if __name__ == "__main__":
    main()
