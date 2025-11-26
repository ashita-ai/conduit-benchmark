"""Core data models for conduit-bench.

Defines the data structures for queries, evaluations, and benchmark results.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    """A query to be benchmarked against bandit algorithms.

    Attributes:
        query_id: Unique query identifier
        query_text: The text of the query
        reference_answer: Reference answer from GPT-4o for evaluation
        metadata: Additional query metadata (may include category/complexity for analysis only)

    Note: category and complexity are NOT exposed as fields because Conduit
    wouldn't have access to them in production. They can be stored in metadata
    for analysis purposes only, but the benchmark must use QueryAnalyzer to
    determine features just like production Conduit does.
    """

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query_text: str = Field(..., min_length=1)
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryEvaluation(BaseModel):
    """Evaluation result for a model's response to a query.

    Attributes:
        query_id: Associated query ID
        model_id: Model that generated the response
        response_text: Model's response
        quality_score: Quality score from Arbiter (0-1 scale)
        cost: Actual cost in USD
        latency: Response time in seconds
        success: Whether execution succeeded
        error: Error message if execution failed
        metadata: Additional evaluation data
    """

    query_id: str
    model_id: str
    response_text: str
    quality_score: float = Field(..., ge=0.0, le=1.0)
    cost: float = Field(..., ge=0.0)
    latency: float = Field(..., ge=0.0)
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlgorithmRun(BaseModel):
    """Results from running a bandit algorithm on the benchmark dataset.

    Attributes:
        algorithm_name: Name of the algorithm (e.g., "ThompsonSampling")
        run_id: Unique identifier for this run
        total_queries: Number of queries processed
        total_cost: Total cost incurred in USD
        average_quality: Average quality score across all queries
        selections: List of (query_id, model_id) selections made
        feedback: List of QueryEvaluation results
        cumulative_regret: Cumulative regret over time (vs Oracle)
        started_at: Run start timestamp
        completed_at: Run completion timestamp
        metadata: Additional run metadata (hyperparameters, etc.)
    """

    algorithm_name: str
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    total_queries: int = Field(..., ge=0)
    total_cost: float = Field(..., ge=0.0)
    average_quality: float = Field(..., ge=0.0, le=1.0)
    selections: list[tuple[str, str]] = Field(default_factory=list)
    feedback: list[QueryEvaluation] = Field(default_factory=list)
    cumulative_regret: list[float] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Complete benchmark results comparing multiple algorithms.

    Attributes:
        benchmark_id: Unique benchmark identifier
        dataset_size: Number of queries in the dataset
        algorithms: List of algorithm runs
        oracle_cost: Theoretical minimum cost (Oracle baseline)
        oracle_quality: Theoretical maximum quality (Oracle baseline)
        created_at: Benchmark creation timestamp
        metadata: Additional benchmark metadata
    """

    benchmark_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_size: int = Field(..., ge=0)
    algorithms: list[AlgorithmRun] = Field(default_factory=list)
    oracle_cost: float | None = None
    oracle_quality: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_algorithm(self, algorithm_name: str) -> AlgorithmRun | None:
        """Get results for a specific algorithm."""
        for algo in self.algorithms:
            if algo.algorithm_name == algorithm_name:
                return algo
        return None
