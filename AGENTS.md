# AGENTS.md - AI Agent Guide

**Purpose**: Quick reference for working on Conduit Bench
**Last Updated**: 2025-11-19

---

## Quick Orientation

**Conduit Bench**: Benchmarking and continuous improvement system for Conduit LLM routing
**Stack**: Python 3.10+, Conduit, Loom, Arbiter, PydanticAI
**Purpose**: Validate Conduit's quality-aware routing through multi-round synthetic benchmarking

### Directory Structure

```
conduit-bench/
├── conduit_bench/              # Main package
│   ├── generators/             # Query generation
│   │   ├── synthetic.py        # Generate diverse queries
│   │   ├── categories.py       # 10 query categories
│   │   ├── complexity.py       # Complexity scoring
│   │   └── references.py       # Reference answer generation
│   ├── runners/                # Execution
│   │   ├── round.py            # Run benchmark round
│   │   ├── conduit_client.py   # Conduit API wrapper
│   │   └── batch.py            # Batch execution with progress
│   ├── analysis/               # Results analysis
│   │   ├── metrics.py          # Performance metrics
│   │   ├── convergence.py      # ML convergence analysis
│   │   ├── visualize.py        # Charts and dashboards
│   │   └── reports.py          # Markdown report generation
│   └── cli.py                  # Command-line interface
├── pipelines/                  # Loom evaluation pipelines
│   ├── round1_evaluate.yaml    # 100% evaluation
│   ├── round2_evaluate.yaml    # Targeted evaluation
│   └── round3_evaluate.yaml    # Final validation
├── data/                       # Datasets (git-ignored)
│   └── queries/                # Generated queries + references
├── results/                    # Results (git-ignored)
│   ├── round1/                 # Baseline results
│   ├── round2/                 # Targeted improvement
│   └── round3/                 # Final validation
├── scripts/                    # CLI scripts
│   ├── generate_queries.py
│   ├── run_round.py
│   └── analyze_results.py
├── tests/                      # Tests
├── docs/                       # Documentation
│   ├── METHODOLOGY.md
│   └── RESULTS.md
└── pyproject.toml              # Dependencies
```

---

## Critical Rules

### 1. No Fine-Tuning Required

**Rule**: Conduit Bench validates Thompson Sampling routing (not fine-tuned models)

**Why**:
- Thompson Sampling is provider-agnostic (routes to ANY model)
- No fine-tuning costs ($0 vs $100s)
- Fast iteration (update routing logic instantly)
- Flexible (can route to any provider/model)

**Phase 1 Goal**: Prove Thompson Sampling achieves 40-50% cost savings

**Future consideration**: Fine-tuning only if specific domain needs >95% quality and base models can't deliver

### 2. Multi-Round Evaluation Strategy

**Rule**: 3 rounds with progressive targeting

**Round 1 (5,000 queries):**
- 100% evaluation of all queries
- Establish baseline (cost, quality, latency)
- Thompson Sampling learns initial patterns

**Round 2 (1,000 queries):**
- Target low-confidence areas
- Categories with <70% quality
- Edge cases (very simple/complex)

**Round 3 (500 queries):**
- Final validation
- Multi-model comparison
- High-stakes edge case stress testing

### 3. Loom + Arbiter Integration

**Rule**: Use Loom pipelines for batch evaluation, not inline evaluation

**Architecture:**
```
Conduit: Route queries → Store responses in DB
    ↓
Loom: Extract responses → Evaluate with Arbiter → Load scores
    ↓
Conduit: Read scores → Update Thompson Sampling bandit
```

**Why Loom**:
- Batch efficiency (evaluate 100s at once)
- Decoupled (Conduit doesn't depend on Arbiter)
- Audit trail (complete pipeline observability)
- Quality gates (Loom's built-in logic)

### 4. Type Safety (Strict Mypy)

All functions require type hints, no `Any` without justification.

### 5. Reproducible Datasets

**Rule**: All query generation must be reproducible with seed

```python
# Good
queries = await generate_dataset(total_queries=5000, seed=42)

# Bad
queries = await generate_dataset(total_queries=5000)  # Non-deterministic
```

### 6. Complete Features Only

If you start, you finish:
- ✅ Implementation complete
- ✅ Tests (>80% coverage)
- ✅ Docstrings
- ✅ Example usage
- ✅ Exported in `__init__.py`

---

## Development Workflow

### Before Starting

1. **Check dependencies:**
   ```bash
   # Ensure sibling projects exist
   ls ../conduit ../loom ../arbiter
   ```

2. **Create feature branch:**
   ```bash
   git checkout -b feature/query-generation
   ```

### During Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Type check
poetry run mypy conduit_bench/

# Format
poetry run black conduit_bench/
```

### Before Committing

```bash
poetry run pytest --cov=conduit_bench   # Tests pass
poetry run mypy conduit_bench/          # Type checking clean
poetry run ruff check conduit_bench/    # Linting clean
poetry run black conduit_bench/         # Formatted
```

---

## Tech Stack

### Core Dependencies
- **Python**: 3.10+ (modern type hints, async/await)
- **Conduit**: ML routing engine (sibling project)
- **Loom**: Pipeline orchestration (sibling project)
- **Arbiter**: Quality evaluation (sibling project)
- **PydanticAI**: LLM provider abstraction

### Data Processing
- **Pandas**: DataFrame operations
- **Polars**: Fast data processing (10-100x faster than pandas)

### LLM Providers (via PydanticAI)
- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini
- **Anthropic**: Claude 3.5 Sonnet, Claude Opus 4
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Groq**: Fast inference (Llama, Mixtral)

### Database
- **PostgreSQL**: Store queries, responses, evaluations
- **SQLAlchemy**: ORM with async support
- **Alembic**: Migrations

### Analysis & Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive dashboards

### CLI
- **Click**: Command-line interface
- **Rich**: Beautiful terminal output
- **tqdm**: Progress bars

---

## Key Patterns

### Query Generation Flow

```python
# 1. Define categories and complexity distribution
categories = {
    "technical_qa": {
        "complexity": {"simple": 0.3, "moderate": 0.5, "complex": 0.2}
    },
    "creative_writing": {
        "complexity": {"simple": 0.2, "moderate": 0.5, "complex": 0.3}
    },
    # ... 8 more categories
}

# 2. Generate queries (reproducible with seed)
queries = await generate_dataset(
    total_queries=5000,
    categories=categories,
    seed=42  # Reproducible
)

# 3. Generate reference answers (using GPT-4o for quality)
for query in queries:
    query.reference_answer = await generate_reference(
        query_text=query.text,
        model="gpt-4o"
    )
```

### Round Execution Flow

```python
# 1. Load queries
queries = load_queries("data/queries/round1_5000.jsonl")

# 2. Route through Conduit
conduit = ConduitClient()
for query in queries:
    decision = await conduit.route(Query(text=query.text))
    response = await conduit.execute(decision)
    save_result(query, decision, response)

# 3. Run Loom evaluation pipeline
await run_loom_pipeline("pipelines/round1_evaluate.yaml")

# 4. Update Conduit with feedback
await update_conduit_bandit(evaluations)

# 5. Analyze results
metrics = analyze_round_results("results/round1/")
generate_report(metrics)
```

### Loom Pipeline Pattern

```yaml
# pipelines/round1_evaluate.yaml
name: conduit_bench_round1
version: 1.0.0

extract:
  source: file://results/round1/routing_decisions.jsonl
  format: jsonl

evaluate:
  evaluators:
    - type: semantic
      reference_field: "reference_answer"
      threshold: 0.75
    - type: custom_criteria
      criteria: "Accurate, helpful, well-formatted"
      threshold: 0.7
  quality_gate: all_pass
  model: gpt-4o-mini
  batch_size: 50

load:
  destination: file://results/round1/evaluations.jsonl
  format: jsonl
```

---

## Code Quality Standards

### Docstrings

```python
async def generate_dataset(
    total_queries: int = 5000,
    categories: dict[str, dict] = CATEGORIES,
    seed: int = 42
) -> list[Query]:
    """Generate synthetic dataset with diversity and balance.

    Args:
        total_queries: Total number of queries to generate
        categories: Category definitions with complexity distributions
        seed: Random seed for reproducibility

    Returns:
        List of Query objects with text, category, complexity, reference

    Example:
        >>> queries = await generate_dataset(total_queries=1000, seed=42)
        >>> len(queries)
        1000
        >>> queries[0].category in CATEGORIES
        True
    """
```

---

## Common Tasks

### Generate Queries

```bash
poetry run python scripts/generate_queries.py \
    --count 5000 \
    --output data/queries/round1_5000.jsonl \
    --seed 42
```

### Run Benchmark Round

```bash
poetry run conduit-bench run --round 1
```

### Analyze Results

```bash
poetry run conduit-bench analyze --round 1
poetry run conduit-bench analyze --all-rounds  # Final report
```

### Visualize Convergence

```bash
poetry run python scripts/visualize_convergence.py \
    --rounds 1,2,3 \
    --output docs/convergence.png
```

---

## Environment Variables

```bash
# LLM Provider API Keys
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...       # NOTE: GEMINI_API_KEY (not GOOGLE_API_KEY)
GROQ_API_KEY=...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/conduit_bench

# Redis (for Conduit caching)
REDIS_URL=redis://localhost:6379

# Benchmarking
BENCHMARK_QUERY_COUNT=5000
BENCHMARK_EVALUATION_MODEL=gpt-4o-mini
```

---

## Expected Metrics

### After Round 1 (5,000 queries):
- Thompson Sampling accuracy: ~70-75%
- Cost baseline established
- Model preferences per category identified

### After Round 2 (6,000 total):
- Accuracy: ~85%
- Cost savings: ~35-40% vs always-gpt-4o
- Confidence scores improved

### After Round 3 (6,500 total):
- **Convergence**: ML model stable
- **Quality**: 95%+ queries score >0.85
- **Cost savings**: 40-50% vs baseline
- **Model preferences**: Learned per category

### Key Performance Indicators
- **Convergence point**: Number of queries until accuracy plateaus
- **Cost-quality frontier**: Optimal balance of cost and quality
- **Category preferences**: Best model per query type
- **Confidence calibration**: How well confidence predicts quality

---

## Testing Strategy

### Unit Tests
```python
# Test query generation
def test_generate_dataset_reproducibility():
    queries1 = await generate_dataset(total_queries=100, seed=42)
    queries2 = await generate_dataset(total_queries=100, seed=42)
    assert queries1 == queries2

# Test complexity distribution
def test_complexity_distribution():
    queries = await generate_dataset(total_queries=1000, seed=42)
    simple = sum(1 for q in queries if q.complexity == "simple")
    assert 250 < simple < 350  # ~30% ± 5%
```

### Integration Tests
```python
# Test full round execution
async def test_round1_execution():
    # Generate queries
    queries = await generate_dataset(total_queries=100, seed=42)

    # Route through Conduit
    results = await run_routing(queries)

    # Evaluate with Loom
    evaluations = await run_loom_pipeline("pipelines/round1_evaluate.yaml")

    # Verify results
    assert len(evaluations) == len(queries)
    assert all(e.score >= 0 and e.score <= 1 for e in evaluations)
```

---

## Working with AI Agents

### Task Management
**TodoWrite enforcement (MANDATORY)**: For ANY task with 3+ distinct steps, use TodoWrite to track progress.

### Output Quality
**Full data display**: Show complete data structures, not summaries. Examples should display real output.

### Audience & Context Recognition
**Auto-detect technical audiences**: No marketing language in engineering contexts.

### Quality & Testing
**Test output quality, not just functionality**: Verify examples produce useful results.

### Workflow Patterns
**Iterate fast**: Ship → test → get feedback → fix.

### Git & Commit Hygiene
**Clean workflow**: Feature branches, meaningful commits.

---

## Quick Reference

### Run Full Benchmark
```bash
# Round 1: Baseline (5,000 queries)
poetry run conduit-bench run --round 1

# Round 2: Targeted (1,000 queries)
poetry run conduit-bench run --round 2

# Round 3: Validation (500 queries)
poetry run conduit-bench run --round 3

# Generate final report
poetry run conduit-bench analyze --all-rounds
```

### Development
```bash
# Tests
poetry run pytest --cov=conduit_bench

# Type check
poetry run mypy conduit_bench/

# Format
poetry run black conduit_bench/

# Lint
poetry run ruff check conduit_bench/
```

---

## Related Documents

- **[README.md](README.md)**: Project overview and quick start
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**: Benchmark design
- **[docs/RESULTS.md](docs/RESULTS.md)**: Final results and insights

## Related Projects

- **[Conduit](../conduit/)**: ML-powered LLM routing
- **[Loom](../loom/)**: AI pipeline orchestration
- **[Arbiter](../arbiter/)**: LLM evaluation framework

---

**Last Updated**: 2025-11-19
