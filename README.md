# Conduit Bench 🎯

**Benchmarking and continuous improvement system for Conduit LLM routing**

Synthetic dataset generation, multi-round evaluation, and iterative ML improvement to validate Conduit's quality-aware routing capabilities.

---

## Purpose

Validate Conduit's value proposition through systematic benchmarking:
- **Generate**: 5,000+ diverse synthetic queries across 10 categories
- **Route**: Run queries through Conduit with Thompson Sampling
- **Evaluate**: Use Loom + Arbiter for comprehensive quality assessment
- **Learn**: Update Conduit's ML model with evaluation feedback
- **Iterate**: Multi-round refinement targeting weak spots

**Goal**: Prove Conduit achieves 40-50% cost savings while maintaining 95%+ quality.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         ROUND 1: Baseline (5,000 queries)       │
│  Generate → Route → Evaluate → Learn           │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│       ROUND 2: Targeted (1,000 queries)         │
│  Target weak spots → Route → Evaluate → Learn  │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│       ROUND 3: Validation (500 queries)         │
│  Edge cases → Multi-model comparison → Report  │
└─────────────────────────────────────────────────┘
```

### Components

**conduit-bench** (this repo):
- Query generation (10 categories × 3 complexity levels)
- Round execution (batch routing + evaluation)
- Analysis and reporting (metrics, convergence, cost savings)

**Conduit** (routing engine):
- Thompson Sampling for model selection
- Query execution via PydanticAI
- Feedback integration

**Loom** (evaluation orchestration):
- Batch evaluation pipelines (Extract → Evaluate → Load)
- Quality gates (all_pass, majority_pass)
- Quarantine pattern for failures

**Arbiter** (quality assessment):
- Semantic similarity evaluation
- Custom criteria validation
- Cost tracking

---

## Quick Start

### Prerequisites

- Python 3.10+
- Conduit, Loom, Arbiter projects (sibling directories)
- PostgreSQL database
- LLM API keys (OpenAI, Anthropic, Google)

### Installation

```bash
# Clone repo
cd /Users/evan/Documents/gh
git clone https://github.com/yourusername/conduit-bench.git
cd conduit-bench

# Install dependencies
poetry install

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
poetry run alembic upgrade head
```

### Run Benchmark

```bash
# Round 1: Generate 5,000 queries and run baseline
poetry run conduit-bench run --round 1

# Round 2: Targeted improvement (1,000 queries)
poetry run conduit-bench run --round 2

# Round 3: Final validation (500 queries)
poetry run conduit-bench run --round 3

# Generate final report
poetry run conduit-bench analyze --all-rounds
```

---

## Repository Structure

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
│   │   └── batch.py            # Batch execution
│   ├── analysis/               # Results analysis
│   │   ├── metrics.py          # Performance metrics
│   │   ├── convergence.py      # ML convergence analysis
│   │   ├── visualize.py        # Charts and dashboards
│   │   └── reports.py          # Markdown reports
│   └── cli.py                  # Command-line interface
├── pipelines/                  # Loom evaluation pipelines
│   ├── round1_evaluate.yaml    # 100% evaluation
│   ├── round2_evaluate.yaml    # Targeted evaluation
│   └── round3_evaluate.yaml    # Final validation
├── data/                       # Datasets (git-ignored)
│   ├── queries/
│   │   ├── round1_5000.jsonl
│   │   ├── round2_1000.jsonl
│   │   └── round3_500.jsonl
│   └── categories.json
├── results/                    # Results (git-ignored)
│   ├── round1/
│   ├── round2/
│   └── round3/
├── scripts/                    # Utility scripts
│   ├── generate_queries.py
│   ├── run_round.py
│   └── analyze_results.py
├── tests/                      # Tests
├── docs/                       # Documentation
│   ├── METHODOLOGY.md
│   └── RESULTS.md
├── AGENTS.md                   # AI agent guide
├── README.md                   # This file
└── pyproject.toml              # Dependencies
```

---

## Query Categories

10 diverse categories with balanced complexity:

1. **technical_qa**: Programming, engineering, technical questions
2. **creative_writing**: Stories, poetry, creative content
3. **analytical**: Data analysis, reasoning, problem-solving
4. **conversational**: Casual chat, simple Q&A
5. **summarization**: Text summarization tasks
6. **translation**: Language translation
7. **code_generation**: Code writing and debugging
8. **mathematical**: Math problems and explanations
9. **factual**: Factual questions and research
10. **instructional**: Step-by-step guides and tutorials

**Complexity levels:**
- **Simple** (30%): Straightforward, single-step queries
- **Moderate** (50%): Multi-step reasoning required
- **Complex** (20%): Advanced analysis, deep expertise

---

## Expected Outcomes

### After Round 1 (5,000 queries):
- Thompson Sampling baseline (not random)
- Category preferences identified
- Accuracy: ~70-75%
- Cost baseline established

### After Round 2 (6,000 total):
- Weak spots improved
- Higher confidence scores
- Accuracy: ~85%
- Cost savings: ~35-40% vs always-gpt-4o

### After Round 3 (6,500 total):
- **Convergence achieved**
- **Quality: 95%+ queries score >0.85**
- **Cost savings: 40-50% vs baseline**
- **Model preferences learned per category**

---

## Metrics

### Performance Metrics
- **Cost savings**: % reduction vs baseline strategies
- **Quality maintained**: % of queries meeting threshold
- **Routing accuracy**: % of queries routed optimally
- **Convergence**: Number of queries to ML stability

### Baseline Comparisons
- **Always GPT-4o**: Highest quality, highest cost
- **Always GPT-4o-mini**: Lowest cost, lower quality
- **Random selection**: No intelligence
- **Conduit routing**: Our approach (Thompson Sampling)

### Analysis Outputs
- Convergence plots (accuracy over queries)
- Cost-quality frontier charts
- Model preference heatmaps (category × model)
- Confidence distribution histograms

---

## Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)**: Benchmark design and approach
- **[RESULTS.md](docs/RESULTS.md)**: Final results and insights
- **[AGENTS.md](AGENTS.md)**: AI agent collaboration guide

---

## Development

```bash
# Run tests
poetry run pytest

# Type checking
poetry run mypy conduit_bench/

# Linting
poetry run ruff check conduit_bench/

# Formatting
poetry run black conduit_bench/

# Run single round (dev)
poetry run python scripts/run_round.py --round 1 --limit 100
```

---

## Privacy & Data

**Status**: Private repository during development

**Data handling:**
- Synthetic queries only (no real user data)
- Reference answers generated by GPT-4o
- Results stored locally (not shared)

**Publishing strategy:**
- Methodology: Public (reproducible research)
- Aggregate results: Public (marketing data)
- Raw datasets: Private (competitive advantage)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Related Projects

- **[Conduit](https://github.com/yourusername/conduit)**: ML-powered LLM routing
- **[Loom](https://github.com/yourusername/loom)**: AI pipeline orchestration
- **[Arbiter](https://github.com/yourusername/arbiter)**: LLM evaluation framework

---

**Built to validate**: "Conduit achieves 40-50% cost savings while maintaining 95%+ quality" 🚀
