# Embedding and Feature Engineering

**Last Updated**: 2025-12-07

Documentation of embedding providers, feature extraction, PCA dimensionality reduction, and their relationship to contextual bandit convergence in conduit-benchmark.

---

## Table of Contents

1. [Overview](#overview)
2. [Embedding Providers](#embedding-providers)
3. [Feature Vector Structure](#feature-vector-structure)
4. [PCA Dimensionality Reduction](#pca-dimensionality-reduction)
5. [Convergence and Feature Dimensions](#convergence-and-feature-dimensions)
6. [Implementation References](#implementation-references)

---

## Overview

Feature engineering is critical for contextual bandit algorithms (LinUCB, Contextual Thompson Sampling, Dueling Bandit). Query features enable the algorithm to learn which models perform best for different types of queries.

### Key Components

1. **Embedding**: Dense vector representation of query semantics
2. **Metadata**: Token count and complexity score
3. **PCA**: Optional dimensionality reduction for faster convergence

### Feature Vector Formula

$$\mathbf{x} = [\underbrace{e_1, e_2, \ldots, e_d}_{\text{embedding}}, \underbrace{t, c}_{\text{metadata}}]$$

Where:
- $e_i$: Embedding dimensions (provider-specific)
- $t$: Token count (normalized)
- $c$: Complexity score (0.0-1.0)

---

## Embedding Providers

### Supported Providers

| Provider | Model | Dimension | API Key | Installation | Notes |
|----------|-------|-----------|---------|--------------|-------|
| **OpenAI** | text-embedding-3-small | 1536 | Required | `pip install openai` | Recommended for production |
| **OpenAI** | text-embedding-3-large | 3072 | Required | `pip install openai` | Higher quality, higher cost |
| **Cohere** | embed-english-v3.0 | 1024 | Required | `pip install cohere` | Fast, high quality |
| **FastEmbed** | BAAI/bge-small-en-v1.5 | 384 | Not needed | `pip install fastembed` | Lightweight ONNX (~100MB) |
| **FastEmbed** | BAAI/bge-base-en-v1.5 | 768 | Not needed | `pip install fastembed` | Better quality |
| **sentence-transformers** | all-MiniLM-L6-v2 | 384 | Not needed | `pip install sentence-transformers` | Full PyTorch (~2GB) |

### Auto-Detection Priority

When `provider="auto"` (default), the system tries providers in order:

```
1. OpenAI (if OPENAI_API_KEY exists) - Fast, high quality
2. Cohere (if COHERE_API_KEY exists) - Fast, high quality
3. FastEmbed (if installed) - Lightweight, no API key
4. sentence-transformers (if installed) - Full PyTorch, no API key
5. Error with installation instructions
```

### Provider Selection

```python
from conduit.engines.embeddings.factory import create_embedding_provider

# Auto-detection (recommended for flexibility)
provider = create_embedding_provider("auto")

# Explicit OpenAI (recommended for production)
provider = create_embedding_provider(
    "openai",
    model="text-embedding-3-small",
    api_key="sk-..."
)

# Local embeddings (no API key needed)
provider = create_embedding_provider("fastembed")
```

### Dimension Summary

| Provider | Default Model | Embedding Dim | Full Feature Dim |
|----------|---------------|---------------|------------------|
| OpenAI | text-embedding-3-small | 1536 | 1538 |
| OpenAI | text-embedding-3-large | 3072 | 3074 |
| Cohere | embed-english-v3.0 | 1024 | 1026 |
| FastEmbed | BAAI/bge-small-en-v1.5 | 384 | 386 |
| sentence-transformers | all-MiniLM-L6-v2 | 384 | 386 |

*Full Feature Dim = Embedding Dim + 2 metadata fields*

---

## Feature Vector Structure

### Components

The feature vector combines semantic embeddings with query metadata:

```python
@dataclass
class QueryFeatures:
    embedding: list[float]    # Semantic embedding vector
    token_count: int          # Estimated token count
    complexity_score: float   # 0.0-1.0 complexity rating
    query_text: str           # Original query (for debugging)
```

### Feature Extraction Pipeline

```
Query Text
    │
    ▼
┌─────────────────┐
│ Embedding       │───► Dense vector (384-3072 dims)
│ Provider        │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Token Count     │───► int (words × 1.3)
│ Estimation      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Complexity      │───► float (0.0-1.0)
│ Scoring         │
└─────────────────┘
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│ PCA Transform   │────►│ Reduced         │ (optional)
│ (if enabled)    │     │ embedding       │
└─────────────────┘     └─────────────────┘
    │
    ▼
QueryFeatures(embedding, token_count, complexity_score)
```

### Complexity Score Calculation

The complexity score combines multiple heuristics:

| Factor | Weight | Condition |
|--------|--------|-----------|
| Length | 0.1-0.3 | <20 tokens: 0.1, <50: 0.2, ≥50: 0.3 |
| Technical terms | 0.06 each | Keywords like "function", "algorithm", "SQL" |
| Code blocks | 0.06 | Backticks or code formatting |
| Multiple questions | 0.05 each | Each "?" after the first |
| Requirement indicators | 0.05 each | "must", "should", "require", etc. |

**Range**: 0.0 (simple) to 1.0 (highly complex)

### Token Count Estimation

$$\text{token\_count} = \lfloor \text{word\_count} \times 1.3 \rfloor$$

This approximation is used for quick estimation without tokenizer overhead.

---

## PCA Dimensionality Reduction

### Purpose

PCA (Principal Component Analysis) reduces embedding dimensions to:
1. **Accelerate convergence**: LinUCB requires $O(d)$ to $O(10d)$ samples
2. **Reduce memory**: Smaller A matrices and b vectors
3. **Improve generalization**: Remove noise from embeddings

### Variance Retention by Provider

| Provider | Original Dim | PCA Dim | Variance Retained | Recommendation |
|----------|-------------|---------|-------------------|----------------|
| OpenAI | 1536 | 64 | 57% | Too aggressive |
| OpenAI | 1536 | 128 | 73% | Acceptable |
| OpenAI | 1536 | 192 | 85% | Good balance |
| OpenAI | 1536 | 418 | 95% | High fidelity |
| FastEmbed | 384 | 64 | 95% | Excellent |
| FastEmbed | 384 | 128 | 99% | Nearly lossless |
| Cohere | 1024 | 64 | ~70% | Acceptable |
| Cohere | 1024 | 128 | ~85% | Good balance |

### PCA Configuration

```yaml
# conduit.yaml
embeddings:
  provider: auto
  model: null  # Use provider default
  pca:
    enabled: false  # Default: disabled
    components: 128  # Target dimensions
    auto_retrain: true
    retrain_threshold: 150
```

### PCA Fitting Process

```python
from conduit.engines.analyzer import QueryAnalyzer

# Create analyzer with PCA enabled
analyzer = QueryAnalyzer(
    embedding_provider_type="openai",
    use_pca=True,
    pca_dimensions=128,
    pca_model_path="models/pca.pkl"
)

# Fit PCA on representative queries (1000+ recommended)
training_queries = load_representative_queries()  # Diverse query set
await analyzer.fit_pca(training_queries)

# PCA is now fitted and saved to models/pca.pkl
# Future analyzer instances will auto-load the fitted model
```

### PCA Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Training queries | 100 | 1,000+ |
| Query diversity | All domains | All domains + complexity levels |
| Refitting frequency | Once | Monthly or on workload shift |

---

## Convergence and Feature Dimensions

### LinUCB Convergence Theory

LinUCB convergence depends on feature dimension $d$:

$$\text{Required samples} = O(d) \text{ to } O(10d)$$

| Feature Dim ($d$) | Min Samples | Conservative | Our Benchmark |
|-------------------|-------------|--------------|---------------|
| 386 (FastEmbed) | 386 | 3,860 | 1,000 |
| 1026 (Cohere) | 1,026 | 10,260 | 1,000 |
| 1538 (OpenAI) | 1,538 | 15,380 | 1,000 |
| 130 (PCA + 2) | 130 | 1,300 | 1,000 |

### Convergence Strategies

| Strategy | Feature Dim | Convergence | Trade-off |
|----------|-------------|-------------|-----------|
| **Raw embeddings** | 384-1538 | Slow | Maximum semantic information |
| **PCA reduction** | 66-130 | Fast | Some information loss |
| **Hybrid warm-start** | Any | Fast | UCB1 phase ignores features |

### Benchmark Configuration

Our benchmark uses hybrid warm-start for convergence reliability:

```yaml
# Hybrid Thompson → LinUCB
hybrid_routing:
  switch_threshold: 2000  # Switch to LinUCB after 2000 queries
  ucb1_c: 1.5
  linucb_alpha: 1.0
```

**Phase 1 (queries 1-2000)**: Thompson Sampling (non-contextual, fast learning)
**Phase 2 (queries 2001+)**: LinUCB (contextual, feature-based)

### Feature Dimension Impact on LinUCB

The LinUCB A matrix is $d \times d$ where $d$ is feature dimension:

| Feature Dim | A Matrix Size | Memory per Arm | 8 Arms Total |
|-------------|---------------|----------------|--------------|
| 386 | 386 × 386 | 1.2 MB | 9.5 MB |
| 1538 | 1538 × 1538 | 18.9 MB | 151 MB |
| 130 (PCA) | 130 × 130 | 135 KB | 1.1 MB |

**Memory formula**: $\text{Memory} = K \times d^2 \times 8$ bytes (float64)

---

## Implementation References

### File Locations

| Component | File | Function/Class |
|-----------|------|----------------|
| Factory | `conduit/engines/embeddings/factory.py` | `create_embedding_provider()` |
| Base interface | `conduit/engines/embeddings/base.py` | `EmbeddingProvider` |
| OpenAI | `conduit/engines/embeddings/openai.py` | `OpenAIEmbeddingProvider` |
| Cohere | `conduit/engines/embeddings/cohere.py` | `CohereEmbeddingProvider` |
| FastEmbed | `conduit/engines/embeddings/fastembed_provider.py` | `FastEmbedProvider` |
| Analyzer | `conduit/engines/analyzer.py` | `QueryAnalyzer` |
| Config | `conduit/core/config.py` | `load_embeddings_config()` |

### Configuration Sources

1. **conduit.yaml** (primary)
   ```yaml
   embeddings:
     provider: auto
     model: null
     pca:
       enabled: false
       components: 128
   ```

2. **Environment variables** (fallback)
   ```bash
   EMBEDDING_PROVIDER=openai
   EMBEDDING_MODEL=text-embedding-3-small
   USE_PCA=false
   PCA_COMPONENTS=128
   ```

3. **Hardcoded defaults** (ultimate fallback)
   - Provider: auto
   - PCA: disabled
   - Components: 128

---

## Appendix A: Provider Comparison

### Performance Characteristics

| Provider | Latency | Cost | Quality | Offline |
|----------|---------|------|---------|---------|
| OpenAI | ~100ms | $0.02/1M | High | No |
| Cohere | ~150ms | $0.10/1M | High | No |
| FastEmbed | ~50ms | Free | Good | Yes |
| sentence-transformers | ~100ms | Free | Good | Yes |

### Quality Benchmarks (MTEB)

| Model | MTEB Score | Dimension | Notes |
|-------|------------|-----------|-------|
| text-embedding-3-large | 64.6 | 3072 | Best quality |
| text-embedding-3-small | 62.3 | 1536 | Best balance |
| embed-english-v3.0 | 64.5 | 1024 | Cohere flagship |
| bge-small-en-v1.5 | 51.7 | 384 | Good for size |
| all-MiniLM-L6-v2 | 56.3 | 384 | Popular open-source |

---

## Appendix B: Feature Engineering Best Practices

### For Production

1. **Use OpenAI embeddings** for consistency with LLM routing
2. **Enable PCA** if using 1000+ dim embeddings
3. **Cache embeddings** via Redis for performance
4. **Fit PCA** on representative production traffic

### For Benchmarking

1. **Use consistent provider** across all algorithms
2. **Disable PCA** for fair comparison (or enable for all)
3. **Document embedding config** in results metadata
4. **Consider hybrid warm-start** for 1000-query benchmarks

### For Low-Resource Environments

1. **Use FastEmbed** (100MB vs 2GB)
2. **384-dim embeddings** converge faster
3. **Skip PCA** if embedding dim already small
4. **Serverless-friendly** (Lambda, Cloud Run)
