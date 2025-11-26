# Pre-Flight Checklist for Main Experiment

**Date:** 2025-11-26
**Experiment:** N=2,500 queries, 11 algorithms × 3 runs = 33 benchmarks
**Estimated Duration:** ~7 days parallel execution
**Estimated Cost:** ~$429

---

## ✅ COMPLETED (Pilot Study Validation)

### Infrastructure Validation
- ✅ **Database Persistence**: PostgreSQL streaming writes working
- ✅ **UCB1 Infinity Bug**: Fixed (Issue #20)
- ✅ **Metrics Module**: 84% test coverage, 26 tests passing
- ✅ **Visualization Module**: 81% test coverage, 22 tests passing
- ✅ **CLI Integration**: 9 tests passing, full pipeline working
- ✅ **Pilot Study**: N=200 completed successfully (4h15m runtime)

### Variance & Sample Size Validation
- ✅ **Quality Variance**: σ² = 0.046679, σ = 0.2161
- ✅ **N=2,500 Power Analysis**:
  - Standard bandits: **14.3x oversampling** (required N=175)
  - Contextual bandits: **2.0-2.9x oversampling** (required N=872-1,220)
- ✅ **Statistical Power**: Adequate for detecting d=0.3 effect size

### Resource Availability
- ✅ **Disk Space**: 55GB available (need ~0.5GB for 33 result files)
- ✅ **Memory**: Sufficient for parallel execution
- ✅ **API Access**: OpenAI, Anthropic, Google credentials configured

---

## ⚠️ ITEMS TO VERIFY BEFORE MAIN EXPERIMENT

### 1. **Algorithm Names - VERIFY COMPLETE LIST** 🟡 MOSTLY COMPLETE

Currently implemented in `conduit_bench/cli.py` (lines 259-275):
```python
algorithm_map = {
    # Standard (non-contextual) bandits
    "thompson": ThompsonSamplingBandit(DEFAULT_ARMS),
    "ucb1": UCB1Bandit(DEFAULT_ARMS, c=1.5),
    "epsilon": EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=0.1),
    # Contextual bandits
    "linucb": LinUCBBandit(DEFAULT_ARMS, alpha=1.0, feature_dim=5),
    "contextual_thompson": ContextualThompsonSamplingBandit(DEFAULT_ARMS, lambda_reg=1.0, feature_dim=5),
    "dueling": DuelingBandit(DEFAULT_ARMS, feature_dim=5),
    # Baselines
    "random": RandomBaseline(DEFAULT_ARMS, random_seed=42),
    "oracle": OracleBaseline(DEFAULT_ARMS),
    "always_best": AlwaysBestBaseline(DEFAULT_ARMS),
    "always_cheapest": AlwaysCheapestBaseline(DEFAULT_ARMS),
}
```

**Status for Issue #22:**
- ✅ `"thompson"` - ThompsonSamplingBandit
- ✅ `"ucb1"` - UCB1Bandit
- ✅ `"epsilon"` - EpsilonGreedyBandit
- ✅ `"linucb"` - LinUCBBandit (contextual, feature_dim=5)
- ✅ `"contextual_thompson"` - ContextualThompsonSamplingBandit (contextual, feature_dim=5)
- ✅ `"dueling"` - DuelingBandit (contextual, feature_dim=5)
- ✅ `"random"` - RandomBaseline
- ✅ `"oracle"` - OracleBaseline (requires 100% references)
- ✅ `"always_best"` - AlwaysBestBaseline
- ✅ `"always_cheapest"` - AlwaysCheapestBaseline
- ❌ `"hybrid"` - HybridRouter (NOT YET IMPLEMENTED in conduit package)

**Remaining Issue:**
- HybridRouter doesn't exist in the conduit package yet
- Would need custom implementation: UCB1 → LinUCB transition at 2000 queries
- **DECISION NEEDED**: Run experiment with 10 algorithms now, or implement HybridRouter first?

### 2. **Reference Answer Strategy** 🟡 IMPORTANT

**Pilot Study:** Used 25% reference probability
**Main Experiment:** Needs **100% references** for Oracle baseline

**Why 100% required:**
- Oracle baseline requires quality score for ALL 9 models per query
- Per-query oracle selects best model for each individual query
- Cannot compute true Oracle regret without complete model evaluation

**ACTION REQUIRED:**
```bash
# Issue #21: Generate dataset with 100% references
uv run conduit-bench generate --queries 2500 --seed 42 \
  --output data/main_2500.jsonl \
  --reference-probability 1.0  # ← CRITICAL: Must be 1.0
```

### 3. **Parallel Execution Strategy** 🟡 IMPORTANT

**Options:**

**Option A: Sequential Runs (Safest)**
```bash
# 11 algorithms × 3 seeds = 33 runs, sequential
for seed in 42 43 44; do
  for algo in thompson ucb1 epsilon linucb contextual_thompson dueling \
              random oracle always_best always_cheapest hybrid; do
    uv run conduit-bench run \
      --dataset data/main_2500.jsonl \
      --algorithms $algo \
      --output results/main/${algo}_seed${seed}.json \
      --seed $seed
  done
done
```
- **Time:** ~1,749 hours (~73 days) 😱
- **Pros:** Simple, no resource contention
- **Cons:** WAY too slow

**Option B: Parallel Algorithms, Sequential Seeds (RECOMMENDED)**
```bash
# Run all 11 algorithms in parallel for each seed
for seed in 42 43 44; do
  echo "=== Running seed $seed ==="

  # Launch all 11 in parallel with &
  uv run conduit-bench run --dataset data/main_2500.jsonl --algorithms thompson --output results/main/thompson_seed${seed}.json &
  uv run conduit-bench run --dataset data/main_2500.jsonl --algorithms ucb1 --output results/main/ucb1_seed${seed}.json &
  uv run conduit-bench run --dataset data/main_2500.jsonl --algorithms epsilon --output results/main/epsilon_seed${seed}.json &
  uv run conduit-bench run --dataset data/main_2500.jsonl --algorithms linucb --output results/main/linucb_seed${seed}.json &
  # ... (all 11 algorithms)

  # Wait for all 11 to complete before next seed
  wait

  echo "=== Seed $seed complete ==="
done
```
- **Time:** ~159 hours (~7 days) ✅
- **Pros:** Good parallelization, safe checkpointing by seed
- **Cons:** High resource usage

**Option C: Full Parallel (Risky)**
- Run all 33 simultaneously
- **Time:** ~53 hours (~2 days)
- **Risk:** Resource exhaustion, harder to recover from failures

**RECOMMENDATION:** Use **Option B** (parallel algorithms, sequential seeds)

### 4. **Checkpointing & Recovery** 🟡 IMPORTANT

**Current State:** Database streaming writes provide some recovery

**Gaps:**
- No way to resume partial algorithm runs
- If run fails at query 2,400/2,500, must restart from 0

**ACTION REQUIRED:**
1. Verify database persistence works for all 11 algorithms
2. Test recovery: Start run, kill it, verify DB has partial results
3. Document recovery procedure for failed runs

### 5. **Cost Monitoring** 🟡 IMPORTANT

**Pilot Study Costs:**
- Thompson: $0.5167 (200 queries)
- UCB1: $0.4947 (200 queries)
- **Average:** $0.5057 per 200 queries

**Main Experiment Estimate:**
- Per algorithm per seed: $0.5057 × (2,500/200) = $6.32
- 33 runs: $6.32 × 33 = **$208**

**Budget:** Issue #22 estimates $429 (2x safety margin) ✅

**ACTION REQUIRED:**
1. Set up cost monitoring during execution
2. Alert if costs exceed $15/run (2x expected)
3. Daily cost checks during 7-day run

### 6. **Error Handling** 🟢 NICE-TO-HAVE

**Potential Issues:**
- API rate limiting (429 errors)
- Provider outages (OpenAI/Anthropic/Google down)
- Out of memory for contextual bandits (large matrices)
- Database connection issues

**ACTION REQUIRED:**
1. Add retry logic with exponential backoff
2. Graceful degradation for provider failures
3. Memory monitoring for LinUCB matrix operations

---

## 📋 PRE-FLIGHT CHECKLIST

Run these checks before starting main experiment:

```bash
# 1. ✅ Verify disk space
df -h . | grep -E "Avail|/dev"

# 2. ⚠️ Test all algorithm names work
for algo in thompson ucb1 epsilon linucb contextual_thompson dueling \
            random oracle always_best always_cheapest hybrid; do
  echo "Testing: $algo"
  # This will fail if algorithm not in algorithm_map!
done

# 3. ✅ Verify database connection
psql $DATABASE_URL -c "SELECT 1" || echo "⚠️  Database not available"

# 4. ✅ Check API credentials
env | grep -E "OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY"

# 5. ✅ Verify pilot results readable
ls -lh results/pilot/run_seed42.json
python -c "import json; json.load(open('results/pilot/run_seed42.json'))"

# 6. ⚠️ Test single algorithm on small dataset (sanity check)
uv run conduit-bench run \
  --dataset data/pilot_200.jsonl \
  --algorithms thompson \
  --output results/test_sanity.json \
  --max-queries 10

# 7. ✅ Verify analyze command works
uv run conduit-bench analyze \
  --results results/test_sanity.json \
  --output analysis/test_sanity_metrics.json
```

---

## 🚀 READY TO PROCEED?

**IF all items marked ✅:**
- Proceed with Issue #21 (generate N=2,500 dataset)
- Then Issue #22 (run main experiment)

**IF any item marked ⚠️ or ❌:**
- **STOP** and address blocking issues first
- Most critical: Add missing 7 algorithms to `algorithm_map`
- Test contextual algorithms work before 7-day run

---

## 📞 ESCALATION

**If issues arise during 7-day run:**
1. Check database for partial results (may be recoverable)
2. Review logs for error patterns
3. Consider pausing and addressing systematic issues
4. Don't blindly restart - investigate root cause first
