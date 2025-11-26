# Known Bugs

## 🐛 UCB1 Infinity Serialization to PostgreSQL

**Status**: Open
**GitHub Issue**: [#20](https://github.com/ashita-ai/conduit-benchmark/issues/20)
**Severity**: High (blocks database persistence for UCB1 algorithm)
**Discovered**: 2025-11-25
**Affects**: UCB1 algorithm, potentially all algorithms with Infinity/NaN values in metadata

### Description
UCB1 algorithm metadata contains `float("inf")` values in `arm_ucb_values` dictionary, which cannot be serialized to PostgreSQL JSONB format. This causes all database writes to fail for UCB1 algorithm runs.

### Root Cause
**File**: `/Users/evan/Documents/gh/conduit/conduit/engines/bandits/ucb.py`

**Lines 166, 281**: UCB calculation sets `float("inf")` for unpulled arms:
```python
# Line 166 - During arm selection
if pulls == 0:
    ucb_values[model_id] = float("inf")

# Line 281 - In get_stats()
else:
    ucb_values[model_id] = float("inf")  # Not yet pulled
```

**Line 298**: Infinity values included in metadata:
```python
return {
    **base_stats,
    "arm_ucb_values": ucb_values,  # <-- Contains Infinity!
    ...
}
```

### Impact
When `BenchmarkRunner` calls `algorithm.get_stats()` (line 218, 335), the returned metadata includes Infinity values. These get passed to:

1. **Database writes** (line 218, 219) → `database.create_algorithm_run(metadata=algorithm.get_stats())`
   - Fails with: `Error: Token "Infinity" is invalid`
   - PostgreSQL JSONB cannot serialize Infinity/NaN values

2. **JSON output files** → Works fine (Python json module handles Infinity)

### Error Messages
```
Warning: Failed to create algorithm run: Token "Infinity" is invalid
Warning: Failed to write query evaluation: insert or update on table "query_evaluations"
    violates foreign key constraint "query_evaluations_run_id_fkey"
```

### Reproduction
```bash
# Generate dataset with any size
uv run conduit-bench generate --queries 100 --output data/test.jsonl

# Run benchmark with UCB1 - observe database errors
uv run conduit-bench run --dataset data/test.jsonl --algorithms ucb1 --output results/test.json

# Check logs - will see "Failed to create algorithm run" warnings
# JSON file still works, but database has no records for this run
```

### Proposed Fix

**Option 1**: Sanitize Infinity/NaN before JSON serialization (recommended)

Add to `conduit_bench/database.py`:
```python
def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Sanitize metadata by converting Infinity/NaN to JSON-safe values."""
    if not metadata:
        return metadata

    import math

    def sanitize_value(v):
        if isinstance(v, float):
            if math.isinf(v):
                return "Infinity" if v > 0 else "-Infinity"
            if math.isnan(v):
                return "NaN"
        elif isinstance(v, dict):
            return {k: sanitize_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [sanitize_value(val) for val in v]
        return v

    return sanitize_value(metadata)
```

Use in database writes:
```python
await self.pool.execute(
    "INSERT INTO algorithm_runs (..., metadata) VALUES (..., $8::jsonb)",
    ...,
    json.dumps(sanitize_metadata(metadata)) if metadata else None,
)
```

**Option 2**: Change UCB1 to use large finite number instead of Infinity

Modify `/Users/evan/Documents/gh/conduit/conduit/engines/bandits/ucb.py`:
```python
# Lines 166, 281 - Use large finite number
MAX_UCB_VALUE = 1e9  # Large but finite

if pulls == 0:
    ucb_values[model_id] = MAX_UCB_VALUE  # Instead of float("inf")
```

**Trade-offs**:
- Option 1: Fixes database serialization without changing algorithm behavior (RECOMMENDED)
- Option 2: Changes algorithm slightly (Infinity vs large number), but affects core Conduit library

### Workaround
For now, UCB1 benchmarks work correctly in terms of algorithm logic and JSON output files. Only database persistence is broken. To continue testing:
```bash
# Run without database writes
uv run conduit-bench run --dataset data/test.jsonl --algorithms ucb1 \
    --output results/ucb1_test.json --no-db
```

Or check JSON output files directly instead of querying database.

### Related Files
- `/Users/evan/Documents/gh/conduit-benchmark/conduit_bench/database.py` (database writes)
- `/Users/evan/Documents/gh/conduit-benchmark/conduit_bench/runners/benchmark_runner.py` (lines 218-219, 313-322)
- `/Users/evan/Documents/gh/conduit/conduit/engines/bandits/ucb.py` (root cause)

### Testing After Fix
1. Run benchmark with UCB1
2. Verify no database error warnings
3. Query database to confirm records exist:
```sql
SELECT * FROM algorithm_runs WHERE algorithm_name = 'ucb1';
SELECT * FROM query_evaluations WHERE run_id IN
    (SELECT run_id FROM algorithm_runs WHERE algorithm_name = 'ucb1');
```
4. Verify metadata contains sanitized Infinity representations
