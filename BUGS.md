# Known Bugs

## 🐛 UCB1 Infinity Serialization to PostgreSQL

**Status**: ✅ Fixed (2025-11-26)
**GitHub Issue**: [#20](https://github.com/ashita-ai/conduit-benchmark/issues/20)
**Severity**: High (blocks database persistence for UCB1 algorithm)
**Discovered**: 2025-11-25
**Fixed**: 2025-11-26
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

### Implemented Fix

**Solution**: Sanitize Infinity/NaN before JSON serialization (Option 1 - IMPLEMENTED)

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

**Implementation Details**:
- Added `sanitize_metadata()` function to `conduit_bench/database.py`
- Recursively converts `float("inf")` → `"Infinity"`, `float("-inf")` → `"-Infinity"`, `NaN` → `"NaN"`
- Applied in `create_benchmark_run()`, `create_algorithm_run()`, and `write_query_evaluation()`
- Preserves algorithm behavior while enabling database persistence

**Testing**: Verified with 3-query UCB1 benchmark - database writes successful, no errors.

---

**Alternative Option (Not Chosen)**: Change UCB1 to use large finite number instead of Infinity

Would modify `/Users/evan/Documents/gh/conduit/conduit/engines/bandits/ucb.py`:
```python
# Lines 166, 281 - Use large finite number
MAX_UCB_VALUE = 1e9  # Large but finite

if pulls == 0:
    ucb_values[model_id] = MAX_UCB_VALUE  # Instead of float("inf")
```

**Why Option 1 Chosen**:
- ✅ Fixes database serialization without changing algorithm behavior
- ✅ Handles all Infinity/NaN values automatically (not just UCB1)
- ✅ Keeps algorithm logic in Conduit library untouched
- ❌ Option 2 would require modifying core Conduit library and changing algorithm behavior

### Related Files
- `/Users/evan/Documents/gh/conduit-benchmark/conduit_bench/database.py` (database writes)
- `/Users/evan/Documents/gh/conduit-benchmark/conduit_bench/runners/benchmark_runner.py` (lines 218-219, 313-322)
- `/Users/evan/Documents/gh/conduit/conduit/engines/bandits/ucb.py` (root cause)

### Verification Completed ✅
1. ✅ Run benchmark with UCB1 - Successful with 3-query test
2. ✅ Verified no database error warnings - Clean execution
3. ✅ Database writes successful - "Database connected for streaming writes" confirmed
4. ✅ Metadata sanitization working - Infinity values converted to strings

**Status**: Bug resolved and tested. UCB1 algorithm now works with full database persistence.
