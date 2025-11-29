# Prompt for Generating Synthetic Programming Dataset

## Task

Generate a JSONL file named `synthetic_{{MODEL_NAME}}.jsonl` containing 100 programming questions with code answers, following the exact format shown in the reference example.

## Reference Example

See `/Users/evan/Documents/gh/conduit-benchmark/data/synthetic_composer_example.jsonl` for the exact format.

## Requirements

### Format
Each line must be a valid JSON object with these exact fields:
- `query_id`: Unique identifier (e.g., "synthetic-001")
- `query_text`: The programming question that:
  - Specifies the language to use (e.g., "Language: Python")
  - Specifies the function name (e.g., "Function name: find_longest_palindrome")
  - Describes what the function should do
- `reference_answer`: The complete, runnable code solution (code only, no explanations)
- `assertions`: Runnable test code that verifies correctness (can be executed to test the answer)
- `metadata`: Object containing:
  - `language`: Programming language name
  - `function_name`: Name of the function to implement
  - `difficulty`: One of "medium", "advanced", or "very_advanced"
  - `category`: Problem category (e.g., "algorithms", "data_structures", "concurrency")
  - `complexity`: Float between 0.0 and 1.0

### Language Distribution (100 total)
- **40 Python** (40%)
- **40 TypeScript** (40%)
- **7 Go** (7%)
- **3 Rust** (3%)
- **10 Other languages** (10%) - Choose from: Java, JavaScript, C++, Ruby, Swift, Kotlin, C#, PHP, Scala, Haskell

### Difficulty Distribution
- Evenly distribute across "medium", "advanced", and "very_advanced"
- Approximately 33-34 problems per difficulty level

### Quality Requirements
1. **Unique problems**: Each question should be different and non-trivial
2. **Runnable code**: All code must be syntactically correct and executable
3. **Testable assertions**: Assertions must be runnable and verify correctness
4. **Diverse topics**: Cover algorithms, data structures, concurrency, async, error handling, design patterns, etc.
5. **Real-world relevance**: Problems should be practical and test real programming skills

### Code Requirements
- **Code only**: The `reference_answer` field should contain ONLY the code, no explanations or comments (unless necessary for clarity)
- **Complete solutions**: Code must be complete and runnable
- **Proper syntax**: All code must be syntactically correct for the specified language
- **Edge cases**: Solutions should handle edge cases appropriately

### Assertion Requirements
- **Runnable**: Assertions must be executable code that tests the function
- **Comprehensive**: Test multiple cases including edge cases
- **Language-appropriate**: Use the correct assertion syntax for each language:
  - Python: `assert function_name(args) == expected`
  - TypeScript/JavaScript: `if (functionName(args) !== expected) throw new Error("Failed")`
  - Go: `if FunctionName(args) != expected { panic("Failed") }`
  - Rust: `assert_eq!(function_name(args), expected);`
  - Other languages: Use appropriate assertion syntax

## Example Entry Format

```json
{
  "query_id": "synthetic-001",
  "query_text": "Write a Python function named `find_longest_palindrome` that takes a string and returns the longest palindromic substring. The function should handle edge cases like empty strings and single characters. Language: Python. Function name: find_longest_palindrome",
  "reference_answer": "def find_longest_palindrome(s: str) -> str:\n    if not s:\n        return \"\"\n    \n    n = len(s)\n    max_len = 1\n    start = 0\n    \n    # Check for odd length palindromes\n    for i in range(n):\n        left, right = i, i\n        while left >= 0 and right < n and s[left] == s[right]:\n            if right - left + 1 > max_len:\n                max_len = right - left + 1\n                start = left\n            left -= 1\n            right += 1\n    \n    # Check for even length palindromes\n    for i in range(n - 1):\n        left, right = i, i + 1\n        while left >= 0 and right < n and s[left] == s[right]:\n            if right - left + 1 > max_len:\n                max_len = right - left + 1\n                start = left\n            left -= 1\n            right += 1\n    \n    return s[start:start + max_len]",
  "assertions": "assert find_longest_palindrome(\"babad\") == \"bab\" or find_longest_palindrome(\"babad\") == \"aba\"\nassert find_longest_palindrome(\"cbbd\") == \"bb\"\nassert find_longest_palindrome(\"a\") == \"a\"\nassert find_longest_palindrome(\"ac\") == \"a\" or find_longest_palindrome(\"ac\") == \"c\"\nassert find_longest_palindrome(\"\") == \"\"\nassert find_longest_palindrome(\"racecar\") == \"racecar\"\nassert find_longest_palindrome(\"noon\") == \"noon\"",
  "metadata": {
    "language": "Python",
    "function_name": "find_longest_palindrome",
    "difficulty": "advanced",
    "category": "algorithms",
    "complexity": 0.75
  }
}
```

## Output File

Save the output as: `synthetic_{{MODEL_NAME}}.jsonl`

Replace `{{MODEL_NAME}}` with your model name (e.g., `gpt-4`, `claude-3-5-sonnet`, `gemini-pro`, etc.)

## Validation Checklist

Before submitting, verify:
- [ ] Exactly 100 entries
- [ ] Language distribution matches requirements (40 Python, 40 TypeScript, 7 Go, 3 Rust, 10 other)
- [ ] Each entry has all required fields
- [ ] All code is syntactically correct
- [ ] All assertions are runnable
- [ ] Function names match between query_text and metadata
- [ ] Languages match between query_text and metadata
- [ ] Difficulty levels are evenly distributed
- [ ] No duplicate problems
- [ ] File is valid JSONL (one JSON object per line)

## Notes

- Focus on quality over quantity - each problem should be meaningful and test real programming skills
- Vary problem types: algorithms, data structures, concurrency, async programming, error handling, design patterns, etc.
- Ensure problems range from medium to very advanced difficulty
- Make sure code solutions are correct and complete
- Test assertions should comprehensively verify correctness

