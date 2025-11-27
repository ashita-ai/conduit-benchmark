"""Code execution evaluator for HumanEval benchmark.

Provides objective evaluation by executing generated code against
unit tests in a sandboxed subprocess.

No LLM-as-judge - most credible evaluation for developers.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any

from conduit_bench.evaluators.base import BaseEvaluator, EvaluationResult


class CodeExecutionEvaluator(BaseEvaluator):
    """Code execution evaluator for HumanEval dataset.

    Executes generated Python code with unit tests in a sandboxed subprocess.
    Pass/fail based on exit code (0 = all tests pass).

    Example:
        >>> evaluator = CodeExecutionEvaluator(timeout=10)
        >>> result = evaluator.evaluate(
        ...     response="    return sum(numbers)",
        ...     ground_truth="assert add([1, 2, 3]) == 6",
        ...     prompt="def add(numbers: List[int]) -> int:\\n",
        ...     entry_point="add"
        ... )
        >>> print(result.correct)  # True or False based on test execution
    """

    def __init__(self, timeout: int = 10) -> None:
        """Initialize the code execution evaluator.

        Args:
            timeout: Maximum execution time in seconds (default: 10)
        """
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Human-readable name for this evaluator."""
        return "CodeExecution (HumanEval)"

    @property
    def description(self) -> str:
        """Description of the evaluation method."""
        return f"Executes Python code with unit tests (timeout: {self.timeout}s)"

    def evaluate(
        self,
        response: str,
        ground_truth: str,
        query: str | None = None,
        prompt: str | None = None,
        entry_point: str | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate generated code by executing unit tests.

        Args:
            response: The model's generated code (function body)
            ground_truth: Test code with assertions (check function)
            query: Original query text (unused, for interface compatibility)
            prompt: Function signature/docstring to prepend
            entry_point: Function name to test (for check() call)
            **kwargs: Additional parameters

        Returns:
            EvaluationResult with score (1.0 if all tests pass, 0.0 otherwise)
        """
        if not response:
            return EvaluationResult(
                score=0.0,
                correct=False,
                predicted=None,
                expected="Tests pass",
                metadata={"error": "Empty response"},
            )

        # Build complete code
        full_code = self._build_test_code(
            response=response,
            prompt=prompt or "",
            test_code=ground_truth,
            entry_point=entry_point,
        )

        # Execute in sandbox
        success, output, error = self._execute_code(full_code)

        return EvaluationResult(
            score=1.0 if success else 0.0,
            correct=success,
            predicted="Tests pass" if success else f"Tests fail: {error[:200] if error else 'Unknown error'}",
            expected="Tests pass",
            metadata={
                "success": success,
                "stdout": output[:500] if output else None,
                "stderr": error[:500] if error else None,
                "timeout": self.timeout,
                "code_length": len(full_code),
            },
        )

    def extract_answer(self, response: str, **kwargs: Any) -> str | None:
        """Extract code from response (clean markdown fences if present).

        Args:
            response: The model's response text
            **kwargs: Additional parameters

        Returns:
            Cleaned code string
        """
        if not response:
            return None

        code = response.strip()

        # Remove markdown code fences
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        return code.strip()

    def _build_test_code(
        self,
        response: str,
        prompt: str,
        test_code: str,
        entry_point: str | None,
    ) -> str:
        """Build complete executable code with tests.

        Args:
            response: Model's generated code
            prompt: Function signature/docstring
            test_code: Test assertions
            entry_point: Function name for check() call

        Returns:
            Complete Python code ready for execution
        """
        # Clean the response
        code = self.extract_answer(response) or response

        # Build test harness
        parts = [
            "# Auto-generated test harness",
            "from typing import *",
            "import math",
            "import re",
            "import sys",
            "import copy",
            "import datetime",
            "import itertools",
            "import collections",
            "import heapq",
            "import statistics",
            "import functools",
            "import hashlib",
            "import numpy",
            "from collections import *",
            "",
            "# Function definition",
            prompt.rstrip() if prompt else "",
            code,
            "",
            "# Test code",
            test_code,
        ]

        # Add check() call if entry point provided
        if entry_point:
            parts.extend([
                "",
                f"check({entry_point})",
            ])

        return "\n".join(parts)

    def _execute_code(self, code: str) -> tuple[bool, str, str]:
        """Execute code in sandboxed subprocess.

        Args:
            code: Complete Python code to execute

        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Execute with timeout
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {self.timeout} seconds"

        except Exception as e:
            return False, "", str(e)

        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass
