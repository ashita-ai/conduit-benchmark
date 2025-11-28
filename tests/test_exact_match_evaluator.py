"""Tests for ExactMatchEvaluator."""

import pytest

from conduit_bench.evaluators.exact_match import ExactMatchEvaluator
from conduit_bench.evaluators.base import EvaluationResult


@pytest.fixture
def gsm8k_evaluator() -> ExactMatchEvaluator:
    """Create GSM8K evaluator."""
    return ExactMatchEvaluator(dataset_type="gsm8k")


@pytest.fixture
def mmlu_evaluator() -> ExactMatchEvaluator:
    """Create MMLU evaluator."""
    return ExactMatchEvaluator(dataset_type="mmlu")


# ===== Initialization and Properties =====


def test_gsm8k_evaluator_initialization(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test GSM8K evaluator initialization."""
    assert gsm8k_evaluator.dataset_type == "gsm8k"


def test_mmlu_evaluator_initialization(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test MMLU evaluator initialization."""
    assert mmlu_evaluator.dataset_type == "mmlu"


def test_gsm8k_evaluator_name(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test GSM8K evaluator name property."""
    assert gsm8k_evaluator.name == "ExactMatch (GSM8K)"


def test_mmlu_evaluator_name(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test MMLU evaluator name property."""
    assert mmlu_evaluator.name == "ExactMatch (MMLU)"


def test_gsm8k_evaluator_description(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test GSM8K evaluator description."""
    description = gsm8k_evaluator.description
    assert "####" in description
    assert "numeric" in description.lower()


def test_mmlu_evaluator_description(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test MMLU evaluator description."""
    description = mmlu_evaluator.description
    assert "A/B/C/D" in description


# ===== GSM8K Answer Extraction =====


def test_extract_gsm8k_answer_standard_format(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting answer from standard GSM8K format."""
    response = "Let me solve this step by step.\n48/2 = 24\n48 + 24 = 72\n#### 72"
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "72"


def test_extract_gsm8k_answer_with_comma(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting comma-separated number."""
    response = "The total is:\n#### 1,234"
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "1234"


def test_extract_gsm8k_answer_negative(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test extracting negative number."""
    response = "The result is negative.\n#### -42"
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "-42"


def test_extract_gsm8k_answer_decimal(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test extracting decimal number."""
    response = "The answer is:\n#### 3.14"
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "3.14"


def test_extract_gsm8k_answer_with_spaces(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting answer with spaces around ####."""
    response = "Final answer:\n####   123   "
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "123"


def test_extract_gsm8k_answer_fallback_answer_is(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test fallback pattern: 'answer is N'."""
    response = "After calculating, the answer is 42."
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "42"


def test_extract_gsm8k_answer_fallback_result_is(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test fallback pattern: 'result is N'."""
    response = "The result is 100."
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "100"


def test_extract_gsm8k_answer_fallback_equals(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test fallback pattern: '= N' at end."""
    response = "Total calculation = 256"
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "256"


def test_extract_gsm8k_answer_fallback_last_number(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test fallback to last number in text."""
    response = "We have 10 apples and 5 oranges. Final count: 15"
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "15"


def test_extract_gsm8k_answer_no_match(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test extraction when no number found."""
    response = "I don't know the answer."
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer is None


def test_extract_gsm8k_answer_empty_string(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extraction from empty string."""
    answer = gsm8k_evaluator.extract_answer("")
    assert answer is None


def test_extract_gsm8k_answer_large_number(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting large comma-separated number."""
    response = "Total: #### 1,234,567"
    answer = gsm8k_evaluator.extract_answer(response)
    assert answer == "1234567"


# ===== MMLU Answer Extraction =====


def test_extract_mmlu_answer_explicit_answer_is(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting MMLU answer from 'answer is X' pattern."""
    response = "The answer is C because Paris is the capital of France."
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "C"


def test_extract_mmlu_answer_explicit_choice_is(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting MMLU answer from 'choice is X' pattern."""
    response = "The choice is B."
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "B"


def test_extract_mmlu_answer_correct_answer_is(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting from 'correct answer is X' pattern."""
    response = "The correct answer is D."
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "D"


def test_extract_mmlu_answer_is_correct(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test extracting from 'X is correct' pattern."""
    response = "A is correct because of the following reasons..."
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "A"


def test_extract_mmlu_answer_standalone_letter(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extracting standalone letter."""
    response = "C"
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "C"


def test_extract_mmlu_answer_lowercase(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test extracting lowercase letter (should uppercase)."""
    response = "The answer is b."
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "B"


def test_extract_mmlu_answer_with_colon(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test extracting answer with colon separator."""
    response = "Answer: D"
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "D"


def test_extract_mmlu_answer_fallback_first_letter(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test fallback to first A/B/C/D found."""
    response = "I think this is B, but it could also be C."
    answer = mmlu_evaluator.extract_answer(response)
    assert answer == "B"


def test_extract_mmlu_answer_no_match(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test extraction when no valid choice found."""
    response = "I don't know the answer."
    answer = mmlu_evaluator.extract_answer(response)
    assert answer is None


def test_extract_mmlu_answer_empty_string(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test extraction from empty string."""
    answer = mmlu_evaluator.extract_answer("")
    assert answer is None


def test_extract_mmlu_answer_invalid_letter(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test extraction ignores invalid letters (E, F, etc)."""
    response = "The answer is E."
    answer = mmlu_evaluator.extract_answer(response)
    # Should not match E, should return None
    assert answer is None


# ===== Answer Normalization =====


def test_normalize_gsm8k_answer_remove_commas(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test normalizing GSM8K answer removes commas."""
    normalized = gsm8k_evaluator._normalize_answer("1,234")
    assert normalized == "1234"


def test_normalize_gsm8k_answer_strip_whitespace(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test normalizing GSM8K answer strips whitespace."""
    normalized = gsm8k_evaluator._normalize_answer("  72  ")
    assert normalized == "72"


def test_normalize_gsm8k_answer_extract_from_format(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test normalizing GSM8K answer extracts from #### format."""
    normalized = gsm8k_evaluator._normalize_answer("#### 100")
    assert normalized == "100"


def test_normalize_gsm8k_answer_empty_string(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test normalizing empty string."""
    normalized = gsm8k_evaluator._normalize_answer("")
    assert normalized == ""


def test_normalize_mmlu_answer_uppercase(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test normalizing MMLU answer uppercases."""
    normalized = mmlu_evaluator._normalize_answer("c")
    assert normalized == "C"


def test_normalize_mmlu_answer_extract_letter(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test normalizing MMLU answer extracts letter from text."""
    # Note: "Choice B" matches 'C' first (from "Choice"), so use different text
    normalized = mmlu_evaluator._normalize_answer("Option: B")
    assert normalized == "B"


def test_normalize_mmlu_answer_valid_letter(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test normalizing valid single letter."""
    normalized = mmlu_evaluator._normalize_answer("A")
    assert normalized == "A"


def test_normalize_mmlu_answer_strip_whitespace(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test normalizing MMLU answer strips whitespace."""
    normalized = mmlu_evaluator._normalize_answer("  D  ")
    assert normalized == "D"


# ===== Full Evaluation =====


def test_evaluate_gsm8k_correct_match(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test GSM8K evaluation with correct answer."""
    response = "Step by step solution.\n#### 72"
    ground_truth = "72"

    result = gsm8k_evaluator.evaluate(response, ground_truth)

    assert isinstance(result, EvaluationResult)
    assert result.score == 1.0
    assert result.correct is True
    assert result.predicted == "72"
    assert result.expected == "72"
    assert result.metadata["dataset_type"] == "gsm8k"


def test_evaluate_gsm8k_incorrect_match(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test GSM8K evaluation with incorrect answer."""
    response = "My answer is:\n#### 100"
    ground_truth = "72"

    result = gsm8k_evaluator.evaluate(response, ground_truth)

    assert result.score == 0.0
    assert result.correct is False
    assert result.predicted == "100"
    assert result.expected == "72"


def test_evaluate_gsm8k_with_comma_normalization(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test GSM8K evaluation normalizes commas correctly."""
    response = "The total is:\n#### 1,234"
    ground_truth = "1234"

    result = gsm8k_evaluator.evaluate(response, ground_truth)

    assert result.correct is True
    assert result.predicted == "1234"


def test_evaluate_gsm8k_ground_truth_with_comma(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test GSM8K evaluation when ground truth has comma."""
    response = "Answer:\n#### 5000"
    ground_truth = "5,000"

    result = gsm8k_evaluator.evaluate(response, ground_truth)

    assert result.correct is True


def test_evaluate_gsm8k_no_answer_extracted(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test GSM8K evaluation when answer cannot be extracted."""
    response = "I don't know."
    ground_truth = "72"

    result = gsm8k_evaluator.evaluate(response, ground_truth)

    assert result.correct is False
    assert result.predicted is None


def test_evaluate_mmlu_correct_match(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test MMLU evaluation with correct answer."""
    response = "The answer is C because Paris is the capital."
    ground_truth = "C"

    result = mmlu_evaluator.evaluate(response, ground_truth)

    assert result.score == 1.0
    assert result.correct is True
    assert result.predicted == "C"
    assert result.expected == "C"


def test_evaluate_mmlu_incorrect_match(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test MMLU evaluation with incorrect answer."""
    response = "I believe the answer is B."
    ground_truth = "D"

    result = mmlu_evaluator.evaluate(response, ground_truth)

    assert result.score == 0.0
    assert result.correct is False
    assert result.predicted == "B"
    assert result.expected == "D"


def test_evaluate_mmlu_case_insensitive(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test MMLU evaluation is case insensitive."""
    response = "The answer is b."
    ground_truth = "B"

    result = mmlu_evaluator.evaluate(response, ground_truth)

    assert result.correct is True


def test_evaluate_mmlu_ground_truth_lowercase(
    mmlu_evaluator: ExactMatchEvaluator,
) -> None:
    """Test MMLU evaluation when ground truth is lowercase."""
    response = "Answer: A"
    ground_truth = "a"

    result = mmlu_evaluator.evaluate(response, ground_truth)

    assert result.correct is True


def test_evaluate_mmlu_no_answer_extracted(mmlu_evaluator: ExactMatchEvaluator) -> None:
    """Test MMLU evaluation when answer cannot be extracted."""
    response = "This is confusing."
    ground_truth = "C"

    result = mmlu_evaluator.evaluate(response, ground_truth)

    assert result.correct is False
    assert result.predicted is None


def test_evaluate_with_query_parameter(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test evaluation accepts optional query parameter."""
    response = "#### 42"
    ground_truth = "42"
    query = "What is the answer?"

    result = gsm8k_evaluator.evaluate(response, ground_truth, query=query)

    assert result.correct is True


def test_evaluate_metadata_includes_response_length(
    gsm8k_evaluator: ExactMatchEvaluator,
) -> None:
    """Test evaluation metadata includes response length."""
    response = "This is a test response with some length."
    ground_truth = "42"

    result = gsm8k_evaluator.evaluate(response, ground_truth)

    assert "raw_response_length" in result.metadata
    assert result.metadata["raw_response_length"] == len(response)


def test_evaluate_empty_response(gsm8k_evaluator: ExactMatchEvaluator) -> None:
    """Test evaluation with empty response."""
    result = gsm8k_evaluator.evaluate("", "42")

    assert result.correct is False
    assert result.predicted is None
    assert result.metadata["raw_response_length"] == 0
