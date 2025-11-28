"""Tests for GSM8K dataset loader."""

import pytest
from unittest.mock import MagicMock, patch

from conduit_bench.datasets.gsm8k import GSM8KLoader
from conduit_bench.benchmark_models import BenchmarkQuery


@pytest.fixture
def loader() -> GSM8KLoader:
    """Create a GSM8K loader instance."""
    return GSM8KLoader()


@pytest.fixture
def mock_dataset_item() -> dict:
    """Create a mock GSM8K dataset item."""
    return {
        "question": "Natalia sold clips to 48 of her friends in April, "
        "and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "#### 72",
    }


def test_loader_initialization(loader: GSM8KLoader) -> None:
    """Test GSM8K loader initialization."""
    assert loader.DATASET_NAME == "openai/gsm8k"
    assert loader.DATASET_CONFIG == "main"
    assert loader._dataset is None


def test_loader_description(loader: GSM8KLoader) -> None:
    """Test loader description property."""
    description = loader.description
    assert isinstance(description, str)
    assert "GSM8K" in description
    assert "8,792" in description
    assert "1,319" in description


def test_extract_answer_standard_format(loader: GSM8KLoader) -> None:
    """Test answer extraction from standard #### format."""
    answer_text = "Some solution steps.\n#### 72"
    result = loader._extract_answer(answer_text)
    assert result == "72"


def test_extract_answer_with_comma(loader: GSM8KLoader) -> None:
    """Test answer extraction with comma-separated numbers."""
    answer_text = "Some solution steps.\n#### 1,234"
    result = loader._extract_answer(answer_text)
    assert result == "1234"


def test_extract_answer_negative_number(loader: GSM8KLoader) -> None:
    """Test answer extraction with negative numbers."""
    answer_text = "The result is negative.\n#### -42"
    result = loader._extract_answer(answer_text)
    assert result == "-42"


def test_extract_answer_decimal(loader: GSM8KLoader) -> None:
    """Test answer extraction with decimal numbers."""
    answer_text = "The answer is a decimal.\n#### 3.14"
    result = loader._extract_answer(answer_text)
    assert result == "3.14"


def test_extract_answer_with_spaces(loader: GSM8KLoader) -> None:
    """Test answer extraction with spaces after ####."""
    answer_text = "Solution here.\n####   123  "
    result = loader._extract_answer(answer_text)
    assert result == "123"


def test_extract_answer_fallback_last_number(loader: GSM8KLoader) -> None:
    """Test fallback to last number when #### not found."""
    answer_text = "The total is 48 and the answer is 72."
    result = loader._extract_answer(answer_text)
    assert result == "72"


def test_extract_answer_fallback_with_comma(loader: GSM8KLoader) -> None:
    """Test fallback extraction with comma-separated numbers."""
    answer_text = "The total comes to 1,234 units."
    result = loader._extract_answer(answer_text)
    assert result == "1234"


def test_extract_answer_no_match(loader: GSM8KLoader) -> None:
    """Test answer extraction when no number found."""
    answer_text = "No numbers here at all."
    result = loader._extract_answer(answer_text)
    assert result == ""


def test_convert_item(loader: GSM8KLoader, mock_dataset_item: dict) -> None:
    """Test conversion of dataset item to BenchmarkQuery."""
    query = loader._convert_item(mock_dataset_item, idx=42, split="test")

    assert isinstance(query, BenchmarkQuery)
    assert query.query_id == "gsm8k_test_42"
    assert "Natalia sold clips" in query.query_text
    assert "step by step" in query.query_text
    assert "#### [answer]" in query.query_text
    assert query.reference_answer == "72"
    assert query.metadata["dataset"] == "gsm8k"
    assert query.metadata["split"] == "test"
    assert "full_solution" in query.metadata
    assert "original_question" in query.metadata


def test_convert_item_preserves_question(
    loader: GSM8KLoader, mock_dataset_item: dict
) -> None:
    """Test that original question is preserved in metadata."""
    query = loader._convert_item(mock_dataset_item, idx=0, split="train")

    assert query.metadata["original_question"] == mock_dataset_item["question"]
    assert query.metadata["full_solution"] == mock_dataset_item["answer"]


@patch("conduit_bench.datasets.gsm8k.load_dataset")
def test_load_full_dataset(mock_load_dataset: MagicMock, loader: GSM8KLoader) -> None:
    """Test loading full dataset without limit."""
    # Mock dataset with 3 items
    mock_data = [
        {
            "question": "Question 1?",
            "answer": "Solution 1\n#### 10",
        },
        {
            "question": "Question 2?",
            "answer": "Solution 2\n#### 20",
        },
        {
            "question": "Question 3?",
            "answer": "Solution 3\n#### 30",
        },
    ]
    mock_dataset = MagicMock()
    mock_dataset.__len__ = lambda self: len(mock_data)
    mock_dataset.__iter__ = lambda self: iter(mock_data)
    mock_load_dataset.return_value = mock_dataset

    queries = loader.load(split="test", limit=None)

    assert len(queries) == 3
    assert all(isinstance(q, BenchmarkQuery) for q in queries)
    assert queries[0].reference_answer == "10"
    assert queries[1].reference_answer == "20"
    assert queries[2].reference_answer == "30"
    mock_load_dataset.assert_called_once_with("openai/gsm8k", "main", split="test")


@patch("conduit_bench.datasets.gsm8k.load_dataset")
def test_load_with_limit(mock_load_dataset: MagicMock, loader: GSM8KLoader) -> None:
    """Test loading dataset with limit."""
    # Mock dataset with shuffle and select methods
    mock_data = [{"question": f"Q{i}?", "answer": f"A\n#### {i}"} for i in range(10)]
    mock_dataset = MagicMock()
    mock_dataset.__len__ = lambda self: len(mock_data)

    # Mock shuffle returns self
    mock_shuffled = MagicMock()
    mock_shuffled.select = MagicMock(
        return_value=[mock_data[i] for i in range(3)]
    )
    mock_dataset.shuffle = MagicMock(return_value=mock_shuffled)

    mock_load_dataset.return_value = mock_dataset

    queries = loader.load(split="train", limit=3, seed=42)

    assert len(queries) == 3
    mock_dataset.shuffle.assert_called_once_with(seed=42)
    mock_shuffled.select.assert_called_once()


@patch("conduit_bench.datasets.gsm8k.load_dataset")
def test_load_test_split(mock_load_dataset: MagicMock, loader: GSM8KLoader) -> None:
    """Test loading test split specifically."""
    mock_dataset = MagicMock()
    mock_dataset.__len__ = lambda self: 0
    mock_dataset.__iter__ = lambda self: iter([])
    mock_load_dataset.return_value = mock_dataset

    loader.load(split="test")

    mock_load_dataset.assert_called_once_with("openai/gsm8k", "main", split="test")


@patch("conduit_bench.datasets.gsm8k.load_dataset")
def test_load_train_split(mock_load_dataset: MagicMock, loader: GSM8KLoader) -> None:
    """Test loading train split specifically."""
    mock_dataset = MagicMock()
    mock_dataset.__len__ = lambda self: 0
    mock_dataset.__iter__ = lambda self: iter([])
    mock_load_dataset.return_value = mock_dataset

    loader.load(split="train")

    mock_load_dataset.assert_called_once_with("openai/gsm8k", "main", split="train")


def test_extract_answer_multiple_hash_marks(loader: GSM8KLoader) -> None:
    """Test answer extraction with multiple #### in text."""
    # Should match the first ####
    answer_text = "First: #### 100\nSecond: #### 200"
    result = loader._extract_answer(answer_text)
    assert result == "100"


def test_extract_answer_large_number(loader: GSM8KLoader) -> None:
    """Test answer extraction with large comma-separated number."""
    answer_text = "Total: #### 1,234,567"
    result = loader._extract_answer(answer_text)
    assert result == "1234567"


def test_query_formatting_includes_instruction(
    loader: GSM8KLoader, mock_dataset_item: dict
) -> None:
    """Test that query includes step-by-step instruction."""
    query = loader._convert_item(mock_dataset_item, idx=0, split="test")

    assert "step by step" in query.query_text.lower()
    assert "show your work" in query.query_text.lower()
    assert "#### [answer]" in query.query_text


def test_query_id_format(loader: GSM8KLoader, mock_dataset_item: dict) -> None:
    """Test query ID format is correct."""
    query_train = loader._convert_item(mock_dataset_item, idx=42, split="train")
    query_test = loader._convert_item(mock_dataset_item, idx=100, split="test")

    assert query_train.query_id == "gsm8k_train_42"
    assert query_test.query_id == "gsm8k_test_100"


def test_metadata_completeness(loader: GSM8KLoader, mock_dataset_item: dict) -> None:
    """Test that all expected metadata fields are present."""
    query = loader._convert_item(mock_dataset_item, idx=0, split="test")

    required_fields = ["dataset", "split", "full_solution", "original_question"]
    for field in required_fields:
        assert field in query.metadata, f"Missing metadata field: {field}"

    assert query.metadata["dataset"] == "gsm8k"
    assert query.metadata["split"] == "test"
