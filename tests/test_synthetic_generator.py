"""Tests for SyntheticQueryGenerator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit_bench.generators import SyntheticQueryGenerator, QUERY_TEMPLATES


@pytest.fixture
def generator() -> SyntheticQueryGenerator:
    """Create a generator with fixed seed."""
    return SyntheticQueryGenerator(seed=42)


def test_generator_initialization() -> None:
    """Test generator initialization."""
    gen = SyntheticQueryGenerator(reference_model="openai:gpt-4o", seed=42)
    assert gen.reference_model == "openai:gpt-4o"
    assert gen.seed == 42


def test_generate_query_text(generator: SyntheticQueryGenerator) -> None:
    """Test query text generation."""
    query_text = generator._generate_query_text("technical")

    assert isinstance(query_text, str)
    assert len(query_text) > 0
    # Should not have unfilled placeholders
    assert "{" not in query_text or "}" not in query_text


def test_generate_query_text_all_categories(
    generator: SyntheticQueryGenerator,
) -> None:
    """Test query generation for all categories."""
    for category in QUERY_TEMPLATES.keys():
        query_text = generator._generate_query_text(category)
        assert isinstance(query_text, str)
        assert len(query_text) > 0


@pytest.mark.asyncio
async def test_generate_simple(generator: SyntheticQueryGenerator) -> None:
    """Test simple generation without reference answers."""
    queries = await generator.generate_simple(n_queries=10)

    assert len(queries) == 10
    assert all(q.query_text for q in queries)
    assert all(q.category in QUERY_TEMPLATES.keys() for q in queries)
    assert all(q.complexity in [0.3, 0.5, 0.8] for q in queries)
    assert all(q.reference_answer is None for q in queries)


@pytest.mark.asyncio
async def test_generate_simple_specific_categories(
    generator: SyntheticQueryGenerator,
) -> None:
    """Test simple generation with specific categories."""
    categories = ["technical", "math"]
    queries = await generator.generate_simple(n_queries=6, categories=categories)

    assert len(queries) == 6
    assert all(q.category in categories for q in queries)


@pytest.mark.asyncio
async def test_generate_with_reference_answers(
    generator: SyntheticQueryGenerator,
) -> None:
    """Test generation with reference answers (mocked)."""
    # Mock PydanticAI Agent
    mock_result = MagicMock()
    mock_result.data = "Mocked reference answer"

    with patch("conduit_bench.generators.synthetic.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        queries = await generator.generate(n_queries=5, show_progress=False)

    assert len(queries) == 5
    assert all(q.reference_answer == "Mocked reference answer" for q in queries)


@pytest.mark.asyncio
async def test_generate_distributes_categories(
    generator: SyntheticQueryGenerator,
) -> None:
    """Test that queries are distributed across categories."""
    categories = ["technical", "math", "creative"]

    with patch("conduit_bench.generators.synthetic.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Answer"
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        queries = await generator.generate(
            n_queries=9, categories=categories, show_progress=False
        )

    # Should have 3 queries per category (evenly distributed)
    category_counts = {cat: 0 for cat in categories}
    for query in queries:
        category_counts[query.category] += 1

    assert all(count == 3 for count in category_counts.values())


@pytest.mark.asyncio
async def test_generate_handles_reference_failure(
    generator: SyntheticQueryGenerator,
) -> None:
    """Test generation handles reference answer failures gracefully."""
    with patch("conduit_bench.generators.synthetic.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("API Error"))
        MockAgent.return_value = mock_agent

        queries = await generator.generate(n_queries=3, show_progress=False)

    assert len(queries) == 3
    # Should have fallback reference answers
    assert all(q.reference_answer is not None for q in queries)
    assert all("failed" in q.reference_answer.lower() for q in queries)


def test_reproducibility_with_seed() -> None:
    """Test that using the same seed produces the same queries."""
    gen1 = SyntheticQueryGenerator(seed=123)
    gen2 = SyntheticQueryGenerator(seed=123)

    # Generate without reference answers for speed
    import asyncio

    queries1 = asyncio.run(gen1.generate_simple(n_queries=5))
    queries2 = asyncio.run(gen2.generate_simple(n_queries=5))

    # Should generate identical queries
    for q1, q2 in zip(queries1, queries2):
        assert q1.query_text == q2.query_text
        assert q1.category == q2.category
        assert q1.complexity == q2.complexity
