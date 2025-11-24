"""Synthetic query generator for benchmark datasets.

Generates diverse queries across multiple categories and complexity levels,
with reference answers from GPT-4o for evaluation.
"""

import random

from pydantic_ai import Agent

from conduit_bench.benchmark_models import BenchmarkQuery


# Query templates by category
QUERY_TEMPLATES = {
    "technical": [
        "Explain how {concept} works in {technology}",
        "What are the best practices for {task} in {technology}?",
        "How do I debug {problem} in {technology}?",
        "Compare {tech1} vs {tech2} for {use_case}",
        "Write a {technology} function to {task}",
    ],
    "creative": [
        "Write a short story about {theme}",
        "Create a marketing slogan for {product}",
        "Compose a poem about {topic}",
        "Generate creative ideas for {problem}",
        "Write a blog post introduction about {subject}",
    ],
    "factual": [
        "What is {topic}?",
        "Who invented {invention}?",
        "When did {event} happen?",
        "Where is {location} located?",
        "Why does {phenomenon} occur?",
    ],
    "math": [
        "Solve: {math_problem}",
        "Calculate {calculation}",
        "What is the formula for {formula}?",
        "Prove that {theorem}",
        "Explain the concept of {math_concept}",
    ],
    "code": [
        "Write a {language} function that {task}",
        "Debug this {language} code: {code_snippet}",
        "Optimize this {language} algorithm for {optimization}",
        "Refactor this {language} code to use {pattern}",
        "Add error handling to this {language} function",
    ],
    "analysis": [
        "Analyze the pros and cons of {topic}",
        "Compare and contrast {item1} and {item2}",
        "What are the implications of {event}?",
        "Evaluate the effectiveness of {approach}",
        "Identify the key factors in {situation}",
    ],
    "howto": [
        "How do I {task}?",
        "What steps are needed to {goal}?",
        "Guide me through {process}",
        "Teach me how to {skill}",
        "What's the best way to {action}?",
    ],
    "creative_writing": [
        "Write a dialogue between {character1} and {character2}",
        "Describe {scene} in vivid detail",
        "Create a character profile for {character}",
        "Outline a plot for a story about {premise}",
        "Write the opening paragraph for {genre} story",
    ],
    "business": [
        "What is the ROI of {investment}?",
        "How to improve {metric} in {business}?",
        "Analyze the market for {product}",
        "Create a business plan for {idea}",
        "What are the risks of {strategy}?",
    ],
    "science": [
        "Explain the scientific principle behind {phenomenon}",
        "What causes {natural_event}?",
        "How does {organism} {function}?",
        "Describe the process of {scientific_process}",
        "What is the evidence for {theory}?",
    ],
}

# Vocabulary for filling templates
VOCABULARY = {
    "concept": ["recursion", "polymorphism", "async/await", "dependency injection"],
    "technology": ["Python", "JavaScript", "React", "Docker", "Kubernetes"],
    "task": ["testing", "deployment", "optimization", "security"],
    "problem": ["memory leaks", "race conditions", "deadlocks", "performance issues"],
    "tech1": ["Python", "Java", "Go", "Rust"],
    "tech2": ["JavaScript", "TypeScript", "C++", "Ruby"],
    "use_case": ["web development", "data processing", "API development", "microservices"],
    "theme": ["artificial intelligence", "space exploration", "time travel", "future cities"],
    "product": ["smart watch", "eco-friendly car", "productivity app", "online course"],
    "topic": ["quantum computing", "climate change", "artificial intelligence", "renewable energy"],
    "subject": ["remote work trends", "sustainable living", "digital marketing", "healthy eating"],
    "invention": ["the internet", "the lightbulb", "the telephone", "the automobile"],
    "event": ["the moon landing", "the fall of the Berlin Wall", "the first flight", "the internet creation"],
    "location": ["the Great Barrier Reef", "Mount Everest", "the Amazon Rainforest", "Silicon Valley"],
    "phenomenon": ["lightning", "the aurora borealis", "photosynthesis", "gravity"],
    "math_problem": ["2x + 5 = 15", "∫x²dx", "d/dx(sin(x))", "Σ(n=1 to 10) n"],
    "calculation": ["compound interest", "standard deviation", "probability", "surface area"],
    "formula": ["area of a circle", "quadratic formula", "pythagorean theorem", "compound interest"],
    "theorem": ["Fermat's Last Theorem", "Pythagorean theorem", "the fundamental theorem of calculus"],
    "math_concept": ["derivatives", "integrals", "limits", "probability distributions"],
    "language": ["Python", "JavaScript", "Java", "C++", "Go"],
    "code_snippet": ["buggy loop", "inefficient algorithm", "unclear variable names"],
    "optimization": ["speed", "memory usage", "readability", "maintainability"],
    "pattern": ["factory pattern", "observer pattern", "strategy pattern", "singleton pattern"],
    "item1": ["SQL databases", "cloud computing", "microservices"],
    "item2": ["NoSQL databases", "on-premise servers", "monolithic architecture"],
    "approach": ["agile methodology", "test-driven development", "pair programming"],
    "situation": ["software project failure", "team productivity", "code quality"],
    "goal": ["learn machine learning", "build a web app", "pass a certification"],
    "process": ["deploying to production", "setting up CI/CD", "code review"],
    "skill": ["public speaking", "data analysis", "web design"],
    "action": ["learn a new language", "improve productivity", "stay motivated"],
    "character1": ["a scientist", "a detective", "an astronaut"],
    "character2": ["an AI", "a philosopher", "a time traveler"],
    "scene": ["a futuristic cityscape", "an abandoned laboratory", "a serene forest"],
    "character": ["a reluctant hero", "a mysterious stranger", "a wise mentor"],
    "premise": ["first contact with aliens", "a world without technology", "time loop"],
    "genre": ["science fiction", "mystery", "fantasy", "thriller"],
    "investment": ["marketing automation", "employee training", "new technology"],
    "metric": ["customer satisfaction", "employee retention", "conversion rate"],
    "business": ["e-commerce", "SaaS", "consulting", "manufacturing"],
    "idea": ["a subscription box service", "a mobile app", "a co-working space"],
    "strategy": ["aggressive expansion", "cost-cutting", "digital transformation"],
    "organism": ["a tree", "a shark", "a bacterium", "a bird"],
    "function": ["survive", "reproduce", "adapt", "communicate"],
    "scientific_process": ["photosynthesis", "DNA replication", "evolution", "the water cycle"],
    "theory": ["evolution", "the Big Bang", "plate tectonics", "quantum mechanics"],
    "natural_event": ["earthquakes", "hurricanes", "volcanic eruptions", "eclipses"],
}


class SyntheticQueryGenerator:
    """Generates synthetic queries for benchmarking.

    Creates diverse queries across categories with varying complexity,
    and generates reference answers using GPT-4o for evaluation.

    Example:
        >>> generator = SyntheticQueryGenerator(seed=42)
        >>> queries = await generator.generate(n_queries=1000)
        >>> print(len(queries))
        1000
        >>> print(queries[0].category)
        "technical"
    """

    def __init__(
        self,
        reference_model: str = "openai:gpt-4o",
        seed: int | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            reference_model: Model to use for generating reference answers
            seed: Random seed for reproducibility
        """
        self.reference_model = reference_model
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    async def generate(
        self,
        n_queries: int,
        categories: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[BenchmarkQuery]:
        """Generate synthetic queries with reference answers.

        Args:
            n_queries: Number of queries to generate
            categories: Specific categories to use (None = all categories)
            show_progress: Whether to show progress

        Returns:
            List of benchmark queries with reference answers
        """
        # Reset seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)

        if categories is None:
            categories = list(QUERY_TEMPLATES.keys())

        queries: list[BenchmarkQuery] = []
        agent = Agent(self.reference_model, system_prompt="You are a helpful assistant.")

        # Distribute queries across categories
        queries_per_category = n_queries // len(categories)
        remainder = n_queries % len(categories)

        for idx, category in enumerate(categories):
            n_category_queries = queries_per_category + (1 if idx < remainder else 0)

            for _ in range(n_category_queries):
                # Generate query text
                query_text = self._generate_query_text(category)

                # Assign complexity (random from low, medium, high)
                complexity = random.choice([0.3, 0.5, 0.8])

                # Generate reference answer
                try:
                    result = await agent.run(query_text)
                    reference_answer = result.data
                except Exception as e:
                    # Fallback if reference generation fails
                    reference_answer = f"Reference answer failed: {e}"

                query = BenchmarkQuery(
                    query_text=query_text,
                    category=category,
                    complexity=complexity,
                    reference_answer=reference_answer,
                )
                queries.append(query)

        # Shuffle to mix categories
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(queries)

        return queries

    def _generate_query_text(self, category: str) -> str:
        """Generate a query text from a template.

        Args:
            category: Query category

        Returns:
            Generated query text
        """
        template = random.choice(QUERY_TEMPLATES[category])

        # Fill in placeholders
        query_text = template
        for placeholder in set(template.split("{")[1:]):
            placeholder = placeholder.split("}")[0]
            if placeholder in VOCABULARY:
                value = random.choice(VOCABULARY[placeholder])
                query_text = query_text.replace(f"{{{placeholder}}}", value)

        return query_text

    async def generate_simple(
        self,
        n_queries: int,
        categories: list[str] | None = None,
    ) -> list[BenchmarkQuery]:
        """Generate queries without reference answers (for quick testing).

        Args:
            n_queries: Number of queries to generate
            categories: Specific categories to use

        Returns:
            List of benchmark queries without reference answers
        """
        # Reset seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)

        if categories is None:
            categories = list(QUERY_TEMPLATES.keys())

        queries: list[BenchmarkQuery] = []
        queries_per_category = n_queries // len(categories)
        remainder = n_queries % len(categories)

        for idx, category in enumerate(categories):
            n_category_queries = queries_per_category + (1 if idx < remainder else 0)

            for _ in range(n_category_queries):
                query_text = self._generate_query_text(category)
                complexity = random.choice([0.3, 0.5, 0.8])

                query = BenchmarkQuery(
                    query_text=query_text,
                    category=category,
                    complexity=complexity,
                    reference_answer=None,  # No reference for quick generation
                )
                queries.append(query)

        # Shuffle
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(queries)

        return queries
