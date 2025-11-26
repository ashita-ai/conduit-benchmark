"""Synthetic query generator for benchmark datasets.

Generates diverse queries across multiple categories and complexity levels,
with reference answers from GPT-4o for evaluation.
"""

import random

from pydantic_ai import Agent

from conduit_bench.benchmark_models import BenchmarkQuery


# Query templates by category - complex, specific, multipart questions
QUERY_TEMPLATES = {
    "technical_specific": [
        "Explain the difference between {algo1} and {algo2} algorithms using real examples for {domain}",
        "I'm implementing {system} with {tech1} but facing {problem}. Should I switch to {tech2} or refactor? What are the trade-offs?",
        "Compare {tech1} vs {tech2} for {use_case} considering latency, cost, and operational complexity. Include benchmarks if possible.",
        "How does {concept} work under the hood in {technology}? Walk me through the implementation details and performance implications.",
        "Debug this {language} issue: {problem}. The stack trace shows {error_pattern}. Is this a threading issue or something else?",
    ],
    "multipart_architecture": [
        "I want to build {system}. I'm thinking {tech1} for {component1} and {tech2} for {component2}. What are the integration challenges and alternatives?",
        "Review my architecture: {tech1} → {tech2} → {tech3}. Should I add {tech4} for {feature}? How would you handle {problem}?",
        "I need to migrate from {old_tech} to {new_tech} while maintaining {requirement}. What's the safest migration path and how do I handle {challenge}?",
        "Building a {system} with {tech1}. Need to add {feature1} and {feature2}. Should I use {option1}, {option2}, or something else entirely?",
        "My {system} uses {tech1} but I need {feature}. Considering {option1} vs {option2}. Which integrates better and why?",
    ],
    "code_analysis": [
        "This {language} code has O(n²) complexity in {function}. How do I optimize to O(n log n) while preserving {constraint}?",
        "Refactor this {pattern1} implementation to use {pattern2}. Show me before/after with specific focus on {aspect}.",
        "I have a {problem} in production: {symptom}. Profiler shows bottleneck in {location}. Root cause and fix?",
        "Code review: This {language} {component} violates {principle}. Suggest refactoring approach that maintains backward compatibility.",
        "Why does this {language} async code deadlock when {condition}? Is this a race condition or improper {resource} handling?",
    ],
    "data_engineering": [
        "Design a real-time pipeline: {source} → {processing} → {sink}. Need sub-second latency with {volume} QPS. {tech1} or {tech2}?",
        "I'm processing {data_type} with {tech1} but getting {problem}. Should I switch to {tech2} or optimize my {tech1} config?",
        "Compare {tech1} vs {tech2} for {use_case} with {volume} scale. Consider cost, latency, and operational complexity.",
        "ETL pipeline: {source} → {transform} → {destination}. How do I handle {problem} while maintaining {requirement}?",
        "Stream processing with {tech1}: need windowing, aggregation, and {feature}. Is {tech2} better suited or should I continue with {tech1}?",
    ],
    "github_review": [
        "Look at {repo_url} and suggest improvements. I want to add {feature1} and {feature2}. I think I need {tech1} and maybe {tech2}. Or perhaps {tech3}?",
        "Review the architecture in {repo_url}. How would you refactor {component} to support {feature}? Should I use {option1} or {option2}?",
        "Analyze {repo_url} for {aspect}. What are the main issues and how would you fix them? Consider migration path from current {tech1}.",
        "Check out {repo_url} - the {component} needs {improvement}. Considering {tech1}, {tech2}, or {tech3}. Which fits best and why?",
    ],
    "system_design": [
        "Design a {system} handling {scale} with {constraint}. I'm considering {tech1} vs {tech2}. What are the failure modes?",
        "Build a distributed {system}: {component1} + {component2} + {component3}. How do I ensure {property1} and {property2}?",
        "Architecture review: {pattern} with {tech1} and {tech2}. Scaling to {scale}. What breaks first and how do I prevent it?",
        "Design {system} with requirements: {req1}, {req2}, {req3}. Trade-offs between {option1} and {option2} approaches?",
    ],
    "debugging_complex": [
        "Production issue: {symptom} when {condition}. Logs show {error}. Metrics indicate {metric_pattern}. Where do I start?",
        "{System} fails intermittently with {error}. Only happens under {condition}. Checked {checked_items}. What am I missing?",
        "Memory leak in {language} {component}: heap grows from {size1} to {size2} over {time}. Profiler shows {finding}. Root cause?",
        "Race condition in {system}: {symptom} occurs randomly. Involves {component1} and {component2}. How do I reproduce and fix?",
    ],
    "creative_technical": [
        "Write a {genre} story about {scenario} that subtly explains {technical_concept} without being didactic.",
        "Create an analogy explaining {complex_concept} to {audience} using {domain} metaphors.",
        "Write technical documentation for {api} that's both comprehensive and approachable for {skill_level} developers.",
    ],
    "creative_writing": [
        "Write the opening scene of a {genre} story where {character1} discovers {discovery}. Make it {tone} and focus on {aspect}.",
        "Create a dialogue between {character1} and {character2} about {philosophical_topic}. One argues {position1}, the other {position2}. Who's right?",
        "Write a {length} story that explores {theme} without explicitly mentioning it. Use {setting} as the backdrop.",
        "Describe {scene} from the perspective of {narrator}. Focus on {sensory_aspect} and create {mood}.",
    ],
    "business_strategic": [
        "My {business_type} is losing to {competitor} in {market}. They're winning on {advantage}. Should I {strategy1}, {strategy2}, or pivot entirely?",
        "Analyzing {company} entering {market} against {established_player}. What's their {moat}? Where are they vulnerable? Investment thesis?",
        "Build vs buy decision: {capability} for {business_type}. Internal build costs {cost1}, vendor charges {cost2}. Hidden costs? Strategic implications?",
        "Pricing strategy for {product} in {market}. Competitors at {price_point1}, premium segment at {price_point2}. We offer {differentiator}. Positioning?",
    ],
    "philosophical_analysis": [
        "Is {ethical_dilemma} justified when {constraint}? Consider {framework1} vs {framework2} perspectives. Where do they conflict?",
        "Analyze {philosophical_concept} in context of {modern_issue}. How would {philosopher1} and {philosopher2} view this differently?",
        "The trolley problem but {variation}. Does {constraint} change the moral calculus? Why or why not?",
        "Explain the difference between {concept1} and {concept2} using {concrete_example}. Where do people typically confuse them?",
    ],
    "science_complex": [
        "Explain {scientific_concept} to {audience} without using {forbidden_term}. Use {real_world_example} to illustrate.",
        "What would happen if {hypothetical_change} to {natural_law}? Walk through {consequence1}, {consequence2}, and {consequence3}.",
        "The {scientific_debate} controversy: what does {side1} get right and where does {side2} have valid points? What's the synthesis?",
        "How does {phenomenon} actually work at {scale}? I've heard {misconception} but that seems wrong. Break it down.",
    ],
    "analysis_comparative": [
        "Compare {approach1} vs {approach2} for {problem}. Not just features - real tradeoffs in {context}. Which fails first under {stress}?",
        "Why does {successful_example} work in {context1} but {failed_example} failed in {context2}? What's the fundamental difference?",
        "Analyze {trend} through {lens1} and {lens2} lenses. Where do these frameworks agree? Where do they diverge fundamentally?",
        "What's the actual difference between {similar_concept1} and {similar_concept2}? Everyone conflates them. Use {specific_example}.",
    ],
    "howto_complex": [
        "How do I {difficult_goal} when {constraint1} and {constraint2}? I've tried {failed_approach} but {problem}.",
        "Teach me {skill} from {starting_level} to {target_level}. What do people get wrong? What's the {critical_insight}?",
        "Step-by-step: {complex_process} while maintaining {requirement1} and {requirement2}. What breaks if I skip {step}?",
        "I need to {objective} but {obstacle1} and {obstacle2}. Path forward? Alternatives I'm missing?",
    ],
    "meta_analysis": [
        "Why do people think {common_belief} when {contradictory_evidence}? What's the underlying {cognitive_bias}?",
        "Analyze the evolution of {field} from {time1} to {time2}. What changed fundamentally vs superficially? Implications for {time3}?",
        "What are the {n} levels of understanding {topic}? Where do beginners get stuck? What insight unlocks {advanced_level}?",
        "Unpack {jargon_term} for {audience}. Why is it confusing? What's the clearer way to think about it?",
    ],
    # Simple questions (low complexity)
    "simple_factual": [
        "What is {simple_concept}?",
        "Define {term} in simple terms.",
        "Who is {notable_person}?",
        "When was {historical_event}?",
        "Where is {location}?",
    ],
    "simple_howto": [
        "How do I {basic_task}?",
        "What are the steps to {simple_process}?",
        "Show me how to {basic_action}.",
        "Explain {simple_concept} like I'm 5.",
    ],
    "simple_comparison": [
        "What's the difference between {simple_term1} and {simple_term2}?",
        "Compare {option_a} vs {option_b}.",
        "Which is better: {choice1} or {choice2}?",
    ],
    "simple_recommendation": [
        "Recommend a good {item_type}.",
        "What's a good way to {goal}?",
        "Suggest some {thing_category}.",
        "Best {product_type} for {use}?",
    ],
}

# Vocabulary for filling templates - specific technical terms
VOCABULARY = {
    # Algorithms and data structures
    "algo1": ["Jaro-Winkler", "BM25", "TF-IDF", "cosine similarity", "Levenshtein", "HNSW"],
    "algo2": ["Levenshtein", "Jaccard similarity", "semantic embeddings", "FAISS", "Annoy", "ScaNN"],
    "domain": ["securities matching", "entity resolution", "duplicate detection", "fuzzy search", "name matching"],

    # Technologies and frameworks
    "tech1": ["Apache Flink", "Kafka Streams", "Spark Streaming", "Redis", "PostgreSQL", "FastAPI"],
    "tech2": ["Bytewax", "Kafka", "Pulsar", "DynamoDB", "MongoDB", "gRPC"],
    "tech3": ["Milvus", "Pinecone", "Weaviate", "Qdrant", "ChromaDB", "pgvector"],
    "tech4": ["Kubernetes", "Prometheus", "Grafana", "ArgoCD", "Istio"],
    "technology": ["PyFlink", "asyncio", "Ray", "Dask", "Prefect", "Airflow"],

    # Systems and components
    "system": ["real-time recommendation engine", "fraud detection pipeline", "event streaming platform", "vector search system", "CDC pipeline"],
    "component": ["ingestion layer", "processing engine", "state store", "API gateway", "feature store"],
    "component1": ["data ingestion", "stream processing", "state management", "API layer"],
    "component2": ["real-time analytics", "caching layer", "vector indexing", "query engine"],
    "component3": ["monitoring", "alerting", "backup", "disaster recovery"],

    # Features and requirements
    "feature": ["streaming support", "embeddings", "windowing", "exactly-once semantics", "vector similarity search"],
    "feature1": ["real-time embedding generation", "incremental indexing", "distributed tracing", "A/B testing"],
    "feature2": ["multi-tenancy", "rate limiting", "caching", "auto-scaling", "backpressure handling"],
    "requirement": ["sub-second latency", "exactly-once processing", "horizontal scalability", "99.99% uptime"],

    # Problems and challenges
    "problem": ["high memory usage", "slow query performance", "backpressure", "state explosion", "cold start latency"],
    "symptom": ["requests timing out", "memory leaks", "connection pool exhaustion", "message lag", "CPU throttling"],
    "challenge": ["zero-downtime migration", "data consistency", "state migration", "backward compatibility"],
    "error": ["OutOfMemoryError", "TimeoutException", "ConnectionRefused", "DeadlineExceeded", "ResourceExhausted"],
    "error_pattern": ["NullPointerException in executor", "ConcurrentModificationException", "StackOverflowError"],

    # Architecture patterns
    "pattern": ["Event Sourcing", "CQRS", "Saga pattern", "Circuit Breaker", "Bulkhead", "Strangler Fig"],
    "pattern1": ["singleton", "factory", "observer", "strategy", "repository"],
    "pattern2": ["dependency injection", "builder", "adapter", "decorator", "facade"],

    # Data and processing
    "data_type": ["click events", "financial transactions", "sensor data", "user profiles", "time series"],
    "source": ["Kafka", "Kinesis", "PostgreSQL CDC", "S3", "API webhooks"],
    "processing": ["windowed aggregation", "enrichment", "deduplication", "filtering", "transformation"],
    "sink": ["Elasticsearch", "Snowflake", "S3", "Redis", "PostgreSQL"],
    "transform": ["normalization", "aggregation", "enrichment", "schema evolution"],
    "destination": ["data warehouse", "feature store", "cache", "search index"],

    # Scale and metrics
    "scale": ["1M QPS", "10TB/day", "100K concurrent users", "PB-scale storage"],
    "volume": ["100K", "1M", "10M", "100M"],
    "size1": ["2GB", "500MB", "1GB", "100MB"],
    "size2": ["8GB", "4GB", "16GB", "32GB"],
    "time": ["24 hours", "6 hours", "1 week", "48 hours"],

    # Options and alternatives
    "option1": ["Apache Flink", "Redis Streams", "PostgreSQL", "REST API", "gRPC"],
    "option2": ["Bytewax", "Kafka Streams", "DynamoDB", "GraphQL", "WebSocket"],
    "old_tech": ["monolithic Python app", "MySQL", "REST API", "EC2 instances"],
    "new_tech": ["microservices", "PostgreSQL", "gRPC", "Kubernetes"],

    # Constraints and properties
    "constraint": ["strict ordering guarantees", "exactly-once semantics", "sub-100ms p99", "ACID transactions"],
    "property1": ["consistency", "durability", "availability", "partition tolerance"],
    "property2": ["low latency", "high throughput", "fault tolerance", "horizontal scalability"],

    # Development aspects
    "language": ["Python", "Rust", "Go", "Java", "TypeScript"],
    "function": ["event handler", "aggregator", "mapper", "reducer", "filter"],
    "aspect": ["memory safety", "error handling", "testability", "maintainability", "performance"],
    "principle": ["DRY", "SOLID", "YAGNI", "separation of concerns", "single responsibility"],

    # GitHub and repos (real trending repos from https://github.com/trending?since=monthly&spoken_language_code=en)
    "repo_url": [
        "https://github.com/microsoft/Web-Dev-For-Beginners",
        "https://github.com/reflex-dev/reflex",
        "https://github.com/codecrafters-io/build-your-own-x",
        "https://github.com/juanfont/headscale",
        "https://github.com/lima-vm/lima",
        "https://github.com/jellyfin/jellyfin",
        "https://github.com/helix-editor/helix",
        "https://github.com/NVIDIA/Megatron-LM",
        "https://github.com/servo/servo",
        "https://github.com/jaywcjlove/awesome-mac",
        "https://github.com/usememos/memos",
        "https://github.com/rclone/rclone",
        "https://github.com/ChristianLempa/boilerplates",
        "https://github.com/tokio-rs/tokio",
        "https://github.com/swiftlang/swift",
        "https://github.com/hoppscotch/hoppscotch",
        "https://github.com/tailscale/tailscale",
        "https://github.com/resend/react-email",
        "https://github.com/projectdiscovery/nuclei-templates",
        "https://github.com/GopeedLab/gopeed",
        "https://github.com/Project-MONAI/MONAI",
        "https://github.com/microsoft/PowerToys",
        "https://github.com/invoke-ai/InvokeAI",
    ],
    "improvement": ["refactoring", "performance optimization", "better error handling", "monitoring"],

    # Debugging
    "condition": ["high load", "concurrent writes", "network partition", "pod restart", "cache invalidation"],
    "checked_items": ["logs", "metrics", "traces", "config files", "network connectivity"],
    "location": ["request handler", "database query", "serialization", "network I/O", "disk write"],
    "finding": ["retained heap in cache", "growing thread pool", "unclosed connections"],
    "metric_pattern": ["memory spike", "latency increase", "error rate surge", "throughput drop"],

    # Creative
    "genre": ["sci-fi", "mystery", "technical thriller"],
    "scenario": ["AI debugging itself", "distributed system achieving consciousness", "time-traveling debugger"],
    "technical_concept": ["eventual consistency", "CAP theorem", "Byzantine fault tolerance"],
    "complex_concept": ["vector embeddings", "transformer architecture", "RAFT consensus"],
    "audience": ["junior developers", "product managers", "executives", "data scientists"],
    "api": ["vector search API", "streaming API", "GraphQL API", "WebSocket server"],
    "skill_level": ["beginner", "intermediate", "advanced"],

    # Additional
    "use_case": ["real-time recommendations", "fraud detection", "log aggregation", "event sourcing"],
    "concept": ["backpressure", "eventual consistency", "vector embeddings", "stream processing"],
    "resource": ["connection pool", "thread pool", "memory buffer", "file descriptor"],
    "req1": ["sub-second latency", "horizontal scalability", "exactly-once delivery"],
    "req2": ["fault tolerance", "observability", "cost efficiency"],
    "req3": ["multi-region support", "zero-downtime deploys", "data encryption"],

    # Creative writing
    "character1": ["a retired astronaut", "a jazz musician", "an AI researcher", "a journalist", "a street artist"],
    "character2": ["a philosopher", "a quantum physicist", "a war veteran", "a child prodigy", "their future self"],
    "discovery": ["an encrypted message from 50 years ago", "they can see other people's memories", "the city doesn't exist on any map"],
    "tone": ["noir and cynical", "hopeful but melancholy", "darkly comedic", "urgently mysterious"],
    "aspect": ["unreliable narration", "sensory details", "internal monologue", "environmental storytelling"],
    "philosophical_topic": ["free will vs determinism", "the nature of consciousness", "ethical AI", "meaning in a finite universe"],
    "position1": ["determinism", "materialism", "utilitarianism", "moral relativism"],
    "position2": ["libertarian free will", "dualism", "deontological ethics", "moral realism"],
    "length": ["500-word", "flash fiction", "one-page", "micro"],
    "theme": ["isolation in hyperconnectivity", "the cost of progress", "identity in flux", "power and complicity"],
    "setting": ["a space station near Jupiter", "Tokyo in 2045", "a monastery in the Himalayas", "an abandoned theme park"],
    "scene": ["a crowded subway at rush hour", "an empty hospital corridor at 3am", "a protest turning violent", "the last sunset on Earth"],
    "narrator": ["an omniscient entity observing humanity", "a building that's witnessed 200 years", "a dying AI", "a synesthete"],
    "sensory_aspect": ["smell and taste", "texture and temperature", "sound and silence", "light and shadow"],
    "mood": ["creeping dread", "bittersweet nostalgia", "electric anticipation", "profound loneliness"],

    # Business
    "business_type": ["B2B SaaS", "consumer marketplace", "fintech startup", "D2C brand", "enterprise software"],
    "competitor": ["Stripe", "Salesforce", "Shopify", "a well-funded competitor", "market incumbent"],
    "market": ["SMB accounting", "developer tools", "enterprise CRM", "consumer fintech", "logistics"],
    "advantage": ["better UX", "lower pricing", "superior integrations", "brand recognition", "network effects"],
    "strategy1": ["double down on our niche", "expand product surface area", "compete on price"],
    "strategy2": ["focus on enterprise", "build marketplace dynamics", "vertical integration"],
    "company": ["OpenAI", "Stripe", "Notion", "Figma", "Linear"],
    "established_player": ["Microsoft", "Adobe", "Oracle", "SAP", "Salesforce"],
    "moat": ["network effects", "switching costs", "brand", "proprietary data", "distribution"],
    "capability": ["payment processing", "analytics platform", "fraud detection", "recommendation engine"],
    "cost1": ["$2M over 18 months", "$500K + 4 engineers", "$1M/year ongoing"],
    "cost2": ["$200K/year", "$50K/month", "3% of revenue"],
    "product": ["AI coding assistant", "project management tool", "expense management software", "CRM platform"],
    "price_point1": ["$99/month", "$10/user/month", "freemium", "$1K/month"],
    "price_point2": ["$499/month", "$50/user/month", "$5K/month enterprise"],
    "differentiator": ["better AI", "simpler UX", "deeper integrations", "industry-specific features"],

    # Philosophy
    "ethical_dilemma": ["lying to save a life", "stealing to feed your family", "sacrificing one to save many", "violating privacy for security"],
    "framework1": ["Kantian deontology", "virtue ethics", "consequentialism", "care ethics"],
    "framework2": ["utilitarianism", "rights-based ethics", "social contract theory", "pragmatism"],
    "philosophical_concept": ["the hard problem of consciousness", "Gettier problems", "the is-ought gap", "the frame problem"],
    "modern_issue": ["AI consciousness", "climate policy", "genetic engineering", "surveillance capitalism", "social media ethics"],
    "philosopher1": ["Kant", "Mill", "Aristotle", "Rawls", "Singer"],
    "philosopher2": ["Nietzsche", "Hume", "Nozick", "Foucault", "Habermas"],
    "variation": ["with self-driving cars", "where pulling the lever also benefits you", "in VR where nothing is real", "with quantum uncertainty"],
    "concept1": ["correlation and causation", "necessary and sufficient conditions", "a priori and a posteriori", "validity and soundness"],
    "concept2": ["causation and correlation", "sufficient and necessary conditions", "a posteriori and a priori", "soundness and validity"],
    "concrete_example": ["medical studies", "legal reasoning", "scientific theories", "everyday decisions"],

    # Science
    "scientific_concept": ["quantum entanglement", "evolution by natural selection", "general relativity", "emergence", "neuroplasticity"],
    "forbidden_term": ["quantum", "evolution", "spacetime", "complexity", "neurons"],
    "real_world_example": ["GPS satellites", "antibiotic resistance", "black holes", "ant colonies", "learning a language"],
    "hypothetical_change": ["gravity was slightly stronger", "light speed was slower", "atoms couldn't form bonds", "time had two dimensions"],
    "natural_law": ["physics", "chemistry", "biology", "thermodynamics"],
    "consequence1": ["stars couldn't form", "no chemical reactions", "no life", "no causality"],
    "consequence2": ["universe collapse", "perpetual motion", "instant evolution", "time loops"],
    "consequence3": ["no elements beyond hydrogen", "frozen universe", "consciousness impossible", "multiple histories coexist"],
    "scientific_debate": ["nature vs nurture", "reductionism vs holism", "many-worlds interpretation", "consciousness theories"],
    "side1": ["genetic determinists", "reductionists", "many-worlds proponents", "illusionists about consciousness"],
    "side2": ["environmental emphasis", "emergentists", "Copenhagen interpretation", "consciousness realists"],
    "phenomenon": ["why we sleep", "how anesthesia works", "what causes gravity", "how life began", "why we age"],

    # Analysis
    "approach1": ["top-down planning", "test-driven development", "waterfall", "centralized control", "optimization"],
    "approach2": ["bottom-up emergence", "behavior-driven development", "agile", "distributed autonomy", "satisficing"],
    "problem": ["project management", "code quality", "team collaboration", "innovation", "decision making"],
    "stress": ["scaling to 100x", "regulatory changes", "team turnover", "economic downturn", "technical debt"],
    "successful_example": ["Spotify's model", "FAANG hiring", "Toyota's lean", "Netflix culture"],
    "failed_example": ["Yahoo's strategy", "Quibi", "Google+", "Microsoft's mobile"],
    "context1": ["startup environment", "2010s tech boom", "manufacturing", "streaming video"],
    "context2": ["enterprise setting", "2020s market", "services", "social network"],
    "trend": ["remote work", "AI automation", "climate tech", "decentralization", "creator economy"],
    "lens1": ["economic", "sociological", "technological", "historical", "psychological"],
    "lens2": ["political", "cultural", "environmental", "ethical", "systems theory"],
    "similar_concept1": ["machine learning and AI", "blockchain and cryptocurrency", "async and concurrent", "UX and UI"],
    "similar_concept2": ["AI and machine learning", "cryptocurrency and blockchain", "concurrent and parallel", "UI and UX"],
    "specific_example": ["recommendation systems", "Bitcoin vs Ethereum", "Python asyncio", "Figma interface"],

    # How-to
    "difficult_goal": ["change careers to tech", "build an audience", "learn a new language fluently", "start a company"],
    "constraint1": ["no formal education", "full-time job", "limited budget", "no network"],
    "constraint2": ["family obligations", "no prior experience", "geographic limitations", "time pressure"],
    "failed_approach": ["online courses without practice", "networking without value-add", "immersion without structure", "side projects without shipping"],
    "skill": ["technical writing", "public speaking", "data analysis", "product thinking", "negotiation"],
    "starting_level": ["complete beginner", "basic familiarity", "intermediate practitioner"],
    "target_level": ["professional competence", "expert", "world-class"],
    "critical_insight": ["deliberate practice principle", "network effects", "compound learning", "feedback loops"],
    "complex_process": ["database migration", "organizational change", "product launch", "research publication"],
    "objective": ["scale from 10 to 100 users", "raise Series A", "publish research", "change company culture"],
    "obstacle1": ["technical debt", "team misalignment", "market timing", "regulatory hurdles"],
    "obstacle2": ["resource constraints", "competitive pressure", "quality requirements", "stakeholder resistance"],
    "step": ["testing phase", "stakeholder alignment", "data migration", "change management"],

    # Meta-analysis
    "common_belief": ["you need 10,000 hours to master something", "IQ determines success", "disruption always wins", "first-mover advantage"],
    "contradictory_evidence": ["evidence shows deliberate practice matters more", "grit and mindset predict outcomes better", "incumbents often win", "fast followers outperform pioneers"],
    "cognitive_bias": ["confirmation bias", "survivorship bias", "availability heuristic", "sunk cost fallacy"],
    "field": ["machine learning", "management theory", "urban planning", "psychology", "economics"],
    "time1": ["1990s", "early 2000s", "2010", "pre-internet era"],
    "time2": ["today", "2020s", "post-pandemic", "AI era"],
    "time3": ["next decade", "2030", "post-AGI", "climate-constrained future"],
    "n": ["three", "five", "four"],
    "topic": ["programming", "leadership", "investing", "scientific method", "writing"],
    "advanced_level": ["expert", "master level", "professional grade", "deep understanding"],
    "jargon_term": ["serverless", "quantum supremacy", "product-market fit", "paradigm shift", "first principles"],

    # Simple questions vocabulary
    "simple_concept": ["Python", "HTTP", "JSON", "databases", "APIs", "Git", "CSS", "HTML", "SQL"],
    "term": ["variable", "function", "class", "API", "database", "server", "client", "algorithm"],
    "notable_person": ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Tim Berners-Lee", "Linus Torvalds"],
    "historical_event": ["the first computer built", "the internet invented", "the iPhone released", "Bitcoin created"],
    "location": ["Silicon Valley", "MIT", "Stanford", "CERN", "Bell Labs"],
    "basic_task": ["print to console", "create a variable", "write a loop", "define a function", "import a module"],
    "simple_process": ["install Python", "create a Git repository", "write a for loop", "define a class"],
    "basic_action": ["create a list", "write an if statement", "use a dictionary", "handle errors"],
    "simple_term1": ["list", "class", "function", "git", "HTTP"],
    "simple_term2": ["tuple", "object", "method", "svn", "HTTPS"],
    "option_a": ["Python", "MySQL", "REST", "monolith"],
    "option_b": ["JavaScript", "PostgreSQL", "GraphQL", "microservices"],
    "choice1": ["tabs", "spaces", "camelCase", "React"],
    "choice2": ["spaces", "tabs", "snake_case", "Vue"],
    "item_type": ["book on Python", "tutorial for beginners", "IDE for web development", "course on databases"],
    "goal": ["learn programming", "start a blog", "improve typing speed", "organize my code"],
    "thing_category": ["Python libraries", "coding tutorials", "tech podcasts", "programming books"],
    "product_type": ["text editor", "database", "framework", "hosting service"],
    "use": ["beginners", "web development", "data science", "mobile apps"],
}

# Map categories to complexity ranges
CATEGORY_COMPLEXITY = {
    # Simple (0.1-0.3)
    "simple_factual": (0.1, 0.3),
    "simple_howto": (0.1, 0.3),
    "simple_comparison": (0.1, 0.3),
    "simple_recommendation": (0.1, 0.3),

    # Medium (0.4-0.6)
    "technical_specific": (0.4, 0.6),
    "creative_writing": (0.4, 0.6),
    "business_strategic": (0.5, 0.7),
    "analysis_comparative": (0.4, 0.6),
    "howto_complex": (0.4, 0.6),

    # Complex (0.7-1.0)
    "multipart_architecture": (0.7, 0.9),
    "code_analysis": (0.6, 0.8),
    "data_engineering": (0.7, 0.9),
    "github_review": (0.7, 0.9),
    "system_design": (0.8, 1.0),
    "debugging_complex": (0.7, 0.9),
    "philosophical_analysis": (0.5, 0.7),
    "science_complex": (0.6, 0.8),
    "meta_analysis": (0.6, 0.8),
    "creative_technical": (0.5, 0.7),
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
        reference_probability: float = 0.7,
    ) -> None:
        """Initialize the generator.

        Args:
            reference_model: Model to use for generating reference answers
            seed: Random seed for reproducibility
            reference_probability: Probability of generating reference answer (0.0-1.0)
        """
        self.reference_model = reference_model
        self.seed = seed
        self.reference_probability = reference_probability
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

                # Assign complexity based on category type
                if category in CATEGORY_COMPLEXITY:
                    min_complexity, max_complexity = CATEGORY_COMPLEXITY[category]
                    complexity = random.uniform(min_complexity, max_complexity)
                else:
                    # Fallback for unmapped categories
                    complexity = random.choice([0.3, 0.5, 0.8])

                # Probabilistically generate reference answer
                reference_answer = None
                if random.random() < self.reference_probability:
                    try:
                        result = await agent.run(query_text)
                        reference_answer = str(result.output)  # Use .output, not .data
                    except Exception as e:
                        # Fallback if reference generation fails - set to None instead of error message
                        reference_answer = None

                query = BenchmarkQuery(
                    query_text=query_text,
                    reference_answer=reference_answer,
                    metadata={
                        "category": category,  # For analysis only - not exposed to Conduit
                        "complexity": complexity,  # For analysis only - not exposed to Conduit
                    },
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
