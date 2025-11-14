"""
Knowledge Graph Integration Layer

This module connects all tools to the dynamic knowledge graph.
Every tool output automatically updates the KG, building memory over time.

Flow:
1. Tool executes (search, sentiment, news, etc.)
2. Result is returned to user
3. Result is ALSO fed to KG for learning
4. KG grows and evolves

Future queries can leverage this accumulated knowledge!
"""

from typing import Dict, Any, Optional, Callable
from .dynamic_knowledge_graph import create_dynamic_knowledge_graph


class KGIntegration:
    """
    Integrates all tools with the knowledge graph.

    Wraps tool calls to automatically learn from outputs.
    """

    def __init__(self):
        """Initialize integration."""
        self.kg = create_dynamic_knowledge_graph()
        self.learning_enabled = True

    def wrap_tool(
        self,
        tool_func: Callable,
        tool_name: str,
        learn_from_input: bool = True,
        learn_from_output: bool = True
    ) -> Callable:
        """
        Wrap a tool function to enable KG learning.

        Args:
            tool_func: Tool function to wrap
            tool_name: Tool name for source tracking
            learn_from_input: Whether to learn from input
            learn_from_output: Whether to learn from output

        Returns:
            Wrapped function
        """
        def wrapped_tool(query: str, context: Optional[Dict] = None, **kwargs):
            # Execute the tool
            result = tool_func(query, context, **kwargs)

            if not self.learning_enabled:
                return result

            # Learn from input query
            if learn_from_input and query:
                try:
                    self.kg.learn_from_text(
                        query,
                        source=f"{tool_name}_query",
                        context=context
                    )
                except Exception as e:
                    print(f"KG learning from input failed: {e}")

            # Learn from output
            if learn_from_output and result and isinstance(result, str):
                try:
                    # Extract first 2000 chars (LLM limit)
                    text_to_learn = result[:2000]
                    self.kg.learn_from_text(
                        text_to_learn,
                        source=tool_name,
                        context={'query': query, **(context or {})}
                    )
                except Exception as e:
                    print(f"KG learning from output failed: {e}")

            return result

        return wrapped_tool

    def learn_from_search(
        self,
        query: str,
        search_results: str,
        context: Optional[Dict] = None
    ):
        """
        Learn from web search results.

        Extracts entities from both query and scraped content.
        """
        self.kg.learn_from_text(query, source="search_query", context=context)
        self.kg.learn_from_text(search_results, source="web_search", context=context)

    def learn_from_sentiment(
        self,
        text: str,
        sentiment_result: str,
        context: Optional[Dict] = None
    ):
        """
        Learn from sentiment analysis.

        Stores sentiment as entity properties.
        """
        # Extract entities from text
        self.kg.learn_from_text(text, source="sentiment_input", context=context)

        # Store sentiment as metadata
        # (LLM will extract entities and add sentiment properties)
        combined = f"{text}\nSentiment: {sentiment_result}"
        self.kg.learn_from_text(combined, source="sentiment_analysis", context=context)

    def learn_from_news(
        self,
        news_results: str,
        context: Optional[Dict] = None
    ):
        """Learn from news aggregator results."""
        self.kg.learn_from_text(news_results, source="news_aggregator", context=context)

    def query_kg_first(self, query: str) -> Optional[str]:
        """
        Query KG before calling other tools.

        Returns:
            Answer from KG if found, None otherwise
        """
        result = self.kg.query(query)

        # Check if we have useful information
        if "No information found" in result or "No relevant information" in result:
            return None

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get KG statistics."""
        return self.kg.get_stats()


# Global integration instance
_kg_integration = None


def get_kg_integration() -> KGIntegration:
    """Get global KG integration instance."""
    global _kg_integration
    if _kg_integration is None:
        _kg_integration = KGIntegration()
    return _kg_integration


def should_use_kg_first(query: str) -> bool:
    """
    Determine if we should query KG first.

    Returns True for factual queries that KG might know.
    """
    query_lower = query.lower()

    # Factual query indicators
    factual_words = [
        'who', 'what', 'when', 'where',
        'movies', 'acted', 'directed',
        'born', 'age', 'career',
        'box office', 'rating', 'awards'
    ]

    for word in factual_words:
        if word in query_lower:
            return True

    return False


# Enhanced tool wrappers that integrate with KG

def kg_aware_google_search(query: str, context: Optional[Dict] = None, **kwargs) -> str:
    """
    Google search that learns from results.

    Flow:
    1. Check KG first
    2. If not found, do web search
    3. Learn from search results
    4. Return combined answer
    """
    from .google_search import google_search_tool

    kg_int = get_kg_integration()

    # Check KG first for factual queries
    if should_use_kg_first(query):
        kg_result = kg_int.query_kg_first(query)
        if kg_result:
            # Found in KG! But still search for updates
            search_result = google_search_tool(query, context, **kwargs)

            # Learn from new search
            kg_int.learn_from_search(query, search_result, context)

            # Combine KG knowledge + new search
            return f"{kg_result}\n\n---\n\n🔍 Latest Updates:\n{search_result}"

    # Not in KG, do search
    result = google_search_tool(query, context, **kwargs)

    # Learn from results
    kg_int.learn_from_search(query, result, context)

    return result


def kg_aware_sentiment_analyzer(query: str, context: Optional[Dict] = None, **kwargs) -> str:
    """Sentiment analyzer that stores results in KG."""
    from .sentiment_analyzer import sentiment_analyzer_tool

    kg_int = get_kg_integration()

    # Run sentiment analysis
    result = sentiment_analyzer_tool(query, context, **kwargs)

    # Learn from input and output
    kg_int.learn_from_sentiment(query, result, context)

    return result


def kg_aware_news_aggregator(query: str, context: Optional[Dict] = None, **kwargs) -> str:
    """News aggregator that stores results in KG."""
    from .news_aggregator import news_aggregator_tool

    kg_int = get_kg_integration()

    # Get news
    result = news_aggregator_tool(query, context, **kwargs)

    # Learn from news
    kg_int.learn_from_news(result, context)

    return result


def kg_aware_summarizer(query: str, context: Optional[Dict] = None, **kwargs) -> str:
    """Summarizer that stores entities in KG."""
    from .summarizer import summarizer_tool

    kg_int = get_kg_integration()

    # Get summary
    result = summarizer_tool(query, context, **kwargs)

    # Learn from both input and summary
    if query:
        kg_int.kg.learn_from_text(query, source="summarizer_input", context=context)
    if result:
        kg_int.kg.learn_from_text(result[:1000], source="summarizer_output", context=context)

    return result


# Test harness
if __name__ == "__main__":
    print("=" * 70)
    print("KNOWLEDGE GRAPH INTEGRATION - Test Mode")
    print("=" * 70)

    kg_int = get_kg_integration()

    # Test 1: Learn from simulated search
    print("\n📝 Test 1: Learning from Search")
    print("-" * 70)
    query = "Pathaan box office collection"
    search_result = """
    Pathaan collected 57 crores on day 1. Shah Rukh Khan stars in the lead role.
    The movie crossed 1000 crores worldwide. Directed by Siddharth Anand.
    Deepika Padukone plays the female lead.
    """
    kg_int.learn_from_search(query, search_result)

    # Test 2: Query KG
    print("\n📝 Test 2: Querying KG")
    print("-" * 70)
    result = kg_int.query_kg_first("Tell me about Shah Rukh Khan")
    print(result)

    # Test 3: Learn from sentiment
    print("\n📝 Test 3: Learning from Sentiment")
    print("-" * 70)
    text = "Jawan is an amazing movie with great action sequences"
    sentiment = "Positive sentiment (score: 9/10). Praised for action."
    kg_int.learn_from_sentiment(text, sentiment)

    # Test 4: Statistics
    print("\n📝 Test 4: Statistics")
    print("-" * 70)
    stats = kg_int.get_statistics()
    print(f"Entities: {stats['total_entities']}")
    print(f"Relations: {stats['total_relations']}")
    print(f"Entity Types: {stats['entity_types']}")

    # Test 5: Check if should use KG first
    print("\n📝 Test 5: Should Use KG First?")
    print("-" * 70)
    test_queries = [
        "Who acted in Pathaan?",
        "What is the box office of Jawan?",
        "Latest Bollywood news"
    ]
    for q in test_queries:
        should_use = should_use_kg_first(q)
        print(f"{q}: {'✅ Yes' if should_use else '❌ No'}")

    print("\n" + "=" * 70)
    print("✅ Tests Complete!")
    print("=" * 70)
