#!/usr/bin/env python3
"""
Dynamic Knowledge Graph - Evolution Demo

This script demonstrates how the knowledge graph grows and evolves
over time as the agent processes queries and tool outputs.

Shows:
1. Empty KG at start
2. Learning from user queries
3. Learning from web search results
4. Learning from sentiment analysis
5. Learning from news
6. Query answering using accumulated knowledge
7. Avoiding redundant information
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.block3_execution.tools.dynamic_knowledge_graph import (
    DynamicKnowledgeGraph
)
from src.block3_execution.tools.kg_integration import (
    get_kg_integration
)


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_stats(kg: DynamicKnowledgeGraph):
    """Print KG statistics."""
    stats = kg.get_stats()
    print(f"📊 Current Stats:")
    print(f"   Entities: {stats['total_entities']}")
    print(f"   Relations: {stats['total_relations']}")
    print(f"   Entity Types: {stats.get('entity_types', {})}")
    print(f"   Relation Types: {stats.get('relation_types', {})}")
    print()


def simulate_user_interaction_1(kg: DynamicKnowledgeGraph):
    """Simulate first user interaction."""
    print_header("INTERACTION 1: User asks about Pathaan")

    query = "Tell me about Pathaan box office collection"
    print(f"👤 User Query: {query}\n")

    # Agent would call web search here, simulate the result
    search_result = """
    🔍 Web Search Results:

    Pathaan is a 2023 Indian action thriller film starring Shah Rukh Khan.
    The movie collected ₹57 crores on its opening day in India.
    Directed by Siddharth Anand, the film crossed ₹1000 crores worldwide.
    Deepika Padukone and John Abraham also star in the film.

    Box office: ₹1055 crores worldwide (as of Feb 2023)
    """

    print("🤖 Agent Response:")
    print(search_result)
    print()

    # Learn from query and result
    print("🧠 Learning from interaction...")
    kg.learn_from_text(query, source="user_query")
    kg.learn_from_text(search_result, source="web_search")
    print()

    print_stats(kg)
    time.sleep(1)


def simulate_user_interaction_2(kg: DynamicKnowledgeGraph):
    """Simulate second user interaction."""
    print_header("INTERACTION 2: User asks about Shah Rukh Khan")

    query = "What movies has Shah Rukh Khan acted in recently?"
    print(f"👤 User Query: {query}\n")

    # Check KG first
    print("🔍 Checking Knowledge Graph first...")
    kg_result = kg.query(query)
    print(kg_result)
    print()

    # Simulate additional search for more info
    search_result = """
    🔍 Additional Web Search:

    Shah Rukh Khan recent movies:
    - Pathaan (2023) - Action thriller, ₹1055 cr worldwide
    - Jawan (2023) - Action drama, ₹1148 cr worldwide
    - Dunki (2023) - Comedy drama, directed by Rajkumar Hirani

    SRK is one of Bollywood's biggest stars with over 100 films.
    """

    print("🤖 Agent provides updated info:")
    print(search_result)
    print()

    # Learn new info
    print("🧠 Learning new information...")
    kg.learn_from_text(search_result, source="web_search")
    print()

    print_stats(kg)
    time.sleep(1)


def simulate_user_interaction_3(kg: DynamicKnowledgeGraph):
    """Simulate third user interaction."""
    print_header("INTERACTION 3: Sentiment analysis of movie review")

    review = "Jawan is an absolutely fantastic movie! Shah Rukh Khan delivers an amazing performance."
    print(f"👤 User Query: Analyze sentiment of this review:\n   '{review}'\n")

    # Simulate sentiment analysis result
    sentiment_result = """
    💭 Sentiment Analysis:

    Overall Sentiment: Highly Positive (9.5/10)

    Key emotions:
    - Excitement and enthusiasm
    - Appreciation for acting
    - Strong positive recommendation

    Aspects praised:
    - Shah Rukh Khan's performance
    - Overall movie quality
    """

    print("🤖 Agent Response:")
    print(sentiment_result)
    print()

    # Learn from review and sentiment
    print("🧠 Learning from review and sentiment...")
    combined = f"{review}\n\n{sentiment_result}"
    kg.learn_from_text(combined, source="sentiment_analysis")
    print()

    print_stats(kg)
    time.sleep(1)


def simulate_user_interaction_4(kg: DynamicKnowledgeGraph):
    """Simulate fourth user interaction."""
    print_header("INTERACTION 4: Query that KG can answer directly")

    query = "What do you know about Deepika Padukone?"
    print(f"👤 User Query: {query}\n")

    # Query KG
    print("📚 Answering from Knowledge Graph (no external search needed):")
    result = kg.query(query)
    print(result)
    print()

    print("✅ No external API calls needed - answered from memory!")
    print()

    print_stats(kg)
    time.sleep(1)


def simulate_user_interaction_5(kg: DynamicKnowledgeGraph):
    """Simulate fifth user interaction - news learning."""
    print_header("INTERACTION 5: Learning from news aggregator")

    print("📰 Agent fetches latest Bollywood news...\n")

    news = """
    Latest Bollywood News:

    1. Shah Rukh Khan to receive lifetime achievement award at Filmfare
       The superstar will be honored for his 30+ year career

    2. Jawan crosses ₹1100 crores, becomes highest grossing Hindi film
       Directed by Atlee, starring Shah Rukh Khan and Nayanthara

    3. Deepika Padukone announces new project with Hrithik Roshan
       Filming to begin in 2024
    """

    print("🤖 Agent Response:")
    print(news)
    print()

    # Learn from news
    print("🧠 Learning from news articles...")
    kg.learn_from_text(news, source="news_aggregator")
    print()

    print_stats(kg)
    time.sleep(1)


def demonstrate_redundancy_handling(kg: DynamicKnowledgeGraph):
    """Demonstrate how KG handles redundant information."""
    print_header("REDUNDANCY TEST: Adding duplicate information")

    # Try to add same info again
    duplicate_info = "Pathaan is a 2023 film starring Shah Rukh Khan. It collected ₹1055 crores."

    print(f"📝 Attempting to add duplicate info:\n   '{duplicate_info}'\n")

    initial_count = len(kg.entities)
    print(f"Entities before: {initial_count}")

    kg.learn_from_text(duplicate_info, source="duplicate_test")

    final_count = len(kg.entities)
    print(f"Entities after: {final_count}")

    if final_count == initial_count:
        print("✅ No new entities created - redundancy avoided!")
    else:
        new_entities = final_count - initial_count
        print(f"ℹ️  Added {new_entities} new entity/entities (may be minor variations)")
    print()


def query_evolved_kg(kg: DynamicKnowledgeGraph):
    """Show queries on evolved KG."""
    print_header("FINAL QUERIES: Testing evolved knowledge")

    queries = [
        "Tell me about Shah Rukh Khan",
        "What are the highest grossing movies?",
        "Information about Pathaan",
        "Deepika Padukone movies"
    ]

    for query in queries:
        print(f"❓ Query: {query}")
        result = kg.query(query)
        print(result)
        print("-" * 80)
        print()


def main():
    """Run the full demo."""
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║          DYNAMIC KNOWLEDGE GRAPH - Evolution Demonstration              ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    print("\nThis demo shows how the KG grows from empty to knowledgeable over time.")
    print("Watch as it learns from queries, searches, sentiment, and news!")

    # Create fresh KG (for demo, use temp storage)
    import tempfile
    temp_dir = tempfile.mkdtemp()
    print(f"\n📁 Using temporary storage: {temp_dir}")

    kg = DynamicKnowledgeGraph(storage_path=temp_dir)

    print_header("INITIAL STATE: Empty Knowledge Graph")
    print_stats(kg)
    time.sleep(2)

    # Simulate user interactions
    simulate_user_interaction_1(kg)
    simulate_user_interaction_2(kg)
    simulate_user_interaction_3(kg)
    simulate_user_interaction_4(kg)
    simulate_user_interaction_5(kg)

    # Test redundancy handling
    demonstrate_redundancy_handling(kg)

    # Final queries
    query_evolved_kg(kg)

    # Final stats
    print_header("FINAL STATE: Evolved Knowledge Graph")
    print_stats(kg)

    print("🎉 Demo Complete!")
    print("\nKey Takeaways:")
    print("✅ KG starts empty and grows organically")
    print("✅ Learns from all tool outputs automatically")
    print("✅ Can answer queries without external calls")
    print("✅ Handles redundancy gracefully")
    print("✅ Persists to disk for long-term memory")
    print("\n" + "=" * 80)

    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary storage")
    except:
        pass


if __name__ == "__main__":
    main()
