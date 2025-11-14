#!/usr/bin/env python3
"""
Test LLM-based query reformulation for web search.

This script tests the new LLM reformulation feature that fixes
the issue of getting irrelevant (Chinese) search results.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.block3_execution.tools.google_search import create_google_search


def test_llm_reformulation():
    """Test LLM query reformulation."""
    print("=" * 80)
    print("LLM QUERY REFORMULATION TEST")
    print("=" * 80)

    tool = create_google_search()

    # Test 1: Original problematic query
    print("\n📝 Test 1: Problematic Query - 'all Bollywood movies'")
    print("-" * 80)

    query = "all Bollywood movies"

    print(f"\n🔍 Original Query: {query}\n")

    # Test with LLM reformulation
    print("Testing WITH LLM reformulation:")
    result = tool.search(
        query=query,
        num_results=3,
        scrape_content=False,  # Faster for testing
        summarize=False,  # Just show raw results
        use_llm_reformulation=True,
        num_parallel_queries=1
    )
    print(result)

    # Test 2: More specific query
    print("\n\n📝 Test 2: More Specific Query")
    print("-" * 80)

    query2 = "list of Bollywood movies released in 2023"

    print(f"\n🔍 Query: {query2}\n")

    result2 = tool.search(
        query=query2,
        num_results=3,
        scrape_content=False,
        summarize=True,  # Use LLM to synthesize
        use_llm_reformulation=True,
        num_parallel_queries=1
    )
    print(result2)

    # Test 3: Multiple parallel queries
    print("\n\n📝 Test 3: Parallel Query Search")
    print("-" * 80)

    query3 = "Shah Rukh Khan movies 2023"

    print(f"\n🔍 Query: {query3}")
    print("   Generating 2 parallel search queries...\n")

    result3 = tool.search(
        query=query3,
        num_results=5,
        scrape_content=False,
        summarize=True,
        use_llm_reformulation=True,
        num_parallel_queries=2  # Generate 2 different queries
    )
    print(result3)

    print("\n" + "=" * 80)
    print("✅ LLM Reformulation Tests Complete!")
    print("=" * 80)


def test_llm_reformulation_only():
    """Test just the query reformulation without actual search."""
    print("\n\n" + "=" * 80)
    print("QUERY REFORMULATION ONLY (No Search)")
    print("=" * 80)

    tool = create_google_search()

    if not tool.openai_client:
        print("\n⚠️  OpenAI client not available. Set OPENAI_API_KEY to test.")
        return

    test_queries = [
        "all Bollywood movies",
        "latest news about Pathaan",
        "Tell me about Shah Rukh Khan recent work",
        "Jawan box office collection worldwide"
    ]

    for query in test_queries:
        print(f"\n📝 Original: {query}")

        # Single reformulation
        reformulated = tool._llm_reformulate_query(query, num_queries=1)
        print(f"   ✨ Reformulated (1): {reformulated}")

        # Multiple reformulations
        reformulated_multi = tool._llm_reformulate_query(query, num_queries=3)
        print(f"   ✨ Reformulated (3):")
        for i, q in enumerate(reformulated_multi, 1):
            print(f"      {i}. {q}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   LLM reformulation will fall back to keyword extraction.")
        print("   Set the environment variable for full functionality.\n")

    # Test reformulation only (fast)
    test_llm_reformulation_only()

    # Ask user if they want to run full search tests
    print("\n" + "=" * 80)
    response = input("\nRun full search tests? (y/n): ").strip().lower()

    if response == 'y':
        test_llm_reformulation()
    else:
        print("Skipped full search tests.")
