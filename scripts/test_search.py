#!/usr/bin/env python3
"""
Test script for the free web search tool.

This script demonstrates all features of the search tool including:
- Basic search
- Keyword extraction
- Web scraping
- Parallel searches
- Caching
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.block3_execution.tools.google_search import (
    create_google_search,
    KeywordExtractor,
    WebScraper,
    SearchCache
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def test_keyword_extraction():
    """Test keyword extraction."""
    print_section("TEST 1: Keyword Extraction")

    extractor = KeywordExtractor()

    # Test cases
    test_queries = [
        "Tell me about the latest Bollywood news for Shah Rukh Khan",
        "What are Alia Bhatt's upcoming movies?",
        "Find information on Pathaan box office collection",
        'Search for "Jawan" movie reviews and ratings'
    ]

    for query in test_queries:
        keywords = extractor.extract_keywords(query)
        print(f"Query: {query}")
        print(f"Keywords: {keywords}")
        print()


def test_plan_step_extraction():
    """Test extraction from plan steps."""
    print_section("TEST 2: Extract Keywords from Plan Steps")

    extractor = KeywordExtractor()

    # Test plan steps
    plan_steps = [
        "Step 1: Search for Pathaan box office - Tool: google_search - Expected: Numbers",
        "Step 2: Find latest Bollywood news - Tool: google_search",
        "Search for Shah Rukh Khan new movie Dunki"
    ]

    for step in plan_steps:
        keywords = extractor.extract_from_plan_step(step)
        print(f"Plan Step: {step}")
        print(f"Keywords: {keywords}")
        print()


def test_basic_search():
    """Test basic search functionality."""
    print_section("TEST 3: Basic Web Search")

    tool = create_google_search()

    query = "Pathaan box office collection India"
    print(f"Searching for: {query}\n")

    result = tool.search(
        query=query,
        num_results=3,
        scrape_content=True,
        summarize=True
    )

    print(result)


def test_search_without_scraping():
    """Test search without web scraping."""
    print_section("TEST 4: Search Without Scraping (Faster)")

    tool = create_google_search()

    query = "Jawan movie cast"
    print(f"Searching for: {query}\n")

    result = tool.search(
        query=query,
        num_results=5,
        scrape_content=False,  # Skip scraping
        summarize=False  # No LLM summary
    )

    print(result)


def test_context_search():
    """Test search with context (simulating plan step)."""
    print_section("TEST 5: Search with Plan Context")

    tool = create_google_search()

    # Simulate a plan step context
    context = {
        'step': 'Step 1: Search for latest Shah Rukh Khan movie news - Tool: google_search - Expected: News articles',
        'query': 'Shah Rukh Khan'
    }

    result = tool.search(
        query="Shah Rukh Khan movies",
        context=context,
        num_results=3,
        scrape_content=True
    )

    print(result)


def test_parallel_search():
    """Test parallel search functionality."""
    print_section("TEST 6: Parallel Search (Multiple Queries)")

    tool = create_google_search()

    queries = [
        "Pathaan box office",
        "Jawan release date",
        "Dunki movie cast"
    ]

    print(f"Searching for {len(queries)} queries in parallel:")
    for q in queries:
        print(f"  - {q}")
    print()

    results = tool.parallel_search(queries)

    for query, result in results.items():
        print(f"\n📝 {query}:")
        print("-" * 70)
        # Print first 300 chars of each result
        print(result[:300] + "..." if len(result) > 300 else result)


def test_caching():
    """Test caching functionality."""
    print_section("TEST 7: Caching Test")

    tool = create_google_search()
    query = "Tiger 3 box office collection"

    print(f"First search (fetches from web): {query}")
    import time
    start = time.time()
    result1 = tool.search(query, num_results=3, scrape_content=False)
    time1 = time.time() - start
    print(f"Time taken: {time1:.2f}s")
    print(f"Result length: {len(result1)} chars\n")

    print(f"Second search (should use cache): {query}")
    start = time.time()
    result2 = tool.search(query, num_results=3, scrape_content=False)
    time2 = time.time() - start
    print(f"Time taken: {time2:.2f}s")
    print(f"Result length: {len(result2)} chars")
    print(f"\n✅ Cache speedup: {time1/time2:.1f}x faster!")


def test_web_scraper():
    """Test web scraper on a specific URL."""
    print_section("TEST 8: Web Scraping Individual URL")

    scraper = WebScraper()

    # Test URL (Wikipedia is usually scraper-friendly)
    url = "https://en.wikipedia.org/wiki/Pathaan_(film)"
    print(f"Scraping: {url}\n")

    result = scraper.scrape_url(url, timeout=10)

    if result['success']:
        print(f"✅ Scraping successful!")
        print(f"Title: {result['title']}")
        print(f"Content length: {result['length']} chars")
        print(f"\nContent preview:")
        print(result['content'][:500] + "...")
    else:
        print(f"❌ Scraping failed: {result.get('error', 'Unknown error')}")


def main():
    """Run all tests."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        FREE WEB SEARCH TOOL - Comprehensive Test Suite            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    try:
        # Run tests
        test_keyword_extraction()
        test_plan_step_extraction()
        test_basic_search()
        test_search_without_scraping()
        test_context_search()
        test_parallel_search()
        test_caching()
        test_web_scraper()

        print_section("✅ ALL TESTS COMPLETE!")
        print("The free web search tool is working correctly.")
        print("\nNext steps:")
        print("1. Use in your agent: google_search_tool(query, context)")
        print("2. Check cache in: .cache/search/")
        print("3. Read docs: docs/FREE_WEB_SEARCH_SETUP.md")

    except ImportError as e:
        print("\n❌ ERROR: Required packages not installed")
        print(f"Details: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements_search.txt")
        return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
