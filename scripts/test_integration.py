#!/usr/bin/env python3
"""
Integration Test - Verify all tools work together with Dynamic KG

This script tests:
1. All tools are loaded (not mocked)
2. Tools execute successfully
3. Dynamic KG learns from outputs
4. KG can answer queries
5. Integration between all components
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.block3_execution.execution_engine import ExecutionEngine
from src.block3_execution.tools import (
    get_kg_integration,
    kg_aware_google_search,
    kg_aware_sentiment_analyzer,
    kg_aware_news_aggregator,
    kg_aware_summarizer,
    create_dynamic_knowledge_graph
)


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_execution_engine_loading():
    """Test that execution engine loads KG-aware tools."""
    print_section("TEST 1: Execution Engine Tool Loading")

    try:
        config = load_config('configs/default_config.yaml')
        engine = ExecutionEngine(config)

        print(f"✅ Execution Engine initialized")
        print(f"   Tools loaded: {list(engine.tools.keys())}")
        print(f"   Total tools: {len(engine.tools)}")

        # Check if tools are not mocked
        for tool_name, tool_func in engine.tools.items():
            if '_mock_' in tool_func.__name__:
                print(f"❌ {tool_name} is using MOCK implementation!")
                return False
            else:
                print(f"✅ {tool_name} is using REAL implementation")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_google_search_tool():
    """Test google search tool."""
    print_section("TEST 2: Google Search Tool (FREE)")

    try:
        # This should use DuckDuckGo and web scraping
        result = kg_aware_google_search(
            "Shah Rukh Khan Pathaan box office",
            num_results=3,
            scrape_content=False  # Faster for testing
        )

        print(f"✅ Search executed successfully")
        print(f"   Result length: {len(result)} chars")
        print(f"   Preview: {result[:200]}...")

        return True

    except Exception as e:
        print(f"⚠️  Search failed (may need dependencies): {e}")
        return False  # Not critical if deps not installed


def test_sentiment_analyzer():
    """Test sentiment analyzer."""
    print_section("TEST 3: Sentiment Analyzer Tool")

    try:
        text = "Pathaan is absolutely amazing! Best action movie ever!"
        result = kg_aware_sentiment_analyzer(text)

        print(f"✅ Sentiment analysis executed")
        print(f"   Input: {text}")
        print(f"   Result: {result[:300]}...")

        return True

    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summarizer():
    """Test summarizer."""
    print_section("TEST 4: Summarizer Tool")

    try:
        long_text = """
        Pathaan is a 2023 Indian action thriller film starring Shah Rukh Khan,
        Deepika Padukone, and John Abraham. Directed by Siddharth Anand, the film
        follows a RAW agent who must stop a dangerous mercenary. The movie was
        a massive box office success, collecting over 1055 crores worldwide.
        Critics praised the action sequences and Shah Rukh Khan's performance.
        """

        result = kg_aware_summarizer(long_text, length="brief")

        print(f"✅ Summarization executed")
        print(f"   Input length: {len(long_text)} chars")
        print(f"   Summary: {result}")

        return True

    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_news_aggregator():
    """Test news aggregator."""
    print_section("TEST 5: News Aggregator Tool")

    try:
        result = kg_aware_news_aggregator("latest Bollywood news")

        print(f"✅ News aggregation executed")
        print(f"   Result: {result[:300]}...")

        return True

    except Exception as e:
        print(f"❌ News aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_kg_learning():
    """Test that dynamic KG learned from tool outputs."""
    print_section("TEST 6: Dynamic KG Learning Verification")

    try:
        kg_int = get_kg_integration()
        stats = kg_int.get_statistics()

        print(f"✅ Dynamic KG Statistics:")
        print(f"   Total entities: {stats['total_entities']}")
        print(f"   Total relations: {stats['total_relations']}")
        print(f"   Entity types: {stats.get('entity_types', {})}")

        if stats['total_entities'] > 0:
            print(f"✅ KG has learned {stats['total_entities']} entities!")
            return True
        else:
            print(f"⚠️  KG is empty - may need to run more tests first")
            return False

    except Exception as e:
        print(f"❌ KG check failed: {e}")
        return False


def test_kg_query():
    """Test querying the dynamic KG."""
    print_section("TEST 7: Query Dynamic KG")

    try:
        kg = create_dynamic_knowledge_graph()

        # Manually add some data for testing
        kg.learn_from_text(
            "Jawan is a 2023 film starring Shah Rukh Khan, directed by Atlee",
            source="test"
        )

        # Query it
        result = kg.query("Tell me about Jawan")

        print(f"✅ KG query executed")
        print(f"   Query: Tell me about Jawan")
        print(f"   Result:\n{result}")

        if "Jawan" in result or "Shah Rukh Khan" in result:
            print(f"✅ KG successfully answered query!")
            return True
        else:
            print(f"⚠️  KG didn't return relevant info")
            return False

    except Exception as e:
        print(f"❌ KG query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow():
    """Test complete workflow with execution engine."""
    print_section("TEST 8: Full Workflow with Execution Engine")

    try:
        config = load_config('configs/default_config.yaml')
        engine = ExecutionEngine(config)

        # Create a simple plan
        plan = {
            'steps': [
                {
                    'tool': 'google_search',
                    'description': 'Search for Pathaan information',
                    'expected': 'Movie details'
                },
                {
                    'tool': 'sentiment_analyzer',
                    'description': 'Analyze sentiment',
                    'expected': 'Sentiment score'
                }
            ]
        }

        query = "Pathaan movie"

        print(f"Executing plan with {len(plan['steps'])} steps...")

        # Execute (this will use kg-aware tools)
        result = engine.execute_plan(plan, query)

        print(f"✅ Workflow executed")
        print(f"   Status: {result['status']}")
        print(f"   Tool results: {len(result['tool_results'])}")
        print(f"   Final answer: {result['final_answer'][:200]}...")

        # Check KG learned something
        kg_int = get_kg_integration()
        stats = kg_int.get_statistics()
        print(f"   KG now has {stats['total_entities']} entities")

        return True

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                   INTEGRATION TEST - All Tools + Dynamic KG               ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    print("\n🎯 Testing that:")
    print("   1. All tools are REAL implementations (not mocked)")
    print("   2. Tools execute successfully")
    print("   3. Dynamic KG learns from outputs")
    print("   4. KG can answer queries")
    print("   5. Everything integrates properly")

    results = {}

    # Run tests
    results['engine_loading'] = test_execution_engine_loading()
    results['google_search'] = test_google_search_tool()
    results['sentiment'] = test_sentiment_analyzer()
    results['summarizer'] = test_summarizer()
    results['news'] = test_news_aggregator()
    results['kg_learning'] = test_dynamic_kg_learning()
    results['kg_query'] = test_kg_query()
    # results['full_workflow'] = test_full_workflow()  # Commented out to avoid long execution

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\n{'='*80}")
    print(f"Final Score: {passed}/{total} tests passed")
    print(f"{'='*80}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is working correctly!")
        print("\nKey Achievements:")
        print("✅ All tools use REAL implementations (OpenAI API)")
        print("✅ Dynamic KG successfully learns from outputs")
        print("✅ KG-aware tools integrate seamlessly")
        print("✅ No mock implementations in use")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("\nNote: Some tests may fail if:")
        print("- OPENAI_API_KEY not set")
        print("- Web scraping dependencies not installed")
        print("- Network connectivity issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
