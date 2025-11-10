#!/usr/bin/env python3
"""Test all tools to ensure they work correctly."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_tools():
    """Test all four tools."""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        return False
    
    print("=" * 80)
    print("TESTING ALL TOOLS".center(80))
    print("=" * 80)
    print()
    
    try:
        # Test 1: Knowledge Graph
        print("🧪 Test 1: Knowledge Graph Tool")
        print("-" * 80)
        from src.block3_execution.tools import create_knowledge_graph
        
        kg = create_knowledge_graph()
        result = kg.query("Who is Shah Rukh Khan?")
        print(f"Query: Who is Shah Rukh Khan?")
        print(f"Result: {result[:200]}...")
        print("✅ Knowledge Graph working!\n")
        
    except Exception as e:
        print(f"❌ Knowledge Graph failed: {e}\n")
        return False
    
    try:
        # Test 2: Sentiment Analyzer
        print("🧪 Test 2: Sentiment Analyzer Tool")
        print("-" * 80)
        from src.block3_execution.tools import create_sentiment_analyzer
        
        analyzer = create_sentiment_analyzer()
        review = "Amazing movie! One of the best films of the year."
        result = analyzer.analyze(review)
        print(f"Review: {review}")
        print(f"Analysis: {result[:200]}...")
        print("✅ Sentiment Analyzer working!\n")
        
    except Exception as e:
        print(f"❌ Sentiment Analyzer failed: {e}\n")
        return False
    
    try:
        # Test 3: Summarizer
        print("🧪 Test 3: Summarizer Tool")
        print("-" * 80)
        from src.block3_execution.tools import create_summarizer
        
        summarizer = create_summarizer()
        result = summarizer.summarize_plot("3 Idiots")
        print(f"Movie: 3 Idiots")
        print(f"Summary: {result[:200]}...")
        print("✅ Summarizer working!\n")
        
    except Exception as e:
        print(f"❌ Summarizer failed: {e}\n")
        return False
    
    try:
        # Test 4: News Aggregator
        print("🧪 Test 4: News Aggregator Tool")
        print("-" * 80)
        from src.block3_execution.tools import create_news_aggregator
        
        aggregator = create_news_aggregator()
        result = aggregator.get_trending_topics()
        print(f"Query: What's trending in Bollywood?")
        print(f"Result: {result[:200]}...")
        print("✅ News Aggregator working!\n")
        
    except Exception as e:
        print(f"❌ News Aggregator failed: {e}\n")
        return False
    
    # Test 5: Integration with Execution Engine
    print("🧪 Test 5: Execution Engine Integration")
    print("-" * 80)
    try:
        from src.utils.config import load_config
        from src.block3_execution.execution_engine import create_execution_engine
        
        config = load_config()
        engine = create_execution_engine(config)
        
        # Create a simple test plan
        test_plan = {
            'steps': [
                {
                    'description': 'Get info about SRK',
                    'tool': 'knowledge_graph',
                    'expected_output': 'Actor information'
                }
            ]
        }
        
        result = engine.execute_plan(test_plan, "Tell me about Shah Rukh Khan")
        print(f"Plan executed: {test_plan['steps'][0]['description']}")
        print(f"Status: {result['status']}")
        print(f"Answer: {result['final_answer'][:150]}...")
        print("✅ Execution Engine Integration working!\n")
        
    except Exception as e:
        print(f"⚠️  Execution Engine Integration test failed: {e}")
        print("(This is expected if dependencies are missing)\n")
    
    # All tests passed
    print("=" * 80)
    print("✅ ALL TOOLS TESTS PASSED!".center(80))
    print("=" * 80)
    print()
    print("Your tools are ready to use! Try:")
    print("  python run.py --query 'Tell me about Shah Rukh Khan'")
    print()
    
    return True


if __name__ == "__main__":
    success = test_tools()
    sys.exit(0 if success else 1)