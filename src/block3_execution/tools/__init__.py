from .create_knowledge_graph import create_knowledge_graph, knowledge_graph_tool
from .sentiment_analyzer import create_sentiment_analyzer, sentiment_analyzer_tool
from .summarizer import create_summarizer, summarizer_tool
from .news_aggregator import create_news_aggregator, news_aggregator_tool
from .google_search import create_google_search, google_search_tool

# Dynamic, self-evolving knowledge graph
from .dynamic_knowledge_graph import (
    create_dynamic_knowledge_graph,
    dynamic_kg_tool,
    DynamicKnowledgeGraph
)

# KG Integration - makes all tools feed into KG
from .kg_integration import (
    get_kg_integration,
    kg_aware_google_search,
    kg_aware_sentiment_analyzer,
    kg_aware_news_aggregator,
    kg_aware_summarizer
)