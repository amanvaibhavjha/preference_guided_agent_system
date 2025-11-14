

# Dynamic Self-Evolving Knowledge Graph

## 🎯 Overview

The Dynamic Knowledge Graph is a **self-evolving memory system** that learns from every interaction and tool output. Unlike traditional static knowledge bases, this KG:

- ✅ **Starts empty** and grows organically
- ✅ **Learns automatically** from all tool outputs
- ✅ **Avoids redundancy** - updates existing entities instead of duplicating
- ✅ **Persists to disk** for long-term memory
- ✅ **Answers queries** using accumulated knowledge
- ✅ **Reduces API calls** by using stored information

Perfect for building agent memory over time!

## 🚀 Quick Start

### Basic Usage

```python
from src.block3_execution.tools import create_dynamic_knowledge_graph

# Create KG
kg = create_dynamic_knowledge_graph()

# Learn from text
kg.learn_from_text(
    "Pathaan is a 2023 film starring Shah Rukh Khan",
    source="user_input"
)

# Query the KG
result = kg.query("Tell me about Shah Rukh Khan")
print(result)

# Get stats
stats = kg.get_stats()
print(f"Entities: {stats['total_entities']}")
```

### With Tool Integration

```python
from src.block3_execution.tools import (
    kg_aware_google_search,
    get_kg_integration
)

# Search - automatically learns from results
result = kg_aware_google_search("Pathaan box office")

# Future queries can use accumulated knowledge
kg_int = get_kg_integration()
answer = kg_int.query_kg_first("What do you know about Pathaan?")
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                         │
│              (Query, Search, Sentiment, News)               │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   KG Integration Layer          │
        │   - Wraps all tool calls        │
        │   - Extracts learning data      │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   Entity Extractor (LLM)        │
        │   - Identifies entities         │
        │   - Extracts relationships      │
        │   - Determines entity types     │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   Dynamic Knowledge Graph       │
        │   - NetworkX graph structure    │
        │   - Entity & relation storage   │
        │   - Confidence scoring          │
        │   - Redundancy detection        │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   Persistence Layer             │
        │   - JSON storage                │
        │   - data/knowledge_graph/       │
        │   - entities.json               │
        │   - relations.json              │
        └─────────────────────────────────┘
```

### Data Model

#### Entity Structure
```python
{
    "id": "shah_rukh_khan",
    "name": "Shah Rukh Khan",
    "type": "Person",
    "properties": {
        "role": "Actor",
        "nationality": "Indian",
        "awards": ["Filmfare", "Padma Shri"]
    },
    "sources": ["web_search", "news_aggregator"],
    "confidence": 0.95,
    "created_at": 1234567890.0,
    "updated_at": 1234567900.0
}
```

#### Relation Structure
```python
{
    "source": "shah_rukh_khan",
    "target": "pathaan",
    "relation_type": "acted_in",
    "properties": {
        "role": "Lead",
        "year": "2023"
    },
    "sources": ["web_search"],
    "confidence": 0.9,
    "created_at": 1234567890.0,
    "updated_at": 1234567890.0
}
```

## 📚 Key Features

### 1. Automatic Learning

The KG learns from:
- **User queries** - entities and intents
- **Web search results** - facts, numbers, relationships
- **Sentiment analysis** - opinions, ratings
- **News articles** - current events, updates
- **Any text input** - flexible learning

```python
# Learns from web search
kg.learn_from_text(
    "Pathaan collected ₹1055 crores worldwide",
    source="web_search"
)

# Learns from news
kg.learn_from_text(
    "Shah Rukh Khan wins lifetime achievement award",
    source="news"
)

# Learns from sentiment
kg.learn_from_text(
    "Jawan is amazing! Rating: 9/10",
    source="sentiment"
)
```

### 2. Intelligent Entity Extraction

Uses GPT-4o-mini to extract:
- **Entities**: People, movies, songs, events, organizations
- **Properties**: Attributes, metadata, metrics
- **Relationships**: acted_in, directed, produced, won, collaborated_with

```python
Input: "Pathaan, directed by Siddharth Anand, stars Shah Rukh Khan"

Extracted:
- Entity: "Shah Rukh Khan" (Person)
- Entity: "Pathaan" (Movie)
- Entity: "Siddharth Anand" (Person, Director)
- Relation: Shah Rukh Khan --[acted_in]--> Pathaan
- Relation: Siddharth Anand --[directed]--> Pathaan
```

### 3. Redundancy Avoidance

The KG **updates** existing entities instead of creating duplicates:

```python
# First learning
kg.learn_from_text("Pathaan is a 2023 film")
# Creates: Entity "Pathaan" with property year=2023

# Second learning (same entity, new info)
kg.learn_from_text("Pathaan collected ₹1055 crores")
# Updates: Adds property box_office=1055cr to existing "Pathaan"
# Does NOT create duplicate entity!
```

### 4. Confidence Scoring

Each entity and relation has a confidence score:
- **Manual input**: 1.0 (highest)
- **LLM extraction**: 0.8 (high)
- **Rule-based**: 0.5 (moderate)

Confidence is updated as more information arrives:
```python
# First source: confidence = 0.8
# Second source confirms: confidence = (0.8 + 0.9) / 2 = 0.85
```

### 5. Source Tracking

Every piece of information tracks its sources:

```python
Entity "Shah Rukh Khan":
  sources: ["web_search", "news_aggregator", "user_query"]

# You can see where each fact came from!
```

### 6. Smart Querying

Query the KG with natural language:

```python
query = "Tell me about Shah Rukh Khan"
result = kg.query(query)
```

Output:
```
📚 From Knowledge Graph:

**Shah Rukh Khan** (Person):
  - role: Actor
  - nationality: Indian
  - awards: Filmfare, Padma Shri
  Relations:
    - acted_in → Pathaan
    - acted_in → Jawan
    - acted_in → Dunki
    - won → Lifetime Achievement Award
  Sources: web_search, news_aggregator, user_query
```

### 7. Persistence

All data is saved to disk:
```
data/knowledge_graph/
├── entities.json    # All entities
├── relations.json   # All relationships
```

Survives restarts and builds memory over time!

## 🔧 Integration with Tools

### Automatic Integration

All tools can be wrapped to automatically feed into KG:

```python
from src.block3_execution.tools import (
    kg_aware_google_search,
    kg_aware_sentiment_analyzer,
    kg_aware_news_aggregator,
    kg_aware_summarizer
)

# These tools automatically:
# 1. Execute normally
# 2. Learn from inputs and outputs
# 3. Update KG
# 4. Return results

result = kg_aware_google_search("Pathaan reviews")
# KG now knows about Pathaan!
```

### Manual Integration

Wrap any tool:

```python
from src.block3_execution.tools import get_kg_integration

kg_int = get_kg_integration()

# Wrap a custom tool
wrapped_tool = kg_int.wrap_tool(
    tool_func=my_custom_tool,
    tool_name="custom_tool",
    learn_from_input=True,
    learn_from_output=True
)
```

## 📊 Usage Patterns

### Pattern 1: Query KG First, Then Search

```python
from src.block3_execution.tools import get_kg_integration

kg_int = get_kg_integration()

# Try KG first
kg_answer = kg_int.query_kg_first("Shah Rukh Khan movies")

if kg_answer:
    # Found in KG!
    print("Answered from memory:", kg_answer)
else:
    # Not in KG, search the web
    result = google_search("Shah Rukh Khan movies")
    # Learn from search for next time
    kg_int.learn_from_search("Shah Rukh Khan movies", result)
```

### Pattern 2: Continuous Learning

```python
# Over time, the agent accumulates knowledge
interactions = [
    ("Pathaan box office", "web_search"),
    ("Jawan reviews", "web_search"),
    ("SRK awards", "news"),
    ("Dunki cast", "web_search")
]

for query, source in interactions:
    result = execute_tool(query)
    kg.learn_from_text(result, source=source)

# After many interactions, KG becomes knowledgeable
# Can answer WITHOUT external calls!
```

### Pattern 3: Cross-Tool Learning

```python
# Search learns about a movie
search_result = kg_aware_google_search("Pathaan")
# KG learns: Pathaan, SRK, box office, etc.

# Later, sentiment analysis
sentiment_result = kg_aware_sentiment_analyzer("Pathaan review")
# KG adds: sentiment, ratings to existing Pathaan entity

# Later, news
news_result = kg_aware_news_aggregator("Pathaan")
# KG updates: latest numbers, awards, etc.

# All information is connected in the graph!
```

## 🎓 Advanced Features

### Custom Entity Types

Define your own entity types:

```python
kg.add_entity(
    name="Filmfare Awards 2024",
    entity_type="Event",
    properties={
        "date": "2024-01-15",
        "location": "Mumbai"
    }
)
```

### Custom Relationships

Define custom relation types:

```python
kg.add_relation(
    source_name="Shah Rukh Khan",
    target_name="Filmfare Awards 2024",
    relation_type="attended",
    properties={"as": "Guest of Honor"}
)
```

### Graph Traversal

Access the NetworkX graph directly:

```python
import networkx as nx

# Get neighbors
neighbors = list(kg.graph.neighbors("shah_rukh_khan"))

# Shortest path
path = nx.shortest_path(
    kg.graph,
    source="shah_rukh_khan",
    target="siddharth_anand"
)

# Community detection
communities = nx.community.greedy_modularity_communities(
    kg.graph.to_undirected()
)
```

### Statistics and Analytics

```python
stats = kg.get_stats()

print(f"Total Entities: {stats['total_entities']}")
print(f"Total Relations: {stats['total_relations']}")
print(f"Entity Types: {stats['entity_types']}")
print(f"Relation Types: {stats['relation_types']}")
print(f"Graph Density: {stats['graph_density']}")
```

## 🧪 Testing

### Run Demo

See the full evolution:

```bash
python scripts/demo_dynamic_kg.py
```

This shows:
1. Empty KG at start
2. Learning from user queries
3. Learning from web searches
4. Learning from sentiment
5. Learning from news
6. Answering queries from memory
7. Handling redundancy

### Run Unit Tests

```bash
python src/block3_execution/tools/dynamic_knowledge_graph.py
python src/block3_execution/tools/kg_integration.py
```

## 📈 Performance Benefits

### Reduced API Calls

| Scenario | Without KG | With KG |
|----------|-----------|---------|
| First query about SRK | Web search | Web search |
| Second query about SRK | Web search | **KG only** |
| Third query about SRK | Web search | **KG only** |
| API calls for 3 queries | 3 | **1** |

### Faster Responses

- **Web search**: 5-10 seconds
- **KG query**: <0.1 seconds
- **Speedup**: 50-100x!

### Better Context

The KG builds connections:
```
Shah Rukh Khan --[acted_in]--> Pathaan
Pathaan --[directed_by]--> Siddharth Anand
Siddharth Anand --[also_directed]--> War

# Agent can answer: "What other movies did Pathaan's director make?"
# Without searching, just from graph traversal!
```

## 🔍 Comparison with Traditional KG

| Feature | Traditional KG | Dynamic KG |
|---------|---------------|------------|
| **Initial State** | Pre-populated | Empty |
| **Growth** | Manual updates | Automatic learning |
| **Sources** | Curated data | All tool outputs |
| **Redundancy** | Manual deduplication | Automatic merging |
| **Memory** | Static | Evolving |
| **Personalization** | No | Yes (per user) |
| **Setup** | Complex | Simple |

## 🛠️ Configuration

### Storage Location

```python
# Default: data/knowledge_graph/
kg = DynamicKnowledgeGraph()

# Custom location
kg = DynamicKnowledgeGraph(storage_path="custom/path")
```

### Enable/Disable Learning

```python
from src.block3_execution.tools import get_kg_integration

kg_int = get_kg_integration()

# Disable learning temporarily
kg_int.learning_enabled = False

# Re-enable
kg_int.learning_enabled = True
```

### Confidence Thresholds

```python
# Only add high-confidence entities
if confidence > 0.7:
    kg.add_entity(name, entity_type, confidence=confidence)
```

## 🐛 Troubleshooting

### "No information found"

The KG is empty or doesn't have that entity yet. It will learn from the first search.

### Duplicate entities

Entities are deduplicated by name (case-insensitive). "Shah Rukh Khan" and "shah rukh khan" are the same.

### Slow learning

Entity extraction uses GPT-4o-mini. If slow:
1. Check OpenAI API key
2. Reduce text length (first 2000 chars)
3. Batch learning operations

### Large graph files

If JSON files get too large:
1. Archive old data
2. Filter by confidence score
3. Remove low-confidence entities

## 🎯 Best Practices

### 1. Always Learn from Tool Outputs

```python
result = any_tool(query)
kg.learn_from_text(result, source="tool_name")
```

### 2. Use Source Tracking

```python
kg.learn_from_text(text, source="web_search_2024-01")
# Later you can see when/where info came from
```

### 3. Query KG First for Factual Questions

```python
if is_factual_query(query):
    kg_answer = kg.query(query)
    if has_useful_info(kg_answer):
        return kg_answer
# Fall back to web search
```

### 4. Periodic Cleanup

```python
# Remove low-confidence entities
for entity_id, entity in list(kg.entities.items()):
    if entity.confidence < 0.3:
        del kg.entities[entity_id]
kg._save_graph()
```

### 5. Monitor Growth

```python
stats = kg.get_stats()
print(f"KG has {stats['total_entities']} entities")

if stats['total_entities'] > 10000:
    print("Consider archiving or cleanup")
```

## 🚀 Future Enhancements

Potential improvements:
- [ ] Graph database backend (Neo4j, ArangoDB)
- [ ] Entity disambiguation (same name, different entities)
- [ ] Temporal reasoning (facts change over time)
- [ ] Uncertainty reasoning (contradicting sources)
- [ ] Graph visualization
- [ ] Export to RDF/OWL
- [ ] Multi-user graphs (shared knowledge)
- [ ] Incremental learning (online updates)

## 📝 Summary

The Dynamic Knowledge Graph:

✅ **Starts empty** and grows with every interaction
✅ **Learns automatically** from all tool outputs
✅ **Avoids redundancy** with intelligent merging
✅ **Persists to disk** for long-term memory
✅ **Answers queries** using accumulated knowledge
✅ **Reduces costs** by avoiding unnecessary API calls
✅ **Builds connections** across different sources

Perfect for creating **intelligent, memory-enabled agents**!

---

**Get Started:**
```bash
python scripts/demo_dynamic_kg.py
```
