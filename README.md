# Preference-Guided Agent System

A sophisticated multi-agent system for Bollywood entertainment queries with preference learning, Tree of Thoughts planning, and reinforcement learning optimization.

## 🎯 Overview

This system combines:
- **Preference Learning**: DPO and PPO training for personalized responses
- **Tree of Thoughts**: Advanced reasoning and planning
- **Multi-Tool Execution**: Knowledge graphs, sentiment analysis, summarization, news aggregation, and **FREE web search**
- **Reinforcement Learning**: RLAIF-based reward optimization

Key Features:
- ✅ **FREE Web Search** - No API keys required (DuckDuckGo + web scraping)
- ✅ **LLM-Based Tools** - Most tools use OpenAI's GPT models
- ✅ **Research-Friendly** - Perfect for academic and research purposes
- ✅ **Flexible** - Easy to extend to other domains

## Available Tools

### 1. Knowledge Graph Tool 📊
**File**: `knowledge_graph.py`

Simulates a knowledge graph using LLM queries for Bollywood entities and relationships.

**Features**:
- Actor filmographies
- Movie details (cast, crew, ratings)
- Director information
- Collaborations between people
- Industry relationships

**Usage**:
```python
from src.block3_execution.tools import create_knowledge_graph

kg = create_knowledge_graph()

# Get actor's movies
result = kg.get_actor_movies("Shah Rukh Khan")

# Get movie details
result = kg.get_movie_details("Dilwale Dulhania Le Jayenge")

# Get collaborations
result = kg.get_collaborations("Shah Rukh Khan", "Kajol")

# General query
result = kg.query("Who directed Sholay?")
```

**Example Queries**:
- "List all movies of Aamir Khan"
- "Tell me about the cast of 3 Idiots"
- "Which movies did Karan Johar direct?"
- "Find collaborations between Deepika and Ranveer"

---

### 2. Sentiment Analyzer Tool 💭
**File**: `sentiment_analyzer.py`

Analyzes sentiment of reviews, comments, and social media posts.

**Features**:
- Single text analysis
- Multiple review aggregation
- Comparative sentiment analysis
- Detailed emotional themes
- Aspect-based sentiment (acting, direction, music, etc.)

**Usage**:
```python
from src.block3_execution.tools import create_sentiment_analyzer

analyzer = create_sentiment_analyzer()

# Analyze single review
review = "Pathaan is amazing! Best action film ever."
result = analyzer.analyze(review)

# Analyze multiple reviews
reviews = ["Great movie!", "Loved it!", "Could be better"]
result = analyzer.analyze_reviews(reviews, movie_name="Jawan")

# Compare sentiments
result = analyzer.compare_sentiment(review1, review2)
```

**Output Includes**:
- Overall sentiment (Positive/Negative/Neutral/Mixed)
- Sentiment score (0-10)
- Key emotional themes
- Specific praised/criticized aspects
- Confidence level

---

### 3. Summarizer Tool 📝
**File**: `summarizer.py`

Summarizes text with support for Hindi-English mixed content.

**Features**:
- Three length options (brief/medium/detailed)
- Movie plot summaries
- Article/news summarization
- Key points extraction
- Hinglish support

**Usage**:
```python
from src.block3_execution.tools import create_summarizer

summarizer = create_summarizer()

# Summarize text
long_text = "..."
result = summarizer.summarize(long_text, length="brief")

# Get movie plot
result = summarizer.summarize_plot("3 Idiots")

# Summarize article
result = summarizer.summarize_article(article_text, focus="box office")

# Extract key points
result = summarizer.extract_key_points(text, num_points=5)
```

**Length Options**:
- `brief`: 2-3 sentences
- `medium`: 1 paragraph (4-6 sentences)
- `detailed`: 2-3 paragraphs

---

### 4. News Aggregator Tool 📰
**File**: `news_aggregator.py`

Aggregates latest Bollywood news, gossip, and industry updates.

**Features**:
- Latest news by category
- Trending topics
- Upcoming releases
- Box office updates
- Celebrity news

**Usage**:
```python
from src.block3_execution.tools import create_news_aggregator

aggregator = create_news_aggregator()

# Get latest news
result = aggregator.get_latest_news()

# Trending topics
result = aggregator.get_trending_topics()

# Box office updates
result = aggregator.get_box_office_updates()

# Celebrity news
result = aggregator.get_celebrity_news("Shah Rukh Khan")

# Upcoming releases
result = aggregator.get_upcoming_releases("next quarter")
```

**News Categories**:
- `all`: General news
- `box_office`: Collections and performance
- `releases`: New movie releases
- `celebrity`: Celebrity updates
- `awards`: Award news and nominations

---

### 5. FREE Web Search Tool 🔍✨
**File**: `google_search.py`

**🎉 No API Keys Required! Completely FREE!**

Web search using DuckDuckGo + real web scraping (like a human browsing).

**Features**:
- ✅ Free DuckDuckGo search (no API keys!)
- ✅ Automatic keyword extraction from queries and plan steps
- ✅ Web scraping of top results (actual content, not just snippets)
- ✅ Parallel search support (multiple queries at once)
- ✅ Smart caching (1-hour cache to avoid duplicate requests)
- ✅ LLM-powered summarization (optional)

**Quick Setup**:
```bash
# Install free dependencies
pip install -r requirements_search.txt

# Test it
python scripts/test_search.py
```

**Usage**:
```python
from src.block3_execution.tools import create_google_search

search = create_google_search()

# Basic search - automatically extracts keywords and scrapes content
result = search.search("Pathaan box office collections")

# With plan context - extracts keywords from step
context = {'step': 'Step 1: Search for latest Bollywood news'}
result = search.search("Bollywood news", context=context)

# Parallel search for multiple queries
results = search.parallel_search([
    "Jawan box office",
    "Dunki release date",
    "Tiger 3 reviews"
])
```

**How It Works Like a Human:**
1. Extracts meaningful keywords from your query
2. Searches DuckDuckGo (free!)
3. Scrapes top 3 websites for actual content
4. Summarizes with GPT-4o-mini
5. Caches results to avoid duplicate work

**Perfect for Research!** ✅ Free ✅ No API Keys ✅ Scrapes Real Content

See [Free Web Search Setup Guide](docs/FREE_WEB_SEARCH_SETUP.md) for complete documentation.

---

## How Tools Work

### Architecture

```
User Query
    ↓
Execution Engine
    ↓
Tool Selection (based on plan)
    ↓
Tool Function Call
    ↓
LLM Processing (GPT-4o-mini)
    ↓
Structured Result
    ↓
Back to Execution Engine
```

### LLM-Based Design

All tools use OpenAI's API internally:
1. **System Prompt**: Defines tool's role and output format
2. **User Prompt**: Contains the actual query with context
3. **Temperature**: Set for consistency (0.3-0.7)
4. **Max Tokens**: Limited for efficiency (300-700)

**Benefits**:
- No external data sources needed
- Works out of the box
- Easy to customize prompts
- Consistent quality

**Limitations**:
- Knowledge cutoff (January 2025)
- API costs (minimal with GPT-4o-mini)
- Requires internet connection

---

## Testing Tools

Each tool file has a `__main__` block for testing:

```bash
# Test knowledge graph
python src/block3_execution/tools/knowledge_graph.py

# Test sentiment analyzer
python src/block3_execution/tools/sentiment_analyzer.py

# Test summarizer
python src/block3_execution/tools/summarizer.py

# Test news aggregator
python src/block3_execution/tools/news_aggregator.py
```

---

## Adding Custom Tools

### Step 1: Create Tool File

```python
# src/block3_execution/tools/my_tool.py

import os
from typing import Dict, Any
from openai import OpenAI

class MyCustomTool:
    """Description of your tool."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def process(self, query: str, context: Dict = None) -> str:
        """Main processing method."""
        system_prompt = """Your tool's system prompt..."""
        user_prompt = f"Query: {query}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

# Simple interface for execution engine
def my_tool_function(query: str, context: Dict = None, **kwargs) -> str:
    tool = MyCustomTool()
    return tool.process(query, context)
```

### Step 2: Register in __init__.py

```python
# src/block3_execution/tools/__init__.py

from .my_tool import my_tool_function

__all__ = [
    'my_tool_function',
    # ... other tools
]
```

### Step 3: Update Execution Engine

```python
# src/block3_execution/execution_engine.py

def _initialize_tools(self):
    from src.block3_execution.tools import my_tool_function
    
    return {
        'my_tool': my_tool_function,
        # ... other tools
    }
```

### Step 4: Use in Plans

Now the planner can use `my_tool` in execution plans!

---

## Tool Prompt Engineering Tips

### For Better Results:

1. **Clear System Prompts**: Define role and output format precisely
2. **Structured Output**: Ask for specific formats (numbered lists, JSON, etc.)
3. **Context Inclusion**: Pass relevant context in prompts
4. **Temperature Tuning**:
   - 0.3: Factual, consistent (knowledge queries)
   - 0.5: Balanced (summaries, analysis)
   - 0.7: Creative (news, trending topics)
5. **Token Limits**: Optimize for cost/quality tradeoff

### Example Improvement:

**Before** (vague):
```python
system_prompt = "You are a helpful assistant."
user_prompt = query
```

**After** (specific):
```python
system_prompt = """You are a Bollywood expert assistant.
Provide accurate information about movies, actors, and industry.
Format output as:
1. Direct answer
2. Supporting details
3. Relevant context"""

user_prompt = f"""Query: {query}
Context: {context}

Provide a comprehensive answer."""
```

---
## Extending to Other Domains

These tools are designed for Bollywood but can be adapted:

### 1. Change Domain Knowledge
Update system prompts with new domain expertise:
```python
system_prompt = """You are a [DOMAIN] expert assistant.
You have knowledge about [DOMAIN SPECIFICS]..."""
```

### 2. Adjust Tool Functions
Modify methods for domain-specific queries:
```python
def get_tech_news(self, topic: str):
    """Get tech industry news."""
    # Similar structure, different domain
```

### 3. Update Tool Names
Rename for clarity:
- `BollywoodKnowledgeGraph` → `TechKnowledgeGraph`
- `knowledge_graph_tool` → `tech_knowledge_tool`

---

## 📊 CORAL Integration & Preference Learning

This system supports preference learning using the [CORAL dataset](https://huggingface.co/datasets/kookeej/CORAL) for conversational recommendations.

### Download & Setup

```bash
# Install dependencies
pip install datasets transformers

# Download CORAL datasets
python scripts/download_coral.py --inspect

# Preprocess for training
python scripts/preprocess_data.py
```

### Training with DPO and PPO

```bash
# Train with Direct Preference Optimization
python scripts/train_preference_model.py --stage dpo

# Train with Proximal Policy Optimization
python scripts/train_preference_model.py --stage ppo

# Full training pipeline
python scripts/train_preference_model.py --stage both
```

### Architecture Enhancements

The CORAL integration adds:
- **Block 0**: Preference encoding and retrieval
- **Enhanced Block 1**: Preference-aware context fusion
- **Enhanced Block 2**: DPO-trained planning policy
- **Enhanced Block 4**: PPO-trained reward model
- **Block 5**: Personalized response generation

See [CORAL Integration Plan](docs/CORAL_INTEGRATION_PLAN.md) for comprehensive details.

### Expected Metrics

| Metric | Target |
|--------|--------|
| Hit Rate @10 | > 0.75 |
| NDCG @10 | > 0.65 |
| Preference Accuracy | > 0.70 |
| Contrasting Score | > 0.85 |

---

## Troubleshooting

**1. "OpenAI API key not set"**
```bash
export OPENAI_API_KEY="sk-..."
```

**2. "Google Search API error"**
- Verify API key is correct
- Enable Custom Search API in Google Cloud
- Check search engine CX ID

**3. "CORAL dataset not found"**
```bash
python scripts/download_coral.py
```

---

## Future Enhancements

### Planned Features:
1. **Caching Layer**: Store frequent query results
2. **Real APIs**: Integrate actual data sources (IMDB, etc.)
3. **Multimodal**: Add image/video processing
4. **Memory**: Maintain conversation context across calls
5. **Streaming**: Stream responses for large outputs

### Contributing:
To add new tools:
1. Follow the template structure
2. Include proper error handling
3. Add comprehensive docstrings
4. Write test cases in `__main__`
5. Update this README

---
