# Tools Documentation

## Overview

This directory contains **LLM-based tools** for the Bollywood domain. All tools use OpenAI's GPT models, making them:
-  Easy to test (no external APIs needed)
-  Flexible and adaptable
-  Domain-agnostic (easy to extend to other domains)

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

**1. "OpenAI API key not set"**
```bash
export OPENAI_API_KEY="sk-..."
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
