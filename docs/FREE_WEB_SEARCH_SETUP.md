# Free Web Search Tool - Setup Guide

## 🎉 No API Keys Required!

This search tool is **completely FREE** and perfect for research purposes. It uses DuckDuckGo and web scraping instead of paid APIs.

## Features

✅ **Free DuckDuckGo Search** - No API keys, no costs
✅ **Keyword Extraction** - Automatically extracts search keywords from queries and plan steps
✅ **Web Scraping** - Scrapes actual content from top results (like humans browsing)
✅ **Parallel Search** - Supports multiple searches simultaneously
✅ **Smart Caching** - Avoids duplicate requests
✅ **LLM Summarization** - Optional GPT-4o-mini summarization of results

## Quick Start

### 1. Install Dependencies

```bash
# Install search dependencies
pip install -r requirements_search.txt
```

Or install manually:

```bash
pip install duckduckgo-search beautifulsoup4 requests trafilatura fake-useragent lxml html5lib
```

### 2. Test the Tool

```bash
# Run the test harness
python src/block3_execution/tools/google_search.py
```

### 3. Use in Your Code

```python
from src.block3_execution.tools import google_search_tool

# Simple search
result = google_search_tool("Pathaan box office collection")
print(result)

# With context (from plan step)
context = {
    'step': 'Step 1: Search for latest Bollywood news - Tool: google_search'
}
result = google_search_tool("Bollywood news", context=context)
print(result)
```

## How It Works

### 1. Keyword Extraction

The tool automatically extracts meaningful keywords from your queries:

```python
Input:  "Tell me about the latest Bollywood news for Shah Rukh Khan"
Output: ['Shah', 'Rukh', 'Khan', 'bollywood', 'news']
```

From plan steps:
```python
Input:  "Step 1: Search for Pathaan reviews - Tool: google_search - Expected: Reviews"
Output: ['pathaan', 'reviews']
```

### 2. Free DuckDuckGo Search

Uses DuckDuckGo's free search (no API keys):
```python
search_query = "Shah Rukh Khan Pathaan"
results = tool._search_duckduckgo(search_query, num_results=5)
# Returns: List of search results with title, URL, snippet
```

### 3. Web Scraping

Automatically scrapes content from top results:
```python
# Scrapes top 3 results in parallel
results = tool._scrape_results(results, max_sites=3)

# Each result now has 'scraped' field with:
# - title: Page title
# - content: Extracted main content (up to 5000 chars)
# - success: Whether scraping succeeded
```

### 4. Parallel Search

Search multiple queries simultaneously:
```python
queries = [
    "Jawan box office",
    "Dunki release date",
    "Tiger 3 reviews"
]
results = tool.parallel_search(queries)
# Returns: Dict mapping each query to its results
```

### 5. Smart Caching

Automatically caches results for 1 hour:
```python
# First search - fetches from web
result1 = tool.search("Pathaan reviews")

# Second search within 1 hour - returns cached
result2 = tool.search("Pathaan reviews")  # Instant!
```

## Usage Examples

### Example 1: Basic Search

```python
from src.block3_execution.tools import create_google_search

tool = create_google_search()

# Simple search
result = tool.search(
    query="Pathaan box office collection",
    num_results=5,
    scrape_content=True,
    summarize=True
)
print(result)
```

**Output:**
```
🔍 Web Search Answer:

Pathaan has achieved remarkable box office success, becoming one of the highest-
grossing Hindi films. The movie collected approximately ₹57 crores on its opening
day and crossed ₹1000 crores worldwide within 10 days. According to multiple
sources, the total domestic collection is around ₹525 crores...

📚 Sources:
1. Pathaan Box Office Collection - bollywoodhungama.com
2. Shah Rukh Khan's Pathaan Crosses 1000 Cr - pinkvilla.com
3. Pathaan Worldwide Collections - boxofficeindia.com
```

### Example 2: Keyword Extraction from Plan

```python
# Plan step from execution engine
plan_step = {
    'description': 'Step 1: Search for latest Shah Rukh Khan movie news - Tool: google_search',
    'tool': 'google_search'
}

# Tool automatically extracts keywords
result = tool.search(
    query="Shah Rukh Khan movies",
    context={'step': plan_step['description']}
)
```

**What happens:**
1. Extracts keywords: `['Shah', 'Rukh', 'Khan', 'movie', 'news']`
2. Searches DuckDuckGo for: "Shah Rukh Khan movie news"
3. Scrapes top 3 websites
4. Returns summarized results

### Example 3: Parallel Searches

```python
# When plan has multiple search steps
queries = [
    "Pathaan box office",
    "Jawan release date",
    "Dunki trailer"
]

# Execute all in parallel
results = tool.parallel_search(queries)

for query, result in results.items():
    print(f"\n=== {query} ===")
    print(result)
```

### Example 4: Without LLM Summarization

If you don't want to use OpenAI for summarization:

```python
result = tool.search(
    query="Bollywood news",
    summarize=False  # Return raw results
)
```

**Output:**
```
🔍 Web Search Results for: bollywood news

1. Latest Bollywood News and Updates
   Source: bollywoodhungama.com
   URL: https://www.bollywoodhungama.com/news/
   Get the latest Bollywood news, celeb gossip, movie reviews...
   📄 Content Preview: The latest from Bollywood includes...

2. Bollywood News Today
   Source: pinkvilla.com
   ...
```

## Advanced Features

### Custom Keyword Extraction

```python
extractor = KeywordExtractor()

# Extract from query
keywords = extractor.extract_keywords("What are the latest movies starring Alia Bhatt?")
# Returns: ['Alia', 'Bhatt', 'movies']

# Extract from plan step
keywords = extractor.extract_from_plan_step(
    "Step 2: Search for reviews - Tool: google_search"
)
# Returns: ['reviews']
```

### Web Scraping Individual URLs

```python
scraper = WebScraper()

content = scraper.scrape_url("https://example.com/article")
print(content['title'])
print(content['content'][:500])  # First 500 chars
```

### Cache Management

```python
cache = SearchCache()

# Get cached result
cached = cache.get("Pathaan reviews")

# Manually cache
cache.set("custom query", {'results': [...]})

# Cache location: .cache/search/
```

## Configuration

### Adjust Number of Results

```python
result = tool.search(
    query="Bollywood news",
    num_results=10  # Get 10 results instead of 5
)
```

### Control Scraping

```python
# Don't scrape website content (faster, less data)
result = tool.search(
    query="Bollywood news",
    scrape_content=False
)

# Scrape more sites
result = tool.search(
    query="Bollywood news",
    scrape_content=True,
    num_results=10  # Will scrape top 3 from 10 results
)
```

### Cache Settings

```python
# Custom cache directory and max age
cache = SearchCache(cache_dir="custom_cache")
cached = cache.get("query", max_age=7200)  # 2 hours
```

## Integration with Agent System

The tool integrates seamlessly with the execution engine:

```python
# In execution_engine.py, the tool is called as:
result = google_search_tool(
    query=step['query'],
    context={'step': step},
    num_results=5
)
```

The tool will:
1. Extract keywords from the step description
2. Search DuckDuckGo
3. Scrape top results
4. Summarize with GPT-4o-mini (if available)
5. Return formatted results

## Troubleshooting

### Error: "Web scraping libraries not installed"

```bash
pip install -r requirements_search.txt
```

### Error: "DuckDuckGo search failed"

- Check internet connection
- DuckDuckGo may have rate limits, try again in a moment
- Use cache to avoid repeated requests

### Error: "Scraping failed for URL"

- Some websites block scrapers (normal)
- Tool will still return search results
- Try with `scrape_content=False` for faster results

### Slow Performance

```python
# Reduce number of results and scraping
result = tool.search(
    query="...",
    num_results=3,      # Fewer results
    scrape_content=False  # Skip scraping
)
```

## Performance Tips

1. **Use caching**: Results are cached for 1 hour automatically
2. **Parallel searches**: Use `parallel_search()` for multiple queries
3. **Limit results**: Use `num_results=3` for faster searches
4. **Skip scraping**: Set `scrape_content=False` if you only need snippets
5. **Summarization**: Enable `summarize=True` for cleaner output

## Comparison: Free vs Paid APIs

| Feature | Free (DuckDuckGo) | Google Custom Search API |
|---------|-------------------|-------------------------|
| Cost | 🟢 FREE | 🔴 $5 per 1000 queries |
| API Key | 🟢 Not required | 🔴 Required |
| Rate Limit | 🟡 Moderate | 🟢 100/day free, then paid |
| Results Quality | 🟢 Good | 🟢 Excellent |
| Web Scraping | 🟢 Included | 🔴 Not included |
| Setup | 🟢 pip install | 🔴 Cloud setup needed |

**For research purposes, the free version is perfect!** ✅

## Dependencies

```txt
duckduckgo-search>=4.0.0    # Free DuckDuckGo search
beautifulsoup4>=4.12.0       # HTML parsing
requests>=2.31.0             # HTTP requests
lxml>=4.9.0                  # Fast HTML parser
html5lib>=1.1                # HTML parser
trafilatura>=1.6.0           # Content extraction
fake-useragent>=1.4.0        # User agent rotation
```

## License & Legal

- DuckDuckGo search: Free for research and personal use
- Web scraping: Respect robots.txt and site terms of service
- For commercial use, check individual site policies
- This tool is intended for research and educational purposes

## Support

If you encounter issues:

1. Check installation: `pip list | grep -E "(duckduckgo|beautifulsoup|trafilatura)"`
2. Test the tool: `python src/block3_execution/tools/google_search.py`
3. Check logs in `.cache/search/`
4. Open an issue on GitHub

---

**Happy searching! 🔍🎉**
