# Google Search Tool

## Overview

The Google Search Tool provides web search capabilities to the preference-guided agent system. It integrates with Google Custom Search JSON API to retrieve current information, verify facts, and supplement the knowledge base with up-to-date web content.

## Features

- **Web Search**: Perform Google searches and retrieve formatted results
- **News Search**: Search specifically for news articles
- **Date Range Filtering**: Search with custom date ranges
- **LLM Summarization**: Automatically summarize search results using GPT-4o-mini
- **Mock Mode**: Graceful fallback when API credentials are not configured

## Setup

### Prerequisites

1. **Google Cloud API Key**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Custom Search API
   - Create credentials (API key)

2. **Custom Search Engine ID**
   - Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Create a new search engine
   - Configure search settings (search the entire web or specific sites)
   - Copy the Search Engine ID (CX)

3. **OpenAI API Key** (for result summarization)
   - Already configured in the system

### Environment Variables

Set the following environment variables:

```bash
export GOOGLE_API_KEY='your-google-api-key'
export GOOGLE_SEARCH_ENGINE_ID='your-search-engine-cx-id'
export OPENAI_API_KEY='your-openai-api-key'  # Already set
```

Or add to your `.env` file:

```env
GOOGLE_API_KEY=your-google-api-key
GOOGLE_SEARCH_ENGINE_ID=your-search-engine-cx-id
OPENAI_API_KEY=your-openai-api-key
```

## Usage

### Basic Search

The tool is automatically available to the planning and execution system. When the planner determines that web search is needed, it will include `google_search` in the plan.

**Example Query:**
```
"What are the latest box office collections for Pathaan?"
```

**Plan Step:**
```
Step 1: Search for current box office information - Tool: google_search - Expected: Current box office numbers
```

### Direct API Usage

You can also use the tool directly in code:

```python
from src.block3_execution.tools.google_search import create_google_search

# Create tool instance
search_tool = create_google_search()

# Basic search
results = search_tool.search(
    query="Bollywood latest releases 2024",
    num_results=5,
    summarize=True
)

# News search
news = search_tool.search_news(
    query="Shah Rukh Khan new movie"
)

# Date range search
dated_results = search_tool.search_with_date_range(
    query="Bollywood awards",
    start_date="20240101",
    end_date="20241231"
)
```

## API Reference

### GoogleSearchTool

**Methods:**

#### `search(query, context=None, num_results=5, summarize=True)`

Perform a Google search and return formatted results.

**Parameters:**
- `query` (str): Search query string
- `context` (dict, optional): Context dictionary with additional parameters
- `num_results` (int): Number of results to retrieve (1-10, default: 5)
- `summarize` (bool): Whether to use LLM to summarize results (default: True)

**Returns:**
- Formatted search results as a string

#### `search_news(query, context=None)`

Search specifically for news articles.

**Parameters:**
- `query` (str): Search query
- `context` (dict, optional): Optional context

**Returns:**
- Formatted news search results

#### `search_with_date_range(query, start_date=None, end_date=None, context=None)`

Search with date range filtering.

**Parameters:**
- `query` (str): Search query
- `start_date` (str, optional): Start date in format YYYYMMDD
- `end_date` (str, optional): End date in format YYYYMMDD
- `context` (dict, optional): Optional context

**Returns:**
- Formatted search results

## Integration Points

The Google Search tool is integrated at the following locations:

1. **Tool Definition**: `/src/block3_execution/tools/google_search.py`
2. **Tool Registration**: `/src/block3_execution/tools/__init__.py`
3. **Execution Engine**: `/src/block3_execution/execution_engine.py` (lines 61, 68, 238-240)
4. **Planning Constraints**: `/src/block2_planning/policy_llm.py` (line 190)
5. **Configuration**: `/configs/default_config.yaml` (line 50)

## Example Outputs

### Without Summarization

```
Google Search Results:

1. Pathaan Box Office Collection Day 1: Shah Rukh Khan Film...
   URL: https://example.com/pathaan-box-office
   The film collected 57 crores on its opening day, setting a new record...

2. Pathaan worldwide box office collection: SRK film crosses...
   URL: https://example.com/pathaan-worldwide
   Pathaan has grossed over 1000 crores worldwide in just 10 days...

...
```

### With Summarization

```
Google Search Summary:

Pathaan, starring Shah Rukh Khan, has achieved remarkable box office success.
The film collected approximately 57 crores on its opening day, setting a new
record for Bollywood releases. Within 10 days of its release, the film crossed
the 1000 crore mark worldwide. According to recent reports from Bollywood
Hungama and Pinkvilla, the total domestic collection stands at around 525 crores,
while international collections have contributed significantly to the overall
gross. The film's success marks a major comeback for Shah Rukh Khan after a
brief hiatus.

Sources: Bollywood Hungama, Pinkvilla, Times of India
```

## Error Handling

The tool includes robust error handling:

- **Missing API Credentials**: Returns mock results with setup instructions
- **API Request Failures**: Catches exceptions and returns error messages
- **LLM Summarization Failures**: Falls back to basic formatting
- **Network Timeouts**: 10-second timeout for API requests

## Mock Mode

When API credentials are not configured, the tool operates in mock mode:

```
[MOCK] Google Search Results for: latest bollywood news

Note: Google Search API credentials not configured.
Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables.

To enable Google Search:
1. Get a Google Cloud API key from https://console.cloud.google.com/
2. Enable Custom Search API
3. Create a Custom Search Engine at https://programmablesearchengine.google.com/
4. Set environment variables:
   export GOOGLE_API_KEY='your-api-key'
   export GOOGLE_SEARCH_ENGINE_ID='your-cx-id'
```

## Rate Limits & Costs

### Google Custom Search API

- **Free Tier**: 100 queries per day
- **Paid Tier**: $5 per 1000 queries (up to 10,000 queries per day)
- **Quota**: Can be increased by contacting Google Cloud support

### OpenAI API (for summarization)

- **Model**: GPT-4o-mini
- **Cost**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **Max Tokens**: 500 per summary

## Best Practices

1. **Use Summarization**: Enable summarization for better integration with the agent system
2. **Limit Results**: Keep `num_results` between 3-7 for optimal performance
3. **Specific Queries**: Craft specific queries to reduce irrelevant results
4. **Date Filtering**: Use date ranges when searching for recent information
5. **Cache Results**: Consider caching frequently searched queries to save API calls

## Troubleshooting

### Issue: "403 Forbidden" error

**Solution:** Verify that:
- Custom Search API is enabled in Google Cloud Console
- API key has proper permissions
- Search Engine ID (CX) is correct

### Issue: No results returned

**Solution:** Check if:
- Search query is too specific
- Date range is too narrow
- Search engine is configured to search the entire web

### Issue: Summarization not working

**Solution:** Ensure:
- OPENAI_API_KEY is set
- OpenAI account has sufficient credits
- Network connection is stable

## Future Enhancements

Potential improvements for the Google Search tool:

1. **Caching Layer**: Implement Redis/SQLite cache for frequently searched queries
2. **Multi-Source Search**: Integrate additional search APIs (Bing, DuckDuckGo)
3. **Image Search**: Add support for Google Image Search
4. **Advanced Filters**: Support for language, region, and safe search filters
5. **Search Analytics**: Track search patterns and popular queries
6. **Custom Scoring**: Implement relevance scoring for search results
7. **Batch Searches**: Support for multiple queries in a single request

## Related Documentation

- [Google Custom Search API Documentation](https://developers.google.com/custom-search/v1/overview)
- [Tool Development Guide](../README.md)
- [Execution Engine Documentation](../src/block3_execution/README.md)
