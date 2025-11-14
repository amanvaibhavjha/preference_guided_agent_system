"""
Free Web Search Tool - No API Keys Required!

This tool uses DuckDuckGo (free) and web scraping to search and retrieve
information from the web. Perfect for research purposes.

Features:
- Free DuckDuckGo search (no API keys needed)
- Automatic keyword extraction from queries
- Web scraping of top results (like a human browsing)
- Parallel search support for multiple keywords
- Content extraction from websites
- Smart caching to avoid duplicate requests
"""

import os
import re
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, quote_plus

# Web scraping imports
try:
    from duckduckgo_search import DDGS
    import requests
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
    import trafilatura
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

# OpenAI for summarization (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class KeywordExtractor:
    """Extract search keywords from queries and plan steps."""

    def __init__(self):
        """Initialize keyword extractor."""
        # Common stop words to remove
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'is', 'was', 'are', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what',
            'when', 'where', 'who', 'which', 'why', 'how', 'get', 'find', 'search',
            'tell', 'me', 'show', 'give', 'latest', 'recent', 'current', 'new'
        }

    def extract_keywords(self, query: str, context: Dict = None) -> List[str]:
        """
        Extract meaningful keywords from a query.

        Args:
            query: Search query or plan step
            context: Optional context for better extraction

        Returns:
            List of extracted keywords
        """
        # Clean the query
        query = query.lower().strip()

        # Remove common phrases
        query = re.sub(r'(search for|find out|tell me about|information on|what is|who is)', '', query)

        # Extract quoted phrases first (these are exact search terms)
        quoted = re.findall(r'"([^"]+)"', query)
        keywords = quoted.copy()

        # Remove quoted phrases from query
        for q in quoted:
            query = query.replace(f'"{q}"', '')

        # Extract capitalized words/names (likely important entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        keywords.extend(capitalized)

        # Extract remaining meaningful words
        words = re.findall(r'\b[a-z]{3,}\b', query)  # Words 3+ chars
        meaningful_words = [w for w in words if w not in self.stop_words]

        # Get top meaningful words (max 5)
        keywords.extend(meaningful_words[:5])

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        return unique_keywords[:10]  # Max 10 keywords

    def extract_from_plan_step(self, step_description: str) -> List[str]:
        """
        Extract keywords from a plan step description.

        Args:
            step_description: Plan step text

        Returns:
            List of keywords
        """
        # Look for specific patterns in plan steps
        # Example: "Step 1: Search for Pathaan box office - Tool: google_search"

        # Extract content before "Tool:" marker
        if "Tool:" in step_description or "tool:" in step_description:
            step_description = re.split(r'Tool:', step_description, flags=re.IGNORECASE)[0]

        # Extract content before "Expected:" marker
        if "Expected:" in step_description or "expected:" in step_description:
            step_description = re.split(r'Expected:', step_description, flags=re.IGNORECASE)[0]

        # Remove step numbers
        step_description = re.sub(r'^Step\s+\d+:', '', step_description, flags=re.IGNORECASE)

        return self.extract_keywords(step_description)


class WebScraper:
    """Scrape content from web pages."""

    def __init__(self):
        """Initialize web scraper."""
        try:
            self.ua = UserAgent()
        except:
            self.ua = None

        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def get_user_agent(self) -> str:
        """Get a random user agent."""
        if self.ua:
            try:
                return self.ua.random
            except:
                pass
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

    def scrape_url(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape
            timeout: Request timeout in seconds

        Returns:
            Dictionary with scraped content
        """
        try:
            # Update user agent
            self.session.headers['User-Agent'] = self.get_user_agent()

            # Make request
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()

            # Extract main content using trafilatura (best for articles)
            content = trafilatura.extract(response.text)

            if not content:
                # Fallback to BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Get text from main content areas
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if main_content:
                    content = main_content.get_text(separator=' ', strip=True)
                else:
                    content = soup.get_text(separator=' ', strip=True)

            # Clean up content
            content = ' '.join(content.split())  # Normalize whitespace

            # Get title
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title')
            title = title.get_text().strip() if title else urlparse(url).netloc

            return {
                'url': url,
                'title': title,
                'content': content[:5000],  # Limit to 5000 chars
                'success': True,
                'length': len(content)
            }

        except Exception as e:
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': str(e)
            }


class SearchCache:
    """Cache search results to avoid duplicate requests."""

    def __init__(self, cache_dir: str = ".cache/search"):
        """Initialize cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().encode()).hexdigest()

    def get(self, query: str, max_age: int = 3600) -> Optional[Dict]:
        """
        Get cached results.

        Args:
            query: Search query
            max_age: Maximum age in seconds (default 1 hour)

        Returns:
            Cached results or None
        """
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            # Check age
            age = time.time() - cache_file.stat().st_mtime
            if age < max_age:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except:
                    pass

        return None

    def set(self, query: str, results: Dict):
        """
        Cache results.

        Args:
            query: Search query
            results: Results to cache
        """
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except:
            pass


class FreeWebSearchTool:
    """
    Free web search tool using DuckDuckGo and web scraping.

    No API keys required! Perfect for research purposes.
    """

    def __init__(self):
        """Initialize the search tool."""
        self.keyword_extractor = KeywordExtractor()
        self.web_scraper = WebScraper()
        self.cache = SearchCache()

        # Check if OpenAI is available for summarization
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                except:
                    pass

        # Check if scraping libraries are available
        if not SCRAPING_AVAILABLE:
            print("Warning: Web scraping libraries not installed.")
            print("Install with: pip install -r requirements_search.txt")

    def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        num_results: int = 5,
        scrape_content: bool = True,
        summarize: bool = True,
        **kwargs
    ) -> str:
        """
        Perform a free web search and scrape top results.

        Args:
            query: Search query
            context: Optional context (can contain plan step info)
            num_results: Number of results to retrieve (default 5)
            scrape_content: Whether to scrape website content (default True)
            summarize: Whether to summarize results with LLM (default True)

        Returns:
            Formatted search results with scraped content
        """
        if not SCRAPING_AVAILABLE:
            return self._get_installation_instructions()

        try:
            # Extract keywords from query
            keywords = self.keyword_extractor.extract_keywords(query, context)

            # Check if plan step has specific search terms
            if context and 'step' in context:
                step_keywords = self.keyword_extractor.extract_from_plan_step(
                    str(context['step'])
                )
                keywords = list(set(keywords + step_keywords))

            # Construct search query from keywords
            search_query = ' '.join(keywords[:5]) if keywords else query

            # Check cache first
            cached = self.cache.get(search_query)
            if cached:
                return self._format_results(cached, summarize)

            # Perform DuckDuckGo search
            results = self._search_duckduckgo(search_query, num_results)

            if not results:
                return f"No results found for: {search_query}"

            # Scrape website content from top results (in parallel)
            if scrape_content:
                results = self._scrape_results(results, max_sites=min(3, num_results))

            # Cache results
            search_data = {
                'query': search_query,
                'keywords': keywords,
                'results': results,
                'timestamp': time.time()
            }
            self.cache.set(search_query, search_data)

            # Format and return
            return self._format_results(search_data, summarize)

        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search using DuckDuckGo (free, no API key needed).

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of search results
        """
        results = []

        try:
            with DDGS() as ddgs:
                # Perform search
                search_results = list(ddgs.text(query, max_results=num_results))

                for item in search_results:
                    results.append({
                        'title': item.get('title', ''),
                        'url': item.get('href', ''),
                        'snippet': item.get('body', ''),
                        'source': urlparse(item.get('href', '')).netloc
                    })
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")

        return results

    def _scrape_results(self, results: List[Dict], max_sites: int = 3) -> List[Dict]:
        """
        Scrape content from top search results in parallel.

        Args:
            results: List of search results
            max_sites: Maximum number of sites to scrape

        Returns:
            Results with scraped content added
        """
        # Select top results to scrape
        urls_to_scrape = [r['url'] for r in results[:max_sites]]

        # Scrape in parallel
        scraped_content = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {
                executor.submit(self.web_scraper.scrape_url, url): url
                for url in urls_to_scrape
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    scraped_content[url] = content
                except Exception as e:
                    scraped_content[url] = {'success': False, 'error': str(e)}

        # Add scraped content to results
        for result in results:
            url = result['url']
            if url in scraped_content:
                result['scraped'] = scraped_content[url]

        return results

    def _format_results(self, search_data: Dict, summarize: bool = True) -> str:
        """
        Format search results for output.

        Args:
            search_data: Search data with results
            summarize: Whether to use LLM summarization

        Returns:
            Formatted string
        """
        query = search_data.get('query', '')
        keywords = search_data.get('keywords', [])
        results = search_data.get('results', [])

        if not results:
            return f"No results found for: {query}"

        # If LLM summarization is requested and available
        if summarize and self.openai_client:
            return self._summarize_with_llm(query, keywords, results)

        # Otherwise, format manually
        output = f"🔍 Web Search Results for: {query}\n"
        if keywords:
            output += f"   Keywords: {', '.join(keywords)}\n"
        output += "\n"

        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"   Source: {result['source']}\n"
            output += f"   URL: {result['url']}\n"
            output += f"   {result['snippet'][:200]}...\n"

            # Add scraped content if available
            if 'scraped' in result and result['scraped'].get('success'):
                content = result['scraped']['content']
                if content:
                    output += f"   📄 Content Preview: {content[:300]}...\n"

            output += "\n"

        return output.strip()

    def _summarize_with_llm(
        self,
        query: str,
        keywords: List[str],
        results: List[Dict]
    ) -> str:
        """
        Summarize search results using LLM.

        Args:
            query: Original query
            keywords: Extracted keywords
            results: Search results with optional scraped content

        Returns:
            LLM-generated summary
        """
        # Prepare content for LLM
        content_parts = []

        for i, result in enumerate(results[:5], 1):
            content_parts.append(f"\n{i}. {result['title']}")
            content_parts.append(f"   Source: {result['source']}")
            content_parts.append(f"   {result['snippet']}")

            # Add scraped content if available (higher quality)
            if 'scraped' in result and result['scraped'].get('success'):
                scraped = result['scraped']['content'][:1000]  # Limit length
                if scraped:
                    content_parts.append(f"   Content: {scraped}")

        results_text = '\n'.join(content_parts)

        # Create prompt
        system_prompt = (
            "You are a helpful research assistant that synthesizes web search results "
            "into concise, accurate answers. Focus on answering the user's query with "
            "factual information from the sources. Cite sources when mentioning specific facts."
        )

        user_prompt = f"""Query: {query}
Keywords: {', '.join(keywords)}

Search Results and Web Content:
{results_text}

Please provide a comprehensive answer to the query using the information from these sources.
Structure your answer clearly and cite sources where appropriate."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=700
            )

            summary = response.choices[0].message.content.strip()

            # Add source links
            output = f"🔍 Web Search Answer:\n\n{summary}\n\n"
            output += "📚 Sources:\n"
            for i, result in enumerate(results[:3], 1):
                output += f"{i}. {result['title']} - {result['url']}\n"

            return output

        except Exception as e:
            # Fall back to basic formatting
            return self._format_results({'query': query, 'keywords': keywords, 'results': results}, False)

    def parallel_search(
        self,
        queries: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Perform multiple searches in parallel.

        Args:
            queries: List of search queries
            context: Optional context

        Returns:
            Dictionary mapping queries to results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {
                executor.submit(self.search, query, context): query
                for query in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results[query] = future.result()
                except Exception as e:
                    results[query] = f"Error: {str(e)}"

        return results

    def _get_installation_instructions(self) -> str:
        """Return installation instructions."""
        return """
🔧 Free Web Search Tool - Installation Required

This tool provides FREE web search without any API keys, but requires some Python packages.

Install dependencies:
    pip install -r requirements_search.txt

Or install manually:
    pip install duckduckgo-search beautifulsoup4 requests trafilatura fake-useragent

Features once installed:
✅ Free DuckDuckGo search (no API keys!)
✅ Automatic keyword extraction
✅ Web content scraping from top results
✅ Parallel search support
✅ Smart caching
✅ Works exactly like human browsing

Perfect for research purposes!
"""


# Factory and interface functions

def create_google_search() -> FreeWebSearchTool:
    """
    Factory function to create search tool.

    Returns:
        FreeWebSearchTool instance
    """
    return FreeWebSearchTool()


def google_search_tool(
    query: str,
    context: Optional[Dict] = None,
    **kwargs
) -> str:
    """
    Simple interface for execution engine.

    This is the primary interface used by ExecutionEngine.

    Args:
        query: Search query
        context: Optional context (may contain plan step info)
        **kwargs: Additional arguments

    Returns:
        Formatted search results
    """
    tool = create_google_search()

    # Extract parameters
    num_results = kwargs.get('num_results', 5)
    scrape_content = kwargs.get('scrape_content', True)
    summarize = kwargs.get('summarize', True)

    return tool.search(
        query=query,
        context=context,
        num_results=num_results,
        scrape_content=scrape_content,
        summarize=summarize
    )


# Test harness
if __name__ == "__main__":
    print("=" * 70)
    print("FREE WEB SEARCH TOOL - Test Mode")
    print("=" * 70)

    tool = create_google_search()

    # Test 1: Simple search
    print("\n📝 Test 1: Simple Search")
    print("-" * 70)
    query = "Shah Rukh Khan Pathaan box office collection"
    result = tool.search(query, num_results=3, scrape_content=True)
    print(result)

    # Test 2: Keyword extraction
    print("\n\n📝 Test 2: Keyword Extraction")
    print("-" * 70)
    query = "Tell me about the latest Bollywood news for Alia Bhatt"
    keywords = tool.keyword_extractor.extract_keywords(query)
    print(f"Query: {query}")
    print(f"Extracted Keywords: {keywords}")

    # Test 3: Plan step extraction
    print("\n\n📝 Test 3: Extract from Plan Step")
    print("-" * 70)
    step = "Step 1: Search for Pathaan movie reviews - Tool: google_search - Expected: Reviews"
    keywords = tool.keyword_extractor.extract_from_plan_step(step)
    print(f"Plan Step: {step}")
    print(f"Extracted Keywords: {keywords}")

    # Test 4: Parallel search
    print("\n\n📝 Test 4: Parallel Search")
    print("-" * 70)
    queries = [
        "Jawan movie cast",
        "Dunki release date",
        "Tiger 3 reviews"
    ]
    print(f"Searching for: {queries}")
    results = tool.parallel_search(queries)
    for q, r in results.items():
        print(f"\n{q}:")
        print(r[:200] + "...")

    print("\n" + "=" * 70)
    print("✅ Tests Complete!")
    print("=" * 70)
