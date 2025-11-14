"""
Google Search Tool for web information retrieval.

This tool provides web search capabilities using Google Custom Search API.
It can be used to retrieve current information, verify facts, and supplement
the knowledge base with up-to-date web content.
"""

import os
import json
from typing import Dict, Any, List, Optional
import requests
from openai import OpenAI


class GoogleSearchTool:
    """
    Google Search tool for web information retrieval.

    This tool uses Google Custom Search JSON API to perform web searches
    and can optionally use LLM to summarize and format results.

    Environment Variables Required:
        GOOGLE_API_KEY: Google Cloud API key with Custom Search enabled
        GOOGLE_SEARCH_ENGINE_ID: Custom Search Engine ID (CX)
        OPENAI_API_KEY: OpenAI API key for result summarization (optional)
    """

    def __init__(self):
        """Initialize the Google Search tool."""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize OpenAI client for result summarization
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.model = "gpt-4o-mini"
        else:
            self.client = None
            self.model = None

        # API endpoint
        self.search_url = "https://www.googleapis.com/customsearch/v1"

    def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        num_results: int = 5,
        summarize: bool = True
    ) -> str:
        """
        Execute a Google search and return formatted results.

        Args:
            query: Search query string
            context: Optional context dictionary with additional parameters
            num_results: Number of results to retrieve (1-10)
            summarize: Whether to use LLM to summarize results

        Returns:
            Formatted search results as a string
        """
        if not self.api_key or not self.search_engine_id:
            return self._get_mock_results(query)

        try:
            # Perform the search
            results = self._perform_search(query, num_results)

            if not results:
                return f"No results found for query: {query}"

            # Format results
            if summarize and self.client:
                return self._summarize_results(query, results, context)
            else:
                return self._format_results(results)

        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _perform_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform the actual Google Custom Search API call.

        Args:
            query: Search query
            num_results: Number of results to retrieve

        Returns:
            List of search result dictionaries
        """
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)  # API max is 10
        }

        response = requests.get(self.search_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract relevant information from results
        results = []
        for item in data.get('items', []):
            results.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'displayLink': item.get('displayLink', '')
            })

        return results

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as a readable string.

        Args:
            results: List of search result dictionaries

        Returns:
            Formatted string of results
        """
        formatted = "Google Search Results:\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   URL: {result['link']}\n"
            formatted += f"   {result['snippet']}\n\n"

        return formatted.strip()

    def _summarize_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Use LLM to summarize and synthesize search results.

        Args:
            query: Original search query
            results: List of search result dictionaries
            context: Optional context for summarization

        Returns:
            LLM-generated summary of results
        """
        # Prepare results for LLM
        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n{i}. {result['title']}\n"
            results_text += f"   Source: {result['displayLink']}\n"
            results_text += f"   {result['snippet']}\n"

        # Create prompt for summarization
        system_prompt = (
            "You are a helpful assistant that synthesizes web search results "
            "into concise, accurate summaries. Focus on answering the user's "
            "query with relevant information from the search results."
        )

        user_prompt = f"Query: {query}\n\nSearch Results:{results_text}\n\n"

        if context:
            user_prompt += f"Additional Context: {context.get('user_context', '')}\n\n"

        user_prompt += (
            "Please provide a concise summary that answers the query using "
            "information from these search results. Include relevant sources."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            summary = response.choices[0].message.content.strip()
            return f"Google Search Summary:\n\n{summary}"

        except Exception as e:
            # Fall back to basic formatting if LLM fails
            return self._format_results(results)

    def _get_mock_results(self, query: str) -> str:
        """
        Return mock results when API credentials are not available.

        This is useful for testing and development without API keys.

        Args:
            query: Search query

        Returns:
            Mock search results
        """
        return (
            f"[MOCK] Google Search Results for: {query}\n\n"
            "Note: Google Search API credentials not configured. "
            "Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables.\n\n"
            "To enable Google Search:\n"
            "1. Get a Google Cloud API key from https://console.cloud.google.com/\n"
            "2. Enable Custom Search API\n"
            "3. Create a Custom Search Engine at https://programmablesearchengine.google.com/\n"
            "4. Set environment variables:\n"
            "   export GOOGLE_API_KEY='your-api-key'\n"
            "   export GOOGLE_SEARCH_ENGINE_ID='your-cx-id'"
        )

    def search_news(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Search specifically for news articles.

        Args:
            query: Search query
            context: Optional context

        Returns:
            Formatted news search results
        """
        news_query = f"{query} news"
        return self.search(news_query, context, num_results=5)

    def search_with_date_range(
        self,
        query: str,
        start_date: str = None,
        end_date: str = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Search with date range filtering.

        Args:
            query: Search query
            start_date: Start date in format YYYYMMDD
            end_date: End date in format YYYYMMDD
            context: Optional context

        Returns:
            Formatted search results
        """
        # Add date range to query if provided
        if start_date and end_date:
            query = f"{query} after:{start_date} before:{end_date}"
        elif start_date:
            query = f"{query} after:{start_date}"
        elif end_date:
            query = f"{query} before:{end_date}"

        return self.search(query, context)


def create_google_search() -> GoogleSearchTool:
    """
    Factory function to create a GoogleSearchTool instance.

    Returns:
        GoogleSearchTool instance
    """
    return GoogleSearchTool()


def google_search_tool(
    query: str,
    context: Optional[Dict] = None,
    **kwargs
) -> str:
    """
    Simple function interface for the execution engine.

    This is the primary interface used by the ExecutionEngine to call
    the Google Search tool.

    Args:
        query: Search query string
        context: Optional context dictionary
        **kwargs: Additional arguments (num_results, summarize, etc.)

    Returns:
        Formatted search results as a string
    """
    tool = create_google_search()

    # Extract optional parameters from kwargs
    num_results = kwargs.get('num_results', 5)
    summarize = kwargs.get('summarize', True)

    return tool.search(
        query=query,
        context=context,
        num_results=num_results,
        summarize=summarize
    )
