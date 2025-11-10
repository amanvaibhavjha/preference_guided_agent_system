"""News and Gossip Aggregator Tool - LLM-based implementation.

This tool aggregates and summarizes the latest Bollywood news,
gossip, and industry updates.
"""

import os
from typing import Dict, Any, List, Literal
from openai import OpenAI
from datetime import datetime


class NewsAndGossipAggregator:
    """LLM-based news aggregator for Bollywood content."""
    
    def __init__(self):
        """Initialize the news aggregator."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def get_latest_news(
        self,
        topic: str = None,
        category: Literal["all", "box_office", "releases", "celebrity", "awards"] = "all",
        context: Dict[str, Any] = None
    ) -> str:
        """Get latest Bollywood news.
        
        Args:
            topic: Specific topic (actor, movie, etc.).
            category: News category.
            context: Optional context.
            
        Returns:
            Latest news summary.
        """
        topic_str = f" about {topic}" if topic else ""
        category_str = f" in the {category.replace('_', ' ')} category" if category != "all" else ""
        current_date = datetime.now().strftime("%B %Y")
        
        system_prompt = f"""You are a Bollywood news aggregator as of {current_date}.
Provide the latest news{topic_str}{category_str}.

Format your response as:
1. Headline/Main news
2. Key details
3. Additional context
4. Industry impact (if relevant)

Be informative and up-to-date based on your knowledge."""
        
        user_prompt = f"""What are the latest Bollywood news updates{topic_str}{category_str}?

Provide 3-5 recent news items with brief summaries."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error fetching news: {str(e)}"
    
    def get_trending_topics(self) -> str:
        """Get current trending topics in Bollywood.
        
        Returns:
            List of trending topics.
        """
        current_date = datetime.now().strftime("%B %Y")
        
        system_prompt = f"""You are analyzing Bollywood trends as of {current_date}.
Identify the most talked-about topics, movies, actors, and controversies.

Provide:
1. Top 5 trending topics
2. Brief explanation of why each is trending
3. Social media buzz indicators"""
        
        user_prompt = """What are the top trending topics in Bollywood right now?

List them with brief explanations."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error fetching trending topics: {str(e)}"
    
    def get_upcoming_releases(self, timeframe: str = "this month") -> str:
        """Get information about upcoming movie releases.
        
        Args:
            timeframe: Timeframe for releases (e.g., "this month", "next quarter").
            
        Returns:
            Information about upcoming releases.
        """
        current_date = datetime.now().strftime("%B %Y")
        
        system_prompt = f"""You are providing information about upcoming Bollywood 
movie releases as of {current_date}.

For each movie, include:
1. Movie title
2. Expected release date
3. Cast and director
4. Genre
5. Pre-release buzz/expectations"""
        
        user_prompt = f"""What are the anticipated Bollywood movie releases {timeframe}?

Provide details about the most awaited films."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error fetching upcoming releases: {str(e)}"
    
    def get_box_office_updates(self) -> str:
        """Get latest box office performance updates.
        
        Returns:
            Box office updates and analysis.
        """
        current_date = datetime.now().strftime("%B %Y")
        
        system_prompt = f"""You are providing box office updates for Bollywood as of {current_date}.

Include:
1. Top performing movies currently
2. Recent releases and their performance
3. Box office records or milestones
4. Weekend collections (if relevant)
5. Industry trends"""
        
        user_prompt = """What are the latest box office updates for Bollywood?

Provide current performance data and notable achievements."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error fetching box office updates: {str(e)}"
    
    def get_celebrity_news(self, celebrity_name: str = None) -> str:
        """Get celebrity news and updates.
        
        Args:
            celebrity_name: Specific celebrity (optional).
            
        Returns:
            Celebrity news.
        """
        celebrity_str = f" about {celebrity_name}" if celebrity_name else ""
        current_date = datetime.now().strftime("%B %Y")
        
        system_prompt = f"""You are providing celebrity news{celebrity_str} as of {current_date}.

Include:
1. Recent activities and projects
2. Personal life updates (public information only)
3. Social media highlights
4. Upcoming work
5. Industry interactions"""
        
        user_prompt = f"""What's the latest news{celebrity_str} in Bollywood?

Provide recent updates and interesting developments."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error fetching celebrity news: {str(e)}"


# Factory function
def create_news_aggregator() -> NewsAndGossipAggregator:
    """Create news aggregator tool instance."""
    return NewsAndGossipAggregator()


# Simple interface for execution engine
def news_aggregator_tool(query: str, context: Dict = None, **kwargs) -> str:
    """Simple function interface for the execution engine.
    
    Args:
        query: News query.
        context: Optional context.
        **kwargs: Additional arguments.
        
    Returns:
        News aggregation results.
    """
    aggregator = create_news_aggregator()
    
    # Parse query to determine what kind of news to fetch
    query_lower = query.lower()
    
    if "trending" in query_lower or "popular" in query_lower:
        return aggregator.get_trending_topics()
    elif "box office" in query_lower or "collection" in query_lower:
        return aggregator.get_box_office_updates()
    elif "upcoming" in query_lower or "release" in query_lower:
        return aggregator.get_upcoming_releases()
    elif any(word in query_lower for word in ["celebrity", "actor", "actress", "star"]):
        # Try to extract celebrity name from query
        return aggregator.get_celebrity_news()
    else:
        # General news
        return aggregator.get_latest_news(topic=query)


if __name__ == "__main__":
    # Test the tool
    aggregator = create_news_aggregator()
    
    print("=== Testing News Aggregator Tool ===\n")
    
    # Test 1: General news
    print("Test 1: Latest Bollywood news")
    print("-" * 50)
    result = aggregator.get_latest_news()
    print(result)
    print()
    
    # Test 2: Trending topics
    print("Test 2: Trending topics")
    print("-" * 50)
    result = aggregator.get_trending_topics()
    print(result)
    print()
    
    # Test 3: Box office
    print("Test 3: Box office updates")
    print("-" * 50)
    result = aggregator.get_box_office_updates()
    print(result)
    print()
    
    # Test 4: Celebrity news
    print("Test 4: Celebrity news")
    print("-" * 50)
    result = aggregator.get_celebrity_news("Shah Rukh Khan")
    print(result)
    print()
    
    # Test 5: Upcoming releases
    print("Test 5: Upcoming releases")
    print("-" * 50)
    result = aggregator.get_upcoming_releases("next quarter")
    print(result)