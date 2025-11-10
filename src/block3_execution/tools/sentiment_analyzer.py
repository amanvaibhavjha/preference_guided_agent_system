"""Sentiment Analyzer Tool - LLM-based implementation.

This tool analyzes sentiment of movie reviews, social media posts,
or any text related to Bollywood content.
"""

import os
from typing import Dict, Any, List
from openai import OpenAI


class SentimentAnalyzer:
    """LLM-based sentiment analyzer for Bollywood content."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def analyze(self, text: str, context: Dict[str, Any] = None) -> str:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze (review, comment, etc.).
            context: Optional context (movie name, actor, etc.).
            
        Returns:
            Sentiment analysis results.
        """
        system_prompt = """You are a sentiment analysis expert for Bollywood content.
Analyze the sentiment of reviews, comments, or social media posts.

Provide:
1. Overall sentiment (Positive/Negative/Neutral/Mixed)
2. Sentiment score (0-10 scale)
3. Key emotional themes
4. Specific aspects mentioned (acting, direction, music, story, etc.)
5. Confidence level

Be objective and nuanced in your analysis."""
        
        user_prompt = f"""Analyze the sentiment of the following text:

Text: {text}

Provide a comprehensive sentiment analysis including overall sentiment, score, key themes, and specific aspects."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"
    
    def analyze_reviews(self, reviews: List[str], movie_name: str = None) -> str:
        """Analyze multiple reviews and provide aggregate sentiment.
        
        Args:
            reviews: List of review texts.
            movie_name: Optional movie name for context.
            
        Returns:
            Aggregate sentiment analysis.
        """
        combined_text = "\n\n---\n\n".join(reviews[:5])  # Limit to 5 reviews
        
        context_str = f" for the movie '{movie_name}'" if movie_name else ""
        
        system_prompt = f"""You are analyzing multiple reviews{context_str}.
Provide an aggregate sentiment analysis summarizing:
1. Overall consensus
2. Distribution of sentiments (% positive, negative, neutral)
3. Common themes across reviews
4. Specific praised aspects
5. Specific criticized aspects"""
        
        user_prompt = f"""Analyze these reviews and provide aggregate sentiment:

{combined_text}

Provide a comprehensive summary of the overall sentiment and key themes."""
        
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
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error analyzing reviews: {str(e)}"
    
    def compare_sentiment(self, text1: str, text2: str, labels: tuple = ("Text 1", "Text 2")) -> str:
        """Compare sentiment between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            labels: Labels for the texts.
            
        Returns:
            Comparative sentiment analysis.
        """
        system_prompt = """You are comparing sentiment between two texts.
Analyze both and highlight:
1. Sentiment differences
2. Tone differences
3. Key distinguishing factors
4. Which is more positive/negative overall"""
        
        user_prompt = f"""Compare the sentiment of these two texts:

{labels[0]}:
{text1}

{labels[1]}:
{text2}

Provide a comparative analysis."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error comparing sentiment: {str(e)}"


# Factory function
def create_sentiment_analyzer() -> SentimentAnalyzer:
    """Create sentiment analyzer tool instance."""
    return SentimentAnalyzer()


# Simple interface for execution engine
def sentiment_analyzer_tool(query: str, context: Dict = None, **kwargs) -> str:
    """Simple function interface for the execution engine.
    
    Args:
        query: Text to analyze or query about sentiment.
        context: Optional context.
        **kwargs: Additional arguments.
        
    Returns:
        Sentiment analysis results.
    """
    analyzer = create_sentiment_analyzer()
    
    # If query is asking about sentiment, analyze it
    # Otherwise, use query as text to analyze
    return analyzer.analyze(query, context)


if __name__ == "__main__":
    # Test the tool
    analyzer = create_sentiment_analyzer()
    
    print("=== Testing Sentiment Analyzer Tool ===\n")
    
    # Test 1: Single review
    print("Test 1: Single review analysis")
    print("-" * 50)
    review = """Pathaan is an absolute blockbuster! Shah Rukh Khan's 
    comeback is spectacular. The action sequences are mind-blowing and 
    Deepika Padukone looks stunning. However, the story could have been 
    better. Overall, a must-watch for SRK fans!"""
    result = analyzer.analyze(review)
    print(result)
    print()
    
    # Test 2: Multiple reviews
    print("Test 2: Aggregate review analysis")
    print("-" * 50)
    reviews = [
        "Amazing movie! Best action film of the year.",
        "SRK is back with a bang. Loved every minute.",
        "Good action but weak story. Still entertaining.",
        "Disappointed with the plot. Only action saves it.",
        "Perfect entertainer! Go watch it in theaters!"
    ]
    result = analyzer.analyze_reviews(reviews, "Pathaan")
    print(result)
    print()
    
    # Test 3: Comparison
    print("Test 3: Sentiment comparison")
    print("-" * 50)
    text1 = "Jawan is a masterpiece! SRK delivers his best performance."
    text2 = "Jawan is mediocre. Expected much more from SRK."
    result = analyzer.compare_sentiment(text1, text2, ("Review A", "Review B"))
    print(result)