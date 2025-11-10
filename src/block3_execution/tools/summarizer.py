"""Multilingual Summarizer Tool - LLM-based implementation.

This tool summarizes movie plots, articles, reviews, and other
Bollywood-related content. Supports Hindi-English mixed text.
"""

import os
from typing import Dict, Any, Literal
from openai import OpenAI


class MultilingualSummarizer:
    """LLM-based summarizer with Hindi-English support."""
    
    def __init__(self):
        """Initialize the summarizer."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def summarize(
        self,
        text: str,
        length: Literal["brief", "medium", "detailed"] = "medium",
        context: Dict[str, Any] = None
    ) -> str:
        """Summarize text.
        
        Args:
            text: Text to summarize.
            length: Summary length (brief/medium/detailed).
            context: Optional context information.
            
        Returns:
            Summary of the text.
        """
        length_map = {
            "brief": "2-3 sentences",
            "medium": "1 paragraph (4-6 sentences)",
            "detailed": "2-3 paragraphs"
        }
        
        system_prompt = f"""You are an expert summarizer for Bollywood content.
You can handle both English and Hindi-English mixed text (Hinglish).

Create clear, concise summaries that:
1. Capture the main points
2. Maintain key details
3. Are easy to understand
4. Preserve important names, dates, and facts

Target length: {length_map.get(length, 'medium')}"""
        
        user_prompt = f"""Summarize the following text:

{text}

Provide a {length} summary that captures the essential information."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=300 if length == "brief" else 500 if length == "medium" else 700
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
    
    def summarize_plot(self, movie_name: str) -> str:
        """Summarize a movie plot.
        
        Args:
            movie_name: Name of the movie.
            
        Returns:
            Plot summary.
        """
        system_prompt = """You are a movie plot summarizer.
Provide plot summaries that:
1. Don't include spoilers for the ending
2. Capture the main storyline
3. Mention key characters
4. Highlight the genre and tone"""
        
        user_prompt = f"""Provide a plot summary for the Bollywood movie: {movie_name}

Include the basic premise, main characters, and story arc without major spoilers."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error summarizing plot: {str(e)}"
    
    def summarize_article(self, article_text: str, focus: str = None) -> str:
        """Summarize a news article or blog post.
        
        Args:
            article_text: Article content.
            focus: Optional focus area (e.g., "box office", "reviews", "controversy").
            
        Returns:
            Article summary.
        """
        focus_str = f" Focus on: {focus}" if focus else ""
        
        system_prompt = f"""You are summarizing a Bollywood news article or blog post.{focus_str}

Provide a summary that includes:
1. Main news/topic
2. Key facts and figures
3. Important quotes (if any)
4. Context and significance"""
        
        user_prompt = f"""Summarize this article:

{article_text}

Provide a clear, informative summary."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error summarizing article: {str(e)}"
    
    def extract_key_points(self, text: str, num_points: int = 5) -> str:
        """Extract key points from text.
        
        Args:
            text: Text to extract points from.
            num_points: Number of key points to extract.
            
        Returns:
            List of key points.
        """
        system_prompt = f"""You are extracting key points from text.
Create a numbered list of exactly {num_points} key points.
Each point should be clear and specific."""
        
        user_prompt = f"""Extract {num_points} key points from this text:

{text}

Format as a numbered list."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error extracting key points: {str(e)}"


# Factory function
def create_summarizer() -> MultilingualSummarizer:
    """Create summarizer tool instance."""
    return MultilingualSummarizer()


# Simple interface for execution engine
def summarizer_tool(query: str, context: Dict = None, **kwargs) -> str:
    """Simple function interface for the execution engine.
    
    Args:
        query: Text to summarize or query requesting a summary.
        context: Optional context.
        **kwargs: Additional arguments (can include 'length').
        
    Returns:
        Summary.
    """
    summarizer = create_summarizer()
    
    # Check if query is asking for a movie plot summary
    if any(keyword in query.lower() for keyword in ["plot", "story of", "about the movie"]):
        # Extract movie name (simple approach)
        return summarizer.summarize_plot(query)
    else:
        # Summarize the query text itself
        length = kwargs.get('length', 'medium')
        return summarizer.summarize(query, length=length, context=context)


if __name__ == "__main__":
    # Test the tool
    summarizer = create_summarizer()
    
    print("=== Testing Summarizer Tool ===\n")
    
    # Test 1: Brief summary
    print("Test 1: Brief summary")
    print("-" * 50)
    long_text = """Shah Rukh Khan, often referred to as SRK, is one of the most 
    successful actors in Bollywood. He began his career in television before making 
    his film debut in 1992. Over the years, he has acted in more than 80 films and 
    won numerous awards. His notable films include Dilwale Dulhania Le Jayenge, 
    Kuch Kuch Hota Hai, and My Name Is Khan. He is known for his romantic roles 
    and has a massive fan following worldwide. In 2023, he made a comeback with 
    Pathaan which became one of the highest-grossing Indian films."""
    result = summarizer.summarize(long_text, length="brief")
    print(result)
    print()
    
    # Test 2: Plot summary
    print("Test 2: Movie plot summary")
    print("-" * 50)
    result = summarizer.summarize_plot("3 Idiots")
    print(result)
    print()
    
    # Test 3: Key points extraction
    print("Test 3: Extract key points")
    print("-" * 50)
    article = """Jawan, directed by Atlee, is Shah Rukh Khan's second release 
    of 2023. The film features SRK in a double role and includes an ensemble cast. 
    It deals with themes of social justice and corruption. The movie was praised 
    for its action sequences and emotional depth. It crossed 1000 crores worldwide, 
    becoming one of the highest-grossing Indian films. Critics appreciated SRK's 
    performance and Atlee's direction."""
    result = summarizer.extract_key_points(article, num_points=3)
    print(result)