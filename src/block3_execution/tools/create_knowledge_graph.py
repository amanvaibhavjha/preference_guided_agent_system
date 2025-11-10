"""Knowledge Graph Tool - LLM-based implementation.

This tool simulates a knowledge graph using LLM queries.
For Bollywood domain, it can answer questions about:
- Actors and their movies
- Directors and filmography
- Collaborations
- Movie details
"""

import os
from typing import Dict, Any
from openai import OpenAI


class BollywoodKnowledgeGraph:
    """LLM-based knowledge graph for Bollywood domain."""
    
    def __init__(self):
        """Initialize the knowledge graph tool."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Query the knowledge graph.
        
        Args:
            query: Natural language query about Bollywood.
            context: Optional context information.
            
        Returns:
            Knowledge graph results as formatted string.
        """
        system_prompt = """You are a Bollywood Knowledge Graph assistant.
You have comprehensive knowledge about:
- Bollywood actors, actresses, and their filmographies
- Directors, producers, and their works
- Movie details (cast, crew, genre, year, ratings)
- Collaborations and relationships in the industry
- Box office performance and awards

Provide accurate, structured information in response to queries.
Format your response as clear, factual statements."""
        
        user_prompt = f"""Query: {query}

Provide detailed information from the knowledge graph.
Include relevant facts, relationships, and data points.
Be specific with names, dates, and numbers where applicable."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower for factual responses
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error querying knowledge graph: {str(e)}"
    
    def get_actor_movies(self, actor_name: str) -> str:
        """Get movies for a specific actor.
        
        Args:
            actor_name: Name of the actor.
            
        Returns:
            List of movies.
        """
        query = f"List the most notable movies of {actor_name} with years and roles"
        return self.query(query)
    
    def get_movie_details(self, movie_name: str) -> str:
        """Get details about a specific movie.
        
        Args:
            movie_name: Name of the movie.
            
        Returns:
            Movie details.
        """
        query = f"Provide comprehensive details about the movie '{movie_name}' including cast, director, year, genre, and plot summary"
        return self.query(query)
    
    def get_collaborations(self, person1: str, person2: str) -> str:
        """Get collaborations between two people.
        
        Args:
            person1: First person's name.
            person2: Second person's name.
            
        Returns:
            Collaboration information.
        """
        query = f"List all movies where {person1} and {person2} worked together"
        return self.query(query)


# Factory function
def create_knowledge_graph() -> BollywoodKnowledgeGraph:
    """Create knowledge graph tool instance."""
    return BollywoodKnowledgeGraph()


# Simple interface for execution engine
def knowledge_graph_tool(query: str, context: Dict = None, **kwargs) -> str:
    """Simple function interface for the execution engine.
    
    Args:
        query: User query.
        context: Optional context.
        **kwargs: Additional arguments.
        
    Returns:
        Knowledge graph results.
    """
    kg = create_knowledge_graph()
    return kg.query(query, context)


if __name__ == "__main__":
    # Test the tool
    kg = create_knowledge_graph()
    
    print("=== Testing Knowledge Graph Tool ===\n")
    
    # Test 1: Actor query
    print("Test 1: Actor filmography")
    print("-" * 50)
    result = kg.get_actor_movies("Shah Rukh Khan")
    print(result)
    print()
    
    # Test 2: Movie details
    print("Test 2: Movie details")
    print("-" * 50)
    result = kg.get_movie_details("Dilwale Dulhania Le Jayenge")
    print(result)
    print()
    
    # Test 3: Collaborations
    print("Test 3: Collaborations")
    print("-" * 50)
    result = kg.get_collaborations("Shah Rukh Khan", "Kajol")
    print(result)