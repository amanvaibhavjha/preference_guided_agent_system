"""Block 1: Context Fusion - Text Encoder using OpenAI Embeddings.

This module handles text embedding using OpenAI's state-of-the-art embedding models.
For now, we use text-embedding-3-large which provides excellent semantic understanding.
"""

import os
from typing import List, Union, Dict
import numpy as np
from openai import OpenAI
from src.utils.logger import get_logger


class TextEncoder:
    """Text encoder using OpenAI embeddings.
    
    This provides the "fused context vector" mentioned in the architecture.
    Currently simplified to use only text embeddings as requested.
    """
    
    def __init__(self, config):
        """Initialize text encoder.
        
        Args:
            config: Configuration object with Block 1 settings.
        """
        self.config = config.block1
        self.logger = get_logger()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = self.config.embedding_model
        self.embedding_dim = self.config.embedding_dim
        
        self.logger.logger.info(f"Initialized TextEncoder with model: {self.model}")
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text into embedding vector(s).
        
        Args:
            text: Single string or list of strings to encode.
            
        Returns:
            numpy array of embeddings. Shape: (embedding_dim,) for single text,
            (num_texts, embedding_dim) for multiple texts.
        """
        # Convert single text to list for consistent processing
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        self.logger.logger.debug(f"Encoding {len(texts)} text(s)")
        
        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Log embedding info
            self.logger.log_embedding({
                'dim': embeddings_array.shape[-1],
                'texts': len(texts),
                'tokens': sum(item.usage.total_tokens for item in [response] if hasattr(item, 'usage'))
            })
            
            # Return single embedding if input was single text
            return embeddings_array[0] if is_single else embeddings_array
            
        except Exception as e:
            self.logger.log_error(e, "TextEncoder.encode")
            raise
    
    def encode_query_context(self, query: str, context: Dict = None) -> np.ndarray:
        """Encode query with optional context into a fused vector.
        
        This is the main entry point for Block 1 that produces the
        "Fused Context Vector" mentioned in the architecture.
        
        Args:
            query: User query string.
            context: Optional context dictionary with additional info.
            
        Returns:
            Fused context vector (numpy array).
        """
        self.logger.logger.info("Creating fused context vector")
        
        # Construct enriched query with context
        enriched_query = self._construct_enriched_query(query, context)
        
        # Generate embedding
        embedding = self.encode(enriched_query)
        
        return embedding
    
    def _construct_enriched_query(self, query: str, context: Dict = None) -> str:
        """Construct enriched query by incorporating context.
        
        Args:
            query: Original user query.
            context: Optional context dictionary.
            
        Returns:
            Enriched query string.
        """
        if not context:
            return query
        
        # Add context information to query
        enriched_parts = [query]
        
        # Add conversation history if available
        if 'history' in context and context['history']:
            history_text = " Previous context: " + " ".join(context['history'][-3:])
            enriched_parts.append(history_text)
        
        # Add user preferences if available
        if 'preferences' in context and context['preferences']:
            prefs_text = f" User preferences: {context['preferences']}"
            enriched_parts.append(prefs_text)
        
        # Add domain context
        if 'domain' in context:
            domain_text = f" Domain: {context['domain']}"
            enriched_parts.append(domain_text)
        
        return " ".join(enriched_parts)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score (0 to 1).
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def batch_encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode multiple texts in batches for efficiency.
        
        Args:
            texts: List of strings to encode.
            batch_size: Batch size for processing. If None, uses config default.
            
        Returns:
            numpy array of embeddings with shape (num_texts, embedding_dim).
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)


def create_text_encoder(config) -> TextEncoder:
    """Factory function to create text encoder.
    
    Args:
        config: Configuration object.
        
    Returns:
        TextEncoder instance.
    """
    return TextEncoder(config)


if __name__ == "__main__":
    # Simple test
    from src.utils.config import load_config
    
    config = load_config()
    encoder = create_text_encoder(config)
    
    # Test encoding
    test_query = "Tell me about Shah Rukh Khan's latest movie"
    embedding = encoder.encode_query_context(test_query)
    
    print(f"Query: {test_query}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test similarity
    query2 = "What are SRK's recent films?"
    emb2 = encoder.encode_query_context(query2)
    sim = encoder.similarity(embedding, emb2)
    print(f"Similarity between queries: {sim:.4f}")