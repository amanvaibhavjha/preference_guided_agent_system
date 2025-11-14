"""
Dynamic Self-Evolving Knowledge Graph

This knowledge graph:
- Starts empty and grows from user interactions
- Learns from all tool outputs (search, sentiment, news, etc.)
- Avoids redundant information
- Persists to disk for long-term memory
- Uses NetworkX for graph operations
- Provides smart querying capabilities

Perfect for building agent memory over time!
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import networkx as nx
from dataclasses import dataclass, asdict
import re

# OpenAI for entity/relation extraction
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str  # Unique identifier (lowercased name)
    name: str  # Display name
    type: str  # Entity type (Person, Movie, Event, etc.)
    properties: Dict[str, Any]  # Additional properties
    sources: List[str]  # Where this info came from
    confidence: float  # Confidence score (0-1)
    created_at: float  # Timestamp
    updated_at: float  # Last update timestamp


@dataclass
class Relation:
    """Represents a relationship between entities."""
    source: str  # Source entity ID
    target: str  # Target entity ID
    relation_type: str  # Type of relationship
    properties: Dict[str, Any]  # Additional properties
    sources: List[str]  # Where this info came from
    confidence: float  # Confidence score (0-1)
    created_at: float  # Timestamp
    updated_at: float  # Last update timestamp


class EntityExtractor:
    """Extract entities and relationships from text using LLM."""

    def __init__(self):
        """Initialize extractor."""
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.client = OpenAI(api_key=api_key)
                except:
                    pass

    def extract_entities_and_relations(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to extract from
            context: Optional context (tool name, query, etc.)

        Returns:
            Tuple of (entities, relations)
        """
        if not self.client:
            return self._simple_extraction(text)

        # Use LLM for intelligent extraction
        system_prompt = """You are an expert knowledge graph builder for Bollywood/entertainment domain.
Extract entities and relationships from the given text.

Return JSON in this format:
{
  "entities": [
    {"name": "Shah Rukh Khan", "type": "Person", "properties": {"role": "Actor"}},
    {"name": "Pathaan", "type": "Movie", "properties": {"year": "2023"}}
  ],
  "relations": [
    {"source": "Shah Rukh Khan", "target": "Pathaan", "type": "acted_in", "properties": {}}
  ]
}

Entity types: Person, Movie, Song, Award, Event, Organization, Location
Relation types: acted_in, directed, produced, released, won, collaborated_with, part_of"""

        user_prompt = f"""Text: {text[:2000]}

Extract all relevant entities and relationships. Focus on:
- People (actors, directors, producers)
- Movies and songs
- Box office numbers, ratings
- Awards and events
- Collaborations and relationships

Return only valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            entities = result.get('entities', [])
            relations = result.get('relations', [])

            return entities, relations

        except Exception as e:
            print(f"LLM extraction failed: {e}, using simple extraction")
            return self._simple_extraction(text)

    def _simple_extraction(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Simple rule-based extraction as fallback."""
        entities = []
        relations = []

        # Extract capitalized names (likely entities)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        for name in set(names):
            entities.append({
                'name': name,
                'type': 'Entity',
                'properties': {}
            })

        # Extract numbers (box office, ratings, etc.)
        numbers = re.findall(r'₹?\s*\d+(?:,\d+)*(?:\.\d+)?\s*(?:crore|cr|million|rating|stars)?', text)
        for num in numbers[:5]:  # Limit to 5
            entities.append({
                'name': num,
                'type': 'Metric',
                'properties': {}
            })

        return entities, relations


class DynamicKnowledgeGraph:
    """
    Dynamic, self-evolving knowledge graph.

    Features:
    - Learns from all tool outputs
    - Avoids redundant information
    - Persists to disk
    - Smart querying
    - Confidence scoring
    """

    def __init__(self, storage_path: str = "data/knowledge_graph"):
        """Initialize knowledge graph."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.graph_file = self.storage_path / "graph.json"
        self.entities_file = self.storage_path / "entities.json"
        self.relations_file = self.storage_path / "relations.json"

        # NetworkX graph for operations
        self.graph = nx.MultiDiGraph()

        # Entity and relation stores
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # Entity extractor
        self.extractor = EntityExtractor()

        # Load existing graph
        self._load_graph()

        print(f"📚 Knowledge Graph loaded: {len(self.entities)} entities, {len(self.relations)} relations")

    def _load_graph(self):
        """Load graph from disk."""
        try:
            if self.entities_file.exists():
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for ent_data in data:
                        entity = Entity(**ent_data)
                        self.entities[entity.id] = entity
                        self.graph.add_node(entity.id, **asdict(entity))

            if self.relations_file.exists():
                with open(self.relations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for rel_data in data:
                        relation = Relation(**rel_data)
                        self.relations.append(relation)
                        self.graph.add_edge(
                            relation.source,
                            relation.target,
                            key=relation.relation_type,
                            **asdict(relation)
                        )
        except Exception as e:
            print(f"Error loading graph: {e}")

    def _save_graph(self):
        """Save graph to disk."""
        try:
            # Save entities
            entities_data = [asdict(e) for e in self.entities.values()]
            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)

            # Save relations
            relations_data = [asdict(r) for r in self.relations]
            with open(self.relations_file, 'w', encoding='utf-8') as f:
                json.dump(relations_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved: {len(self.entities)} entities, {len(self.relations)} relations")

        except Exception as e:
            print(f"Error saving graph: {e}")

    def _entity_id(self, name: str) -> str:
        """Generate entity ID from name."""
        return name.lower().strip().replace(' ', '_')

    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict] = None,
        source: str = "manual",
        confidence: float = 1.0
    ) -> str:
        """
        Add or update an entity.

        Args:
            name: Entity name
            entity_type: Type (Person, Movie, etc.)
            properties: Additional properties
            source: Where this info came from
            confidence: Confidence score (0-1)

        Returns:
            Entity ID
        """
        entity_id = self._entity_id(name)
        properties = properties or {}

        now = time.time()

        if entity_id in self.entities:
            # Update existing entity
            entity = self.entities[entity_id]

            # Merge properties (keep higher confidence values)
            for key, value in properties.items():
                if key not in entity.properties:
                    entity.properties[key] = value
                elif confidence >= entity.confidence:
                    entity.properties[key] = value

            # Add source if not already present
            if source not in entity.sources:
                entity.sources.append(source)

            # Update confidence (weighted average)
            entity.confidence = (entity.confidence + confidence) / 2
            entity.updated_at = now

            print(f"✏️  Updated entity: {name} ({entity_type})")

        else:
            # Create new entity
            entity = Entity(
                id=entity_id,
                name=name,
                type=entity_type,
                properties=properties,
                sources=[source],
                confidence=confidence,
                created_at=now,
                updated_at=now
            )
            self.entities[entity_id] = entity
            self.graph.add_node(entity_id, **asdict(entity))

            print(f"➕ Added entity: {name} ({entity_type})")

        self._save_graph()
        return entity_id

    def add_relation(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        properties: Optional[Dict] = None,
        source: str = "manual",
        confidence: float = 1.0
    ):
        """
        Add or update a relationship.

        Args:
            source_name: Source entity name
            target_name: Target entity name
            relation_type: Relationship type
            properties: Additional properties
            source: Where this info came from
            confidence: Confidence score
        """
        source_id = self._entity_id(source_name)
        target_id = self._entity_id(target_name)
        properties = properties or {}

        # Ensure entities exist
        if source_id not in self.entities:
            self.add_entity(source_name, "Entity", source=source, confidence=confidence)
        if target_id not in self.entities:
            self.add_entity(target_name, "Entity", source=source, confidence=confidence)

        # Check if relation exists
        existing = None
        for rel in self.relations:
            if (rel.source == source_id and
                rel.target == target_id and
                rel.relation_type == relation_type):
                existing = rel
                break

        now = time.time()

        if existing:
            # Update existing relation
            for key, value in properties.items():
                if key not in existing.properties:
                    existing.properties[key] = value

            if source not in existing.sources:
                existing.sources.append(source)

            existing.confidence = (existing.confidence + confidence) / 2
            existing.updated_at = now

            print(f"✏️  Updated relation: {source_name} --[{relation_type}]--> {target_name}")

        else:
            # Create new relation
            relation = Relation(
                source=source_id,
                target=target_id,
                relation_type=relation_type,
                properties=properties,
                sources=[source],
                confidence=confidence,
                created_at=now,
                updated_at=now
            )
            self.relations.append(relation)
            self.graph.add_edge(source_id, target_id, key=relation_type, **asdict(relation))

            print(f"➕ Added relation: {source_name} --[{relation_type}]--> {target_name}")

        self._save_graph()

    def learn_from_text(
        self,
        text: str,
        source: str = "unknown",
        context: Optional[Dict] = None
    ):
        """
        Learn entities and relations from text.

        Args:
            text: Text to learn from
            source: Source identifier (e.g., 'google_search', 'user_query')
            context: Optional context
        """
        print(f"\n🧠 Learning from {source}...")

        # Extract entities and relations
        entities, relations = self.extractor.extract_entities_and_relations(text, context)

        # Add entities
        for ent in entities:
            self.add_entity(
                name=ent['name'],
                entity_type=ent.get('type', 'Entity'),
                properties=ent.get('properties', {}),
                source=source,
                confidence=0.8  # LLM-extracted has good confidence
            )

        # Add relations
        for rel in relations:
            self.add_relation(
                source_name=rel['source'],
                target_name=rel['target'],
                relation_type=rel['type'],
                properties=rel.get('properties', {}),
                source=source,
                confidence=0.8
            )

        print(f"✅ Learned {len(entities)} entities and {len(relations)} relations")

    def query(
        self,
        query: str,
        max_hops: int = 2
    ) -> str:
        """
        Query the knowledge graph.

        Args:
            query: Natural language query
            max_hops: Maximum hops for graph traversal

        Returns:
            Answer string
        """
        query_lower = query.lower()

        # Extract entities from query
        entities_in_query = []
        for entity_id, entity in self.entities.items():
            if entity.name.lower() in query_lower:
                entities_in_query.append(entity)

        if not entities_in_query:
            return "No information found in knowledge graph. Try other tools."

        # Build answer from graph
        answer_parts = []

        for entity in entities_in_query:
            answer_parts.append(f"\n**{entity.name}** ({entity.type}):")

            # Add properties
            if entity.properties:
                for key, value in entity.properties.items():
                    answer_parts.append(f"  - {key}: {value}")

            # Find outgoing relations
            outgoing = [r for r in self.relations if r.source == entity.id]
            if outgoing:
                answer_parts.append(f"  Relations:")
                for rel in outgoing[:5]:  # Limit to 5
                    target_entity = self.entities.get(rel.target)
                    if target_entity:
                        answer_parts.append(
                            f"    - {rel.relation_type} → {target_entity.name}"
                        )

            # Add sources
            answer_parts.append(f"  Sources: {', '.join(entity.sources[:3])}")

        if not answer_parts:
            return "No relevant information found in knowledge graph."

        result = "📚 From Knowledge Graph:\n" + "\n".join(answer_parts)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        entity_types = {}
        for entity in self.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1

        relation_types = {}
        for relation in self.relations:
            relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1

        return {
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'entity_types': entity_types,
            'relation_types': relation_types,
            'graph_density': nx.density(self.graph),
            'storage_path': str(self.storage_path)
        }


# Factory and interface functions

def create_dynamic_knowledge_graph() -> DynamicKnowledgeGraph:
    """Create dynamic knowledge graph instance."""
    return DynamicKnowledgeGraph()


def dynamic_kg_tool(
    query: str,
    context: Optional[Dict] = None,
    **kwargs
) -> str:
    """
    Interface for execution engine.

    This tool:
    1. Queries the knowledge graph first
    2. If found, returns info
    3. If not found, suggests using other tools

    Args:
        query: User query
        context: Optional context
        **kwargs: Additional arguments

    Returns:
        Answer string
    """
    kg = create_dynamic_knowledge_graph()

    # Query the graph
    result = kg.query(query)

    # Learn from the query itself
    kg.learn_from_text(query, source="user_query", context=context)

    return result


# Test harness
if __name__ == "__main__":
    print("=" * 70)
    print("DYNAMIC KNOWLEDGE GRAPH - Test Mode")
    print("=" * 70)

    kg = DynamicKnowledgeGraph()

    # Test 1: Add entities manually
    print("\n📝 Test 1: Adding Entities")
    print("-" * 70)
    kg.add_entity("Shah Rukh Khan", "Person", {"role": "Actor", "nationality": "Indian"})
    kg.add_entity("Pathaan", "Movie", {"year": "2023", "genre": "Action"})
    kg.add_entity("Jawan", "Movie", {"year": "2023", "genre": "Action"})

    # Test 2: Add relations
    print("\n📝 Test 2: Adding Relations")
    print("-" * 70)
    kg.add_relation("Shah Rukh Khan", "Pathaan", "acted_in", {"role": "Lead"})
    kg.add_relation("Shah Rukh Khan", "Jawan", "acted_in", {"role": "Lead"})

    # Test 3: Learn from text
    print("\n📝 Test 3: Learning from Text")
    print("-" * 70)
    text = """
    Pathaan is a 2023 Indian action thriller film starring Shah Rukh Khan.
    The movie collected over 1000 crores at the box office worldwide.
    Deepika Padukone also starred in the film. It was directed by Siddharth Anand.
    """
    kg.learn_from_text(text, source="test_text")

    # Test 4: Query
    print("\n📝 Test 4: Querying Knowledge Graph")
    print("-" * 70)
    queries = [
        "Tell me about Shah Rukh Khan",
        "What do you know about Pathaan?",
        "Shah Rukh Khan movies"
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        result = kg.query(q)
        print(result)

    # Test 5: Statistics
    print("\n📝 Test 5: Statistics")
    print("-" * 70)
    stats = kg.get_stats()
    print(json.dumps(stats, indent=2))

    print("\n" + "=" * 70)
    print("✅ Tests Complete!")
    print("=" * 70)
