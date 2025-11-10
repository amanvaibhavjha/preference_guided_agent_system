"""Tree of Thoughts (ToT) Implementation with Beam Search.

This module implements the Tree of Thoughts algorithm from Yao et al. (2023).
It explores multiple reasoning paths and selects the optimal one using beam search.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import heapq
from src.block2_planning.policy_llm import PolicyLLM
from src.utils.logger import get_logger


@dataclass
class ThoughtNode:
    """A node in the Tree of Thoughts."""
    thought: str
    score: float
    depth: int
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For heap comparison (higher score is better)."""
        return self.score > other.score
    
    def get_path(self) -> List[str]:
        """Get path from root to this node."""
        path = []
        current = self
        while current is not None:
            if current.thought:  # Skip root
                path.append(current.thought)
            current = current.parent
        return list(reversed(path))


class TreeOfThoughts:
    """Tree of Thoughts planner with beam search.
    
    This implements the SOTA planning technique that explores multiple
    reasoning paths and selects the most promising one.
    """
    
    def __init__(self, config, policy_llm: PolicyLLM):
        """Initialize Tree of Thoughts planner.
        
        Args:
            config: Configuration object with ToT settings.
            policy_llm: Policy LLM for generating and evaluating thoughts.
        """
        self.config = config.block2.tot
        self.policy_llm = policy_llm
        self.logger = get_logger()
        
        self.beam_width = self.config.beam_width
        self.max_depth = self.config.max_depth
        self.min_score_threshold = self.config.min_score_threshold
        self.exploration_factor = self.config.exploration_factor
        
        self.logger.logger.info(
            f"Initialized ToT: beam_width={self.beam_width}, "
            f"max_depth={self.max_depth}, "
            f"threshold={self.min_score_threshold}"
        )
    
    def plan(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Generate optimal plan using Tree of Thoughts.
        
        Args:
            query: User query to plan for.
            context: Optional context information.
            
        Returns:
            Dictionary containing:
                - best_path: List of thoughts in best reasoning path
                - plan: Structured execution plan
                - score: Score of best path
                - explored_nodes: Number of nodes explored
        """
        self.logger.log_planning_start()
        
        # Initialize root node
        root = ThoughtNode(
            thought="",
            score=10.0,  # Root has perfect score
            depth=0,
            state=context or {}
        )
        
        # Beam search
        best_leaf = self._beam_search(query, root)
        
        # Get best path
        best_path = best_leaf.get_path()
        
        self.logger.logger.info(
            f"ToT complete: best_path_length={len(best_path)}, "
            f"score={best_leaf.score:.2f}"
        )
        
        # Generate structured plan from best path
        plan = self.policy_llm.generate_plan(query, best_path, context)
        
        result = {
            'best_path': best_path,
            'plan': plan,
            'score': best_leaf.score,
            'depth': best_leaf.depth
        }
        
        self.logger.log_planning_complete(result)
        
        return result
    
    def _beam_search(self, query: str, root: ThoughtNode) -> ThoughtNode:
        """Perform beam search through thought space.
        
        Args:
            query: User query.
            root: Root node to start from.
            
        Returns:
            Best leaf node found.
        """
        # Priority queue (max heap) for beam
        beam = [root]
        best_leaf = root
        
        for depth in range(1, self.max_depth + 1):
            self.logger.logger.debug(f"ToT depth {depth}/{self.max_depth}")
            
            # Generate candidates for next level
            candidates = []
            
            for node in beam:
                # Generate child thoughts
                children = self._generate_children(query, node)
                candidates.extend(children)
            
            if not candidates:
                break
            
            # Select top-k candidates for next beam
            # Sort by score (descending)
            candidates.sort(reverse=True)
            beam = candidates[:self.beam_width]
            
            # Update best leaf
            for node in beam:
                if node.score > best_leaf.score:
                    best_leaf = node
            
            # Log progress
            avg_score = sum(n.score for n in beam) / len(beam)
            self.logger.logger.debug(
                f"Depth {depth}: {len(beam)} nodes, avg_score={avg_score:.2f}, "
                f"best_score={best_leaf.score:.2f}"
            )
            
            # Early stopping if all scores are below threshold
            if all(n.score < self.min_score_threshold for n in beam):
                self.logger.logger.info("Early stopping: all scores below threshold")
                break
        
        return best_leaf
    
    def _generate_children(
        self,
        query: str,
        parent: ThoughtNode
    ) -> List[ThoughtNode]:
        """Generate and evaluate child thoughts for a node.
        
        Args:
            query: User query.
            parent: Parent node.
            
        Returns:
            List of child ThoughtNodes.
        """
        # Generate thought candidates
        thoughts = self.policy_llm.generate_thoughts(
            query=query,
            current_state=parent.state,
            num_thoughts=self.beam_width
        )
        
        children = []
        
        for thought in thoughts:
            # Evaluate thought
            score = self.policy_llm.evaluate_thought(
                query=query,
                thought=thought,
                current_state=parent.state
            )
            
            # Apply exploration bonus for diversity
            # Slightly boost scores to encourage exploration
            if self.exploration_factor > 0:
                score = score * (1 + self.exploration_factor * 0.1)
            
            # Create child node
            child = ThoughtNode(
                thought=thought,
                score=score,
                depth=parent.depth + 1,
                parent=parent,
                state=self._update_state(parent.state, thought)
            )
            
            parent.children.append(child)
            children.append(child)
            
            self.logger.log_planning_step(
                step=child.depth,
                thought=thought,
                score=score
            )
        
        return children
    
    def _update_state(self, parent_state: Dict, thought: str) -> Dict:
        """Update state based on new thought.
        
        Args:
            parent_state: Parent node's state.
            thought: New thought.
            
        Returns:
            Updated state dictionary.
        """
        new_state = parent_state.copy()
        
        # Track reasoning history
        if 'thoughts_history' not in new_state:
            new_state['thoughts_history'] = []
        new_state['thoughts_history'].append(thought)
        
        return new_state
    
    def visualize_tree(self, root: ThoughtNode, max_depth: int = 3) -> str:
        """Visualize the thought tree (for debugging).
        
        Args:
            root: Root node.
            max_depth: Maximum depth to display.
            
        Returns:
            String representation of tree.
        """
        lines = []
        
        def _traverse(node: ThoughtNode, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            if node.thought:  # Skip root
                lines.append(
                    f"{prefix}[{node.score:.1f}] {node.thought[:60]}..."
                )
            
            for i, child in enumerate(node.children):
                is_last = i == len(node.children) - 1
                child_prefix = prefix + ("└── " if is_last else "├── ")
                next_prefix = prefix + ("    " if is_last else "│   ")
                _traverse(child, child_prefix, depth + 1)
        
        _traverse(root)
        return "\n".join(lines)


def create_tree_of_thoughts(config, policy_llm: PolicyLLM) -> TreeOfThoughts:
    """Factory function to create Tree of Thoughts planner.
    
    Args:
        config: Configuration object.
        policy_llm: Policy LLM instance.
        
    Returns:
        TreeOfThoughts instance.
    """
    return TreeOfThoughts(config, policy_llm)


if __name__ == "__main__":
    # Test Tree of Thoughts
    from src.utils.config import load_config
    from src.block2_planning.policy_llm import create_policy_llm
    
    config = load_config()
    policy_llm = create_policy_llm(config)
    tot = create_tree_of_thoughts(config, policy_llm)
    
    # Test planning
    query = "What are Shah Rukh Khan's top 3 movies and their IMDB ratings?"
    result = tot.plan(query)
    
    print("\n=== Tree of Thoughts Planning ===")
    print(f"Query: {query}")
    print(f"\nBest Path ({len(result['best_path'])} thoughts):")
    for i, thought in enumerate(result['best_path'], 1):
        print(f"{i}. {thought}")
    print(f"\nScore: {result['score']:.2f}")
    print(f"\nPlan: {result['plan']['total_steps']} steps")
    for step in result['plan']['steps']:
        print(f"  - {step['description']}")