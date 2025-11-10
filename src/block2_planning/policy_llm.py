"""Policy LLM - The planning "brain" of the agent.

This module wraps the LLM used for planning decisions.
Currently uses GPT-4o-mini but can be swapped with other models.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.utils.logger import get_logger


class PolicyLLM:
    """Wrapper for the policy LLM used in planning."""
    
    def __init__(self, config):
        """Initialize policy LLM.
        
        Args:
            config: Configuration object with Block 2 settings.
        """
        self.config = config.block2
        self.logger = get_logger()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = self.config.policy_model
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        
        self.logger.logger.info(f"Initialized PolicyLLM with model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        system_message: str = None,
        temperature: float = None,
        max_tokens: int = None,
        stop: List[str] = None
    ) -> str:
        """Generate text completion from the LLM.
        
        Args:
            prompt: User prompt.
            system_message: Optional system message.
            temperature: Sampling temperature (overrides default).
            max_tokens: Max tokens to generate (overrides default).
            stop: Stop sequences.
            
        Returns:
            Generated text.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stop=stop
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.log_error(e, "PolicyLLM.generate")
            raise
    
    def generate_thoughts(
        self,
        query: str,
        current_state: Dict,
        num_thoughts: int = 3
    ) -> List[str]:
        """Generate multiple thought candidates for Tree of Thoughts.
        
        Args:
            query: Original user query.
            current_state: Current state in the planning tree.
            num_thoughts: Number of thoughts to generate.
            
        Returns:
            List of thought strings.
        """
        system_message = """You are an AI planning assistant using Tree of Thoughts reasoning.
Generate diverse, logical next steps for solving the user's query.
Each thought should be a clear, actionable step."""
        
        prompt = f"""User Query: {query}

Current State:
{self._format_state(current_state)}

Generate {num_thoughts} diverse next steps to progress toward answering the query.
Format your response as a numbered list:
1. [First thought]
2. [Second thought]
3. [Third thought]
"""
        
        response = self.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=0.8  # Higher temperature for diversity
        )
        
        # Parse thoughts from response
        thoughts = self._parse_thoughts(response, num_thoughts)
        
        return thoughts
    
    def evaluate_thought(
        self,
        query: str,
        thought: str,
        current_state: Dict
    ) -> float:
        """Evaluate how promising a thought is (1-10 scale).
        
        Args:
            query: Original user query.
            thought: Thought to evaluate.
            current_state: Current state context.
            
        Returns:
            Score from 1-10.
        """
        system_message = """You are an AI evaluator for Tree of Thoughts reasoning.
Rate how promising and logical a thought is for solving the query.
Respond with ONLY a number from 1-10."""
        
        prompt = f"""User Query: {query}

Current State:
{self._format_state(current_state)}

Thought to Evaluate: {thought}

Rate this thought from 1-10 where:
- 1-3: Poor, illogical, or unhelpful
- 4-6: Okay, somewhat relevant
- 7-9: Good, logical, and helpful
- 10: Excellent, optimal next step

Score:"""
        
        try:
            response = self.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,  # Lower temperature for consistent scoring
                max_tokens=10
            )
            
            # Extract numeric score
            score = self._extract_score(response)
            return score
            
        except Exception as e:
            self.logger.logger.warning(f"Error evaluating thought: {e}. Returning default score.")
            return 5.0  # Default to middle score
    
    def generate_plan(
        self,
        query: str,
        thought_path: List[str],
        context: Dict = None
    ) -> Dict[str, Any]:
        """Generate final structured plan from a thought path.
        
        Args:
            query: Original user query.
            thought_path: List of thoughts forming the reasoning path.
            context: Optional additional context.
            
        Returns:
            Structured plan dictionary.
        """
        system_message = """You are an AI planning assistant.
Convert a reasoning path into a structured, executable plan with clear steps and tools."""
        
        prompt = f"""User Query: {query}

Reasoning Path:
{self._format_path(thought_path)}

Create a structured plan with:
1. Clear, numbered steps
2. Specific tools to use for each step
3. Expected outputs

Format:
Step 1: [Action] - Tool: [tool_name] - Expected: [output]
Step 2: [Action] - Tool: [tool_name] - Expected: [output]
...
"""
        
        response = self.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=0.5
        )
        
        # Parse plan from response
        plan = self._parse_plan(response)
        
        return plan
    
    def _format_state(self, state: Dict) -> str:
        """Format state dictionary for display."""
        if not state:
            return "Initial state"
        
        parts = []
        for key, value in state.items():
            parts.append(f"- {key}: {value}")
        
        return "\n".join(parts)
    
    def _format_path(self, path: List[str]) -> str:
        """Format thought path for display."""
        return "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(path))
    
    def _parse_thoughts(self, response: str, expected_num: int) -> List[str]:
        """Parse thoughts from LLM response."""
        thoughts = []
        
        for line in response.split('\n'):
            line = line.strip()
            # Look for numbered items
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and add to thoughts
                thought = line.lstrip('0123456789.-) ').strip()
                if thought:
                    thoughts.append(thought)
        
        # Ensure we have enough thoughts
        while len(thoughts) < expected_num:
            thoughts.append(f"Continue with current approach (variant {len(thoughts)+1})")
        
        return thoughts[:expected_num]
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from response."""
        # Try to find a number in the response
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            score = float(numbers[0])
            # Clamp to 1-10 range
            return max(1.0, min(10.0, score))
        
        return 5.0  # Default
    
    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse structured plan from response."""
        steps = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line and ('Step' in line or line[0].isdigit()):
                # Extract step components
                step_dict = {
                    'description': line,
                    'tool': self._extract_tool(line),
                    'expected_output': self._extract_expected(line)
                }
                steps.append(step_dict)
        
        return {
            'steps': steps,
            'total_steps': len(steps)
        }
    
    def _extract_tool(self, text: str) -> Optional[str]:
        """Extract tool name from step description."""
        import re
        # Look for "Tool: [name]" pattern
        match = re.search(r'Tool:\s*(\w+)', text, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_expected(self, text: str) -> Optional[str]:
        """Extract expected output from step description."""
        import re
        # Look for "Expected: [description]" pattern
        match = re.search(r'Expected:\s*(.+?)(?:Step|\n|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None


def create_policy_llm(config) -> PolicyLLM:
    """Factory function to create policy LLM.
    
    Args:
        config: Configuration object.
        
    Returns:
        PolicyLLM instance.
    """
    return PolicyLLM(config)