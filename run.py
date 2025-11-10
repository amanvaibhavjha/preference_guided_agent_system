#!/usr/bin/env python3
"""Main runner for the Preference-Guided Agent System.

This script orchestrates all four blocks:
1. Context Fusion (Text Encoding)
2. Planning (Tree of Thoughts + DPO)
3. Execution (LangGraph Tools)
4. RL Update (PPO + Rewards)
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logger
from src.block1_context_fusion.text_encoder import create_text_encoder
from src.block2_planning.policy_llm import create_policy_llm
from src.block2_planning.tree_of_thoughts import create_tree_of_thoughts
from src.block3_execution.execution_engine import create_execution_engine
from src.block4_rl.reward_calculator import create_reward_calculator


class PreferenceGuidedAgentSystem:
    """Main agent system integrating all blocks."""
    
    def __init__(self, config_path: str = None):
        """Initialize the agent system.
        
        Args:
            config_path: Path to configuration file.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logger
        self.logger = get_logger(self.config)
        self.logger.logger.info("=" * 80)
        self.logger.logger.info("Initializing Preference-Guided Agent System")
        self.logger.logger.info("=" * 80)
        
        # Initialize all blocks
        self.logger.logger.info("Initializing Block 1: Context Fusion...")
        self.text_encoder = create_text_encoder(self.config)
        
        self.logger.logger.info("Initializing Block 2: Planning...")
        self.policy_llm = create_policy_llm(self.config)
        self.tree_of_thoughts = create_tree_of_thoughts(
            self.config, self.policy_llm
        )
        
        self.logger.logger.info("Initializing Block 3: Execution...")
        self.execution_engine = create_execution_engine(self.config)
        
        self.logger.logger.info("Initializing Block 4: RL & Rewards...")
        self.reward_calculator = create_reward_calculator(self.config)
        
        self.logger.logger.info("System initialization complete!")
        self.logger.logger.info("=" * 80)
    
    def process_query(
        self,
        query: str,
        context: dict = None,
        ground_truth: dict = None
    ) -> dict:
        """Process a user query through the complete system.
        
        Args:
            query: User query string.
            context: Optional context dictionary.
            ground_truth: Optional ground truth for evaluation.
            
        Returns:
            Dictionary with complete results including:
                - embedding: Context vector
                - planning_result: ToT planning results
                - execution_result: Tool execution results
                - rewards: Reward scores
        """
        self.logger.log_query(query)
        
        # === BLOCK 1: Context Fusion ===
        self.logger.logger.info("\n[BLOCK 1] Context Fusion")
        self.logger.logger.info("-" * 40)
        
        embedding = self.text_encoder.encode_query_context(query, context)
        self.logger.logger.info(
            f"Generated embedding: shape={embedding.shape}, "
            f"norm={embedding.dot(embedding)**0.5:.4f}"
        )
        
        # === BLOCK 2: Planning ===
        self.logger.logger.info("\n[BLOCK 2] Planning with Tree of Thoughts")
        self.logger.logger.info("-" * 40)
        
        planning_result = self.tree_of_thoughts.plan(query, context)
        
        self.logger.logger.info(f"Best reasoning path ({len(planning_result['best_path'])} steps):")
        for i, thought in enumerate(planning_result['best_path'], 1):
            self.logger.logger.info(f"  {i}. {thought}")
        
        self.logger.logger.info(f"\nGenerated plan: {planning_result['plan']['total_steps']} steps")
        for step in planning_result['plan']['steps']:
            self.logger.logger.info(f"  - {step['description'][:80]}...")
        
        # === BLOCK 3: Execution ===
        self.logger.logger.info("\n[BLOCK 3] Execution")
        self.logger.logger.info("-" * 40)
        
        execution_result = self.execution_engine.execute_plan(
            plan=planning_result['plan'],
            query=query,
            context=context
        )
        
        self.logger.logger.info(f"Execution status: {execution_result['status']}")
        self.logger.logger.info(f"Tool results: {len(execution_result['tool_results'])}")
        self.logger.logger.info(f"\nFinal Answer:\n{execution_result['final_answer']}")
        
        # === BLOCK 4: Reward Calculation ===
        self.logger.logger.info("\n[BLOCK 4] Reward Calculation")
        self.logger.logger.info("-" * 40)
        
        rewards = self.reward_calculator.calculate_reward(
            query=query,
            agent_plan=planning_result['plan'],
            agent_answer=execution_result['final_answer'],
            ground_truth_plan=ground_truth.get('plan') if ground_truth else None,
            ground_truth_answer=ground_truth.get('answer') if ground_truth else None
        )
        
        self.logger.logger.info("Reward Components:")
        for component, value in rewards.items():
            self.logger.logger.info(f"  {component}: {value:.4f}")
        
        # Compile complete results
        results = {
            'query': query,
            'embedding': embedding,
            'planning_result': planning_result,
            'execution_result': execution_result,
            'rewards': rewards
        }
        
        return results
    
    def interactive_mode(self):
        """Run in interactive mode."""
        self.logger.logger.info("\n" + "=" * 80)
        self.logger.logger.info("INTERACTIVE MODE")
        self.logger.logger.info("Type 'exit' or 'quit' to end session")
        self.logger.logger.info("=" * 80 + "\n")
        
        while True:
            try:
                query = input("\n🤖 Enter your query: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    self.logger.logger.info("Ending session. Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n" + "=" * 80)
                results = self.process_query(query)
                print("=" * 80)
                
                # Print summary
                print("\n" + "🎯 SUMMARY".center(80))
                print("-" * 80)
                print(f"Query: {results['query']}")
                print(f"Planning Score: {results['planning_result']['score']:.2f}/10")
                print(f"Execution Status: {results['execution_result']['status']}")
                print(f"Final Reward: {results['rewards']['r_final']:.4f}")
                print(f"\nFinal Answer:")
                print(results['execution_result']['final_answer'])
                print("=" * 80)
                
            except KeyboardInterrupt:
                self.logger.logger.info("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                self.logger.log_error(e, "interactive_mode")
                print(f"\n❌ Error: {str(e)}\n")
    
    def close(self):
        """Clean up resources."""
        self.logger.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preference-Guided Agent System with RL"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    try:
        system = PreferenceGuidedAgentSystem(config_path=args.config)
        
        if args.interactive:
            # Interactive mode
            system.interactive_mode()
        elif args.query:
            # Single query mode
            results = system.process_query(args.query)
            
            print("\n" + "=" * 80)
            print("RESULTS".center(80))
            print("=" * 80)
            print(f"\nQuery: {results['query']}")
            print(f"\nPlanning Score: {results['planning_result']['score']:.2f}/10")
            print(f"Execution Status: {results['execution_result']['status']}")
            print(f"\nRewards:")
            for k, v in results['rewards'].items():
                print(f"  {k}: {v:.4f}")
            print(f"\nFinal Answer:")
            print(results['execution_result']['final_answer'])
            print("=" * 80 + "\n")
        else:
            # No query provided, show help
            parser.print_help()
            
        system.close()
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()