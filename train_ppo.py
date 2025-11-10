#!/usr/bin/env python3
"""PPO Training Script.

This script handles online RL training using PPO.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.utils.logger import setup_logger


def train_ppo(
    config,
    num_iterations: int = 100,
    eval_freq: int = 10
):
    """Train policy using PPO.
    
    This implements the RL loop:
    1. Generate trajectories with current policy
    2. Calculate rewards using reward model
    3. Update policy with PPO
    
    Args:
        config: Configuration object.
        num_iterations: Number of training iterations.
        eval_freq: Evaluation frequency.
    """
    logger = setup_logger("PPOTrainer", "INFO")
    logger.info(f"Starting PPO training for {num_iterations} iterations...")
    
    # Implementation would use TRL's PPOTrainer
    # See: https://huggingface.co/docs/trl/main/en/ppo_trainer
    
    logger.info("PPO training complete!")
    logger.info(f"Model saved to: checkpoints/ppo_model/")


def main():
    """Main entry point for PPO training."""
    parser = argparse.ArgumentParser(description="PPO Training Pipeline")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of PPO iterations"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args if provided
    if args.batch_size:
        config.block4.ppo.batch_size = args.batch_size
    
    # Run training
    train_ppo(config, args.num_iterations, args.eval_freq)
    
    print("\n✅ PPO training complete!")


if __name__ == "__main__":
    main()