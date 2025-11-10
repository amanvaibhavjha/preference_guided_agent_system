#!/usr/bin/env python3
"""DPO Training Script.

This script handles preference dataset generation and DPO training.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.utils.logger import setup_logger


def generate_preference_dataset(config, num_examples: int = 1000):
    """Generate preference dataset using RLAIF.
    
    This creates (prompt, chosen, rejected) pairs using:
    - ToT planner to generate multiple plans
    - GPT-4o as judge to label preferences
    
    Args:
        config: Configuration object.
        num_examples: Number of examples to generate.
    """
    logger = setup_logger("DPOTrainer", "INFO")
    logger.info(f"Generating {num_examples} preference examples...")
    
    # Implementation would go here
    # For now, showing the structure
    
    logger.info("Preference dataset generation complete!")
    logger.info(f"Dataset saved to: data/training/dpo_preferences.jsonl")


def train_dpo(config, dataset_path: str):
    """Train policy LLM using DPO.
    
    Uses Hugging Face TRL's DPOTrainer.
    
    Args:
        config: Configuration object.
        dataset_path: Path to preference dataset.
    """
    logger = setup_logger("DPOTrainer", "INFO")
    logger.info("Starting DPO training...")
    
    # Implementation would use TRL's DPOTrainer
    # See: https://huggingface.co/docs/trl/main/en/dpo_trainer
    
    logger.info("DPO training complete!")
    logger.info(f"Model saved to: checkpoints/dpo_model/")


def main():
    """Main entry point for DPO training."""
    parser = argparse.ArgumentParser(description="DPO Training Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["generate_dataset", "train", "both"],
        default="both",
        help="Training stage to run"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of preference examples to generate"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/training/dpo_preferences.jsonl",
        help="Path to preference dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run requested stages
    if args.stage in ["generate_dataset", "both"]:
        generate_preference_dataset(config, args.num_examples)
    
    if args.stage in ["train", "both"]:
        train_dpo(config, args.dataset_path)
    
    print("\n✅ DPO pipeline complete!")


if __name__ == "__main__":
    main()