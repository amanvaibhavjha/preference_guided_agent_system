"""Logging utilities for the agent system."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(
    name: str = "PreferenceGuidedAgent",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup and configure logger.
    
    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class AgentLogger:
    """Specialized logger for agent operations."""
    
    def __init__(self, config):
        """Initialize agent logger.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        log_config = config.general.logging
        
        self.logger = setup_logger(
            name="AgentSystem",
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file')
        )
        
        # Initialize wandb if enabled
        self.wandb_enabled = log_config.get('wandb_enabled', False)
        if self.wandb_enabled:
            try:
                import wandb
                wandb.init(
                    project=log_config.get('wandb_project', 'preference-guided-agent'),
                    config=self._config_to_dict()
                )
                self.wandb = wandb
            except ImportError:
                self.logger.warning("wandb not installed. Disabling wandb logging.")
                self.wandb_enabled = False
    
    def _config_to_dict(self):
        """Convert config to dictionary for wandb."""
        return {
            'block1': self.config.block1.__dict__,
            'block2': {
                **{k: v for k, v in self.config.block2.__dict__.items() if k not in ['tot', 'dpo']},
                'tot': self.config.block2.tot.__dict__,
                'dpo': self.config.block2.dpo.__dict__
            },
            'block3': self.config.block3.__dict__,
            'block4': {
                **{k: v for k, v in self.config.block4.__dict__.items() if k != 'ppo'},
                'ppo': self.config.block4.ppo.__dict__
            },
        }
    
    def log_query(self, query: str):
        """Log user query."""
        self.logger.info(f"User Query: {query}")
    
    def log_embedding(self, embedding_info: dict):
        """Log embedding information."""
        self.logger.debug(f"Embedding: dim={embedding_info.get('dim')}, tokens={embedding_info.get('tokens')}")
    
    def log_planning_start(self):
        """Log planning phase start."""
        self.logger.info("Starting Tree of Thoughts planning...")
    
    def log_planning_step(self, step: int, thought: str, score: float):
        """Log planning step."""
        self.logger.debug(f"ToT Step {step}: score={score:.2f}, thought={thought[:100]}...")
    
    def log_planning_complete(self, plan: dict):
        """Log completed plan."""
        self.logger.info(f"Planning complete: {len(plan.get('steps', []))} steps")
        if self.wandb_enabled:
            self.wandb.log({"plan_steps": len(plan.get('steps', []))})
    
    def log_execution_start(self):
        """Log execution phase start."""
        self.logger.info("Starting execution...")
    
    def log_tool_call(self, tool_name: str, args: dict):
        """Log tool invocation."""
        self.logger.debug(f"Tool call: {tool_name}({args})")
    
    def log_tool_result(self, tool_name: str, success: bool, result: any):
        """Log tool result."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.debug(f"Tool {tool_name}: {status}")
    
    def log_execution_complete(self, final_answer: str):
        """Log execution completion."""
        self.logger.info(f"Execution complete. Answer: {final_answer[:100]}...")
    
    def log_reward(self, reward_components: dict):
        """Log reward calculation."""
        self.logger.info(f"Rewards: {reward_components}")
        if self.wandb_enabled:
            self.wandb.log(reward_components)
    
    def log_training_step(self, step: int, loss: float, metrics: dict):
        """Log training step."""
        self.logger.info(f"Training step {step}: loss={loss:.4f}")
        if self.wandb_enabled:
            self.wandb.log({"step": step, "loss": loss, **metrics})
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error."""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def close(self):
        """Close logger and wandb."""
        if self.wandb_enabled:
            self.wandb.finish()


# Global logger instance
_global_logger = None


def get_logger(config=None) -> AgentLogger:
    """Get or create global logger instance.
    
    Args:
        config: Configuration object (required for first call).
        
    Returns:
        AgentLogger instance.
    """
    global _global_logger
    if _global_logger is None:
        if config is None:
            raise ValueError("Config required for first logger initialization")
        _global_logger = AgentLogger(config)
    return _global_logger