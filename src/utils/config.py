"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class Block1Config:
    """Block 1: Context Fusion Configuration."""
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    batch_size: int = 100
    max_tokens: int = 8192


@dataclass
class ToTConfig:
    """Tree of Thoughts Configuration."""
    beam_width: int = 3
    max_depth: int = 5
    min_score_threshold: float = 6.0
    exploration_factor: float = 0.7
    backtracking_enabled: bool = True


@dataclass
class DPOConfig:
    """DPO Training Configuration."""
    beta: float = 0.1
    learning_rate: float = 5e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100


@dataclass
class Block2Config:
    """Block 2: Planning Configuration."""
    policy_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    tot: ToTConfig = field(default_factory=ToTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)


@dataclass
class Block3Config:
    """Block 3: Execution Configuration."""
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    tools: list = field(default_factory=list)
    langgraph: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PPOConfig:
    """PPO Training Configuration."""
    learning_rate: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs_per_iteration: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.01


@dataclass
class Block4Config:
    """Block 4: RL Configuration."""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward_weights: Dict[str, float] = field(default_factory=dict)
    judge_model: str = "gpt-4o"
    judge_temperature: float = 0.3
    num_iterations: int = 100
    eval_frequency: int = 10
    save_frequency: int = 20


@dataclass
class GeneralConfig:
    """General Configuration."""
    seed: int = 42
    device: str = "cuda"
    mixed_precision: str = "bf16"
    logging: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Main Configuration Class."""
    block1: Block1Config = field(default_factory=Block1Config)
    block2: Block2Config = field(default_factory=Block2Config)
    block3: Block3Config = field(default_factory=Block3Config)
    block4: Block4Config = field(default_factory=Block4Config)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    domain: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Config object with all settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create nested config objects
    config = Config()
    
    # Block 1
    if 'block1' in config_dict:
        config.block1 = Block1Config(**config_dict['block1'])
    
    # Block 2
    if 'block2' in config_dict:
        block2_dict = config_dict['block2'].copy()
        tot_dict = block2_dict.pop('tot', {})
        dpo_dict = block2_dict.pop('dpo', {})
        
        config.block2 = Block2Config(
            **block2_dict,
            tot=ToTConfig(**tot_dict),
            dpo=DPOConfig(**dpo_dict)
        )
    
    # Block 3
    if 'block3' in config_dict:
        config.block3 = Block3Config(**config_dict['block3'])
    
    # Block 4
    if 'block4' in config_dict:
        block4_dict = config_dict['block4'].copy()
        ppo_dict = block4_dict.pop('ppo', {})
        
        config.block4 = Block4Config(
            **block4_dict,
            ppo=PPOConfig(**ppo_dict)
        )
    
    # General
    if 'general' in config_dict:
        config.general = GeneralConfig(**config_dict['general'])
    
    # Domain
    if 'domain' in config_dict:
        config.domain = config_dict['domain']
    
    # Create directories
    for path_key, path_value in config.general.paths.items():
        os.makedirs(path_value, exist_ok=True)
    
    return config


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        save_path: Path to save the config file.
    """
    # Convert dataclasses to dict
    config_dict = {
        'block1': config.block1.__dict__,
        'block2': {
            **{k: v for k, v in config.block2.__dict__.items() if k not in ['tot', 'dpo']},
            'tot': config.block2.tot.__dict__,
            'dpo': config.block2.dpo.__dict__
        },
        'block3': config.block3.__dict__,
        'block4': {
            **{k: v for k, v in config.block4.__dict__.items() if k != 'ppo'},
            'ppo': config.block4.ppo.__dict__
        },
        'general': config.general.__dict__,
        'domain': config.domain
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)