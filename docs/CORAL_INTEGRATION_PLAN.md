# CORAL Integration & Architecture Enhancement Plan

## Executive Summary

This document outlines a comprehensive plan to integrate the CORAL (Conversational Recommendation with cOntrasting user pReferences via multi-Agent coLlaboration) dataset into the preference-guided agent system and enhance the architecture to support preference learning through DPO/PPO training.

## Table of Contents

1. [Understanding CORAL Dataset](#1-understanding-coral-dataset)
2. [Data Integration Strategy](#2-data-integration-strategy)
3. [Architecture Enhancements](#3-architecture-enhancements)
4. [Preference Learning Framework](#4-preference-learning-framework)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Expected Outcomes](#8-expected-outcomes)

---

## 1. Understanding CORAL Dataset

### 1.1 Dataset Overview

The CORAL dataset contains three conversational recommendation datasets:
- **Pearl**: General recommendation conversations
- **Inspired**: Creative recommendation dialogues
- **Redial**: Movie/entertainment recommendation dialogues

**Key Characteristics:**
- Multi-turn conversational data
- User preference annotations
- Contrasting preferences (user likes vs. dislikes)
- Agent-user interaction trajectories
- Ground truth recommendations

### 1.2 Expected Data Structure

Based on conversational recommendation standards, we expect:

```json
{
  "conversation_id": "unique_id",
  "user_id": "user_identifier",
  "turns": [
    {
      "turn_id": 1,
      "user_utterance": "I'm looking for a movie like Inception",
      "system_response": "I recommend Interstellar...",
      "items_mentioned": ["Inception"],
      "items_recommended": ["Interstellar"],
      "user_feedback": "positive"
    }
  ],
  "user_preferences": {
    "liked_items": ["Inception", "The Matrix"],
    "disliked_items": ["Romantic comedies"],
    "preference_features": {
      "genres": ["sci-fi", "thriller"],
      "themes": ["mind-bending", "philosophical"]
    }
  },
  "final_recommendation": "Interstellar",
  "success": true
}
```

### 1.3 Preference Annotations

CORAL emphasizes **contrasting preferences**:
- Positive preferences (what users like)
- Negative preferences (what users dislike)
- Implicit preferences (inferred from behavior)
- Explicit preferences (stated directly)

---

## 2. Data Integration Strategy

### 2.1 Data Download & Preparation

**Step 1: Download CORAL Dataset**

```python
# src/data_processing/download_coral.py

from datasets import load_dataset
import json
import os

def download_coral_datasets(output_dir="data/coral"):
    """Download CORAL datasets from HuggingFace."""
    os.makedirs(output_dir, exist_ok=True)

    datasets_to_download = ["pearl", "inspired", "redial"]

    for dataset_name in datasets_to_download:
        print(f"Downloading {dataset_name}...")
        dataset = load_dataset("kookeej/CORAL", dataset_name)

        # Save to local files
        for split in dataset.keys():
            output_path = f"{output_dir}/{dataset_name}/{split}.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w') as f:
                for item in dataset[split]:
                    f.write(json.dumps(item) + '\n')

        print(f"✓ {dataset_name} saved to {output_dir}/{dataset_name}/")

if __name__ == "__main__":
    download_coral_datasets()
```

**Step 2: Data Preprocessing**

```python
# src/data_processing/preprocess_coral.py

import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    turn_id: int
    user_utterance: str
    system_response: str
    items_mentioned: List[str]
    items_recommended: List[str]
    user_feedback: str  # 'positive', 'negative', 'neutral'

@dataclass
class PreferenceData:
    """User preference information."""
    liked_items: List[str]
    disliked_items: List[str]
    preference_features: Dict[str, Any]

@dataclass
class ConversationSample:
    """Complete conversation sample."""
    conversation_id: str
    user_id: str
    turns: List[ConversationTurn]
    user_preferences: PreferenceData
    final_recommendation: str
    success: bool
    domain: str  # 'pearl', 'inspired', 'redial'

class CORALPreprocessor:
    """Preprocess CORAL data for training."""

    def __init__(self, data_dir="data/coral"):
        self.data_dir = data_dir

    def load_dataset(self, dataset_name: str, split: str) -> List[ConversationSample]:
        """Load and parse a CORAL dataset."""
        file_path = f"{self.data_dir}/{dataset_name}/{split}.jsonl"
        samples = []

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                sample = self._parse_sample(data, dataset_name)
                samples.append(sample)

        return samples

    def _parse_sample(self, data: Dict, domain: str) -> ConversationSample:
        """Parse a single data sample."""
        # This will be customized based on actual CORAL format
        return ConversationSample(
            conversation_id=data.get('conversation_id', ''),
            user_id=data.get('user_id', ''),
            turns=self._parse_turns(data.get('turns', [])),
            user_preferences=self._parse_preferences(data.get('user_preferences', {})),
            final_recommendation=data.get('final_recommendation', ''),
            success=data.get('success', False),
            domain=domain
        )

    def create_preference_pairs(
        self,
        samples: List[ConversationSample]
    ) -> List[Dict[str, Any]]:
        """
        Create preference pairs for DPO training.

        For each conversation, create (chosen, rejected) pairs based on:
        - User feedback (positive vs. negative)
        - Successful vs. failed recommendations
        - Aligned vs. misaligned responses
        """
        preference_pairs = []

        for sample in samples:
            for i, turn in enumerate(sample.turns):
                # Create context from previous turns
                context = self._create_context(sample.turns[:i], sample.user_preferences)

                # Create preference pair
                if turn.user_feedback == 'positive':
                    pair = {
                        'context': context,
                        'query': turn.user_utterance,
                        'chosen': turn.system_response,
                        'rejected': self._generate_negative_sample(turn, sample),
                        'preference_type': 'explicit_positive'
                    }
                    preference_pairs.append(pair)
                elif turn.user_feedback == 'negative':
                    pair = {
                        'context': context,
                        'query': turn.user_utterance,
                        'chosen': self._generate_positive_sample(turn, sample),
                        'rejected': turn.system_response,
                        'preference_type': 'explicit_negative'
                    }
                    preference_pairs.append(pair)

        return preference_pairs

    def _create_context(
        self,
        previous_turns: List[ConversationTurn],
        preferences: PreferenceData
    ) -> str:
        """Create context string from conversation history and preferences."""
        context_parts = []

        # Add preference information
        if preferences.liked_items:
            context_parts.append(f"User likes: {', '.join(preferences.liked_items[:3])}")
        if preferences.disliked_items:
            context_parts.append(f"User dislikes: {', '.join(preferences.disliked_items[:3])}")

        # Add conversation history
        for turn in previous_turns[-3:]:  # Last 3 turns
            context_parts.append(f"User: {turn.user_utterance}")
            context_parts.append(f"Assistant: {turn.system_response}")

        return "\n".join(context_parts)
```

### 2.2 Adaptation to Bollywood Domain

Since CORAL is general-purpose and our system is Bollywood-focused, we'll:

1. **Domain Transfer**: Map CORAL patterns to Bollywood context
2. **Synthetic Data**: Generate Bollywood-specific preference pairs
3. **Hybrid Training**: Mix CORAL data with domain-specific data

```python
# src/data_processing/bollywood_adapter.py

class BollywoodDomainAdapter:
    """Adapt CORAL data to Bollywood domain."""

    def __init__(self):
        self.genre_mapping = {
            'action': 'bollywood_action',
            'romance': 'bollywood_romance',
            'drama': 'bollywood_drama',
            # ... more mappings
        }

    def adapt_sample(self, coral_sample: ConversationSample) -> ConversationSample:
        """Adapt a CORAL sample to Bollywood domain."""
        # Replace movie names with Bollywood equivalents
        # Adapt preferences to Bollywood context
        # Maintain conversation structure
        pass

    def generate_bollywood_preferences(self) -> List[Dict[str, Any]]:
        """Generate synthetic Bollywood preference data."""
        # Use existing Bollywood knowledge graph
        # Create preference pairs from KG
        pass
```

---

## 3. Architecture Enhancements

### 3.1 Current Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER QUERY                           │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Block 1: Context Fusion    │
          │  (Text Encoder)             │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Block 2: Planning          │
          │  (Tree of Thoughts + LLM)   │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Block 3: Execution         │
          │  (Tool Execution)           │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Block 4: RL Rewards        │
          │  (Reward Calculation)       │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     FINAL ANSWER            │
          └─────────────────────────────┘
```

### 3.2 Enhanced Architecture with Preference Learning

```
┌───────────────────────────────────────────────────────────────┐
│                    USER QUERY + PREFERENCES                   │
└────────────────────────────┬──────────────────────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │  Block 0: Preference Encoding         │
         │  - Encode user preferences            │
         │  - Retrieve preference history        │
         │  - Contrast positive/negative prefs   │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │  Block 1: Context Fusion (ENHANCED)   │
         │  - Fuse query + preferences           │
         │  - Multi-modal embeddings             │
         │  - Preference-aware encoding          │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │  Block 2: Planning (DPO-TRAINED)      │
         │  - Preference-guided planning         │
         │  - Tree of Thoughts with preferences  │
         │  - DPO-optimized policy LLM           │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │  Block 3: Execution (ENHANCED)        │
         │  - Preference-aware tool selection    │
         │  - Personalized tool parameters       │
         │  - Feedback collection                │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │  Block 4: RL Rewards (PPO-TRAINED)    │
         │  - Preference alignment reward        │
         │  - RLAIF with preference model        │
         │  - Multi-objective reward             │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │  Block 5: Response Generation         │
         │  - Preference-aligned synthesis       │
         │  - Personalized response              │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │     PERSONALIZED ANSWER                │
         └───────────────────────────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │  Feedback Loop                        │
         │  - Collect user feedback              │
         │  - Update preference model            │
         │  - Store for future training          │
         └───────────────────────────────────────┘
```

### 3.3 New Components

#### **Block 0: Preference Encoding**

```python
# src/block0_preference_encoding/preference_encoder.py

from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class UserPreference:
    """User preference structure."""
    liked_items: List[str]
    disliked_items: List[str]
    preference_features: Dict[str, Any]
    preference_embedding: np.ndarray = None

class PreferenceEncoder:
    """Encode user preferences into embeddings."""

    def __init__(self, config):
        self.config = config
        self.embedding_model = self._load_embedding_model()

    def encode_preferences(
        self,
        preferences: UserPreference,
        query: str
    ) -> np.ndarray:
        """
        Encode user preferences into a dense vector.

        Args:
            preferences: User preference data
            query: Current query

        Returns:
            Preference embedding vector
        """
        # Create preference text
        pref_text = self._create_preference_text(preferences)

        # Encode with contrastive learning
        positive_emb = self._encode_text(pref_text['positive'])
        negative_emb = self._encode_text(pref_text['negative'])
        query_emb = self._encode_text(query)

        # Combine with attention
        combined_emb = self._attend(query_emb, positive_emb, negative_emb)

        return combined_emb

    def _create_preference_text(self, preferences: UserPreference) -> Dict[str, str]:
        """Create text representations of preferences."""
        positive_text = "User likes: " + ", ".join(preferences.liked_items[:5])
        negative_text = "User dislikes: " + ", ".join(preferences.disliked_items[:5])

        return {
            'positive': positive_text,
            'negative': negative_text
        }

    def retrieve_user_history(self, user_id: str) -> UserPreference:
        """Retrieve user preference history from database."""
        # Query preference database
        # Return aggregated preferences
        pass
```

#### **Enhanced Block 1: Preference-Aware Context Fusion**

```python
# src/block1_context_fusion/preference_aware_encoder.py

class PreferenceAwareContextEncoder:
    """Context encoder that incorporates user preferences."""

    def __init__(self, config):
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.preference_encoder = PreferenceEncoder(config)

    def encode(
        self,
        query: str,
        context: Dict,
        preferences: UserPreference = None
    ) -> np.ndarray:
        """
        Encode query and context with preference awareness.

        Args:
            query: User query
            context: Additional context
            preferences: User preferences (optional)

        Returns:
            Fused embedding
        """
        # Encode query
        query_emb = self.text_encoder.encode_query(query)

        # Encode context
        context_emb = self.text_encoder.encode_context(context)

        # Encode preferences if available
        if preferences:
            pref_emb = self.preference_encoder.encode_preferences(
                preferences, query
            )
            # Fuse all three
            fused_emb = self._fuse_with_preferences(
                query_emb, context_emb, pref_emb
            )
        else:
            fused_emb = self._fuse_without_preferences(query_emb, context_emb)

        return fused_emb

    def _fuse_with_preferences(
        self,
        query_emb: np.ndarray,
        context_emb: np.ndarray,
        pref_emb: np.ndarray
    ) -> np.ndarray:
        """Fuse query, context, and preferences using attention."""
        # Multi-head attention
        # Weighted combination
        # Return fused embedding
        pass
```

---

## 4. Preference Learning Framework

### 4.1 Direct Preference Optimization (DPO)

DPO will be used to train the **planning policy** (Block 2) to generate plans that align with user preferences.

**DPO Loss Function:**

```
L_DPO(π_θ) = -E[(x,y_w,y_l)~D] [
    log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)))
]
```

Where:
- `π_θ`: Policy model being trained (PolicyLLM)
- `π_ref`: Reference model (frozen copy)
- `y_w`: Preferred/winning response
- `y_l`: Rejected/losing response
- `β`: KL penalty coefficient (0.1 from config)

**Implementation:**

```python
# src/training/dpo_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any

class DPOTrainer:
    """Trainer for Direct Preference Optimization."""

    def __init__(self, config):
        self.config = config.block2.dpo
        self.beta = self.config.beta

        # Load policy model (to be trained)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            "gpt-4o-mini"  # Or your policy model
        )

        # Load reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            "gpt-4o-mini"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained("gpt-4o-mini")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate
        )

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs from policy for chosen responses
            policy_rejected_logps: Log probs from policy for rejected responses
            ref_chosen_logps: Log probs from reference for chosen responses
            ref_rejected_logps: Log probs from reference for rejected responses

        Returns:
            DPO loss
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # DPO loss
        loss = -torch.nn.functional.logsigmoid(
            self.beta * (chosen_logratios - rejected_logratios)
        ).mean()

        return loss

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step."""
        self.policy_model.train()

        # Extract batch data
        contexts = batch['context']
        queries = batch['query']
        chosen_responses = batch['chosen']
        rejected_responses = batch['rejected']

        # Tokenize
        chosen_inputs = self.tokenizer(
            [f"{ctx}\n{q}\n{r}" for ctx, q, r in zip(contexts, queries, chosen_responses)],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        rejected_inputs = self.tokenizer(
            [f"{ctx}\n{q}\n{r}" for ctx, q, r in zip(contexts, queries, rejected_responses)],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Get log probabilities from policy
        with torch.no_grad():
            ref_chosen_logps = self._get_log_probs(self.ref_model, chosen_inputs)
            ref_rejected_logps = self._get_log_probs(self.ref_model, rejected_inputs)

        policy_chosen_logps = self._get_log_probs(self.policy_model, chosen_inputs)
        policy_rejected_logps = self._get_log_probs(self.policy_model, rejected_inputs)

        # Compute loss
        loss = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    def _get_log_probs(self, model, inputs):
        """Get log probabilities for inputs."""
        outputs = model(**inputs)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        # (implementation details...)

        return log_probs.sum(dim=-1)

    def train(
        self,
        train_dataset: List[Dict],
        eval_dataset: List[Dict] = None,
        num_epochs: int = None
    ):
        """Full training loop."""
        num_epochs = num_epochs or self.config.num_epochs

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Evaluation
            if eval_dataset and epoch % self.config.eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataset)
                print(f"Eval metrics: {eval_metrics}")

            # Save checkpoint
            if epoch % self.config.save_steps == 0:
                self.save_checkpoint(f"dpo_epoch_{epoch}")

    def evaluate(self, eval_dataset: List[Dict]) -> Dict[str, float]:
        """Evaluate the model."""
        # Compute evaluation metrics
        # Return metrics dict
        pass

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        torch.save(
            self.policy_model.state_dict(),
            f"checkpoints/{name}.pt"
        )
```

### 4.2 Proximal Policy Optimization (PPO)

PPO will be used to train the **reward model and execution strategy** (Block 4).

**PPO Algorithm:**

```
1. Collect trajectories using current policy
2. Compute advantages using GAE
3. Optimize policy with clipped objective:
   L_CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
4. Update value function
```

**Implementation:**

```python
# src/training/ppo_trainer.py

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple

class PPOTrainer:
    """Trainer for Proximal Policy Optimization."""

    def __init__(self, config):
        self.config = config.block4.ppo

        # Hyperparameters
        self.gamma = self.config.gamma
        self.gae_lambda = self.config.gae_lambda
        self.clip_epsilon = self.config.clip_epsilon
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef

        # Models (simplified - in practice would be more complex)
        self.policy_model = None  # Your agent's planning policy
        self.value_model = None   # Value function estimator

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_model.parameters()),
            lr=self.config.learning_rate
        )

    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of episode termination flags

        Returns:
            Advantages and returns
        """
        advantages = []
        returns = []

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PPO loss.

        Args:
            log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from old policy
            advantages: Computed advantages
            values: Value estimates
            returns: Computed returns
            entropy: Policy entropy

        Returns:
            Total PPO loss
        """
        # Ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        return total_loss, policy_loss, value_loss, entropy_loss

    def train_step(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Single PPO training step.

        Args:
            trajectories: List of collected trajectories

        Returns:
            Training metrics
        """
        # Extract data from trajectories
        states = [t['state'] for t in trajectories]
        actions = [t['action'] for t in trajectories]
        rewards = [t['reward'] for t in trajectories]
        old_log_probs = [t['log_prob'] for t in trajectories]
        dones = [t['done'] for t in trajectories]

        # Get values
        with torch.no_grad():
            values = [self.value_model(s).item() for s in states]

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones)

        # PPO update epochs
        metrics = {}
        for epoch in range(self.config.num_epochs_per_iteration):
            # Forward pass
            new_log_probs = []
            new_values = []
            entropies = []

            for state, action in zip(states, actions):
                # Get new log probs and values
                log_prob = self.policy_model.get_log_prob(state, action)
                value = self.value_model(state)
                entropy = self.policy_model.get_entropy(state)

                new_log_probs.append(log_prob)
                new_values.append(value)
                entropies.append(entropy)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            entropies = torch.stack(entropies)
            old_log_probs_tensor = torch.tensor(old_log_probs)

            # Compute loss
            total_loss, policy_loss, value_loss, entropy_loss = self.compute_ppo_loss(
                new_log_probs,
                old_log_probs_tensor,
                advantages,
                new_values,
                returns,
                entropies
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_model.parameters()) + list(self.value_model.parameters()),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            # Store metrics
            metrics = {
                'total_loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item()
            }

        return metrics

    def collect_trajectories(
        self,
        num_trajectories: int,
        env: Any
    ) -> List[Dict[str, Any]]:
        """Collect trajectories using current policy."""
        # Implementation for trajectory collection
        # In the agent context, this means running the full pipeline
        # and collecting intermediate states, actions, rewards
        pass
```

---

## 5. Training Pipeline

### 5.1 Data Preparation Pipeline

```python
# src/training/data_pipeline.py

from typing import List, Dict, Any
from src.data_processing.preprocess_coral import CORALPreprocessor
from src.data_processing.bollywood_adapter import BollywoodDomainAdapter

class TrainingDataPipeline:
    """Complete pipeline for preparing training data."""

    def __init__(self, config):
        self.config = config
        self.coral_preprocessor = CORALPreprocessor()
        self.bollywood_adapter = BollywoodDomainAdapter()

    def prepare_dpo_data(self) -> Dict[str, List[Dict]]:
        """
        Prepare data for DPO training.

        Returns:
            Dictionary with 'train' and 'eval' datasets
        """
        # 1. Load CORAL datasets
        coral_data = []
        for dataset_name in ['pearl', 'inspired', 'redial']:
            train_samples = self.coral_preprocessor.load_dataset(dataset_name, 'train')
            coral_data.extend(train_samples)

        # 2. Create preference pairs
        preference_pairs = self.coral_preprocessor.create_preference_pairs(coral_data)

        # 3. Adapt to Bollywood domain
        bollywood_pairs = self.bollywood_adapter.adapt_preference_pairs(preference_pairs)

        # 4. Add synthetic Bollywood data
        synthetic_pairs = self.bollywood_adapter.generate_bollywood_preferences()

        # 5. Combine and split
        all_pairs = preference_pairs + bollywood_pairs + synthetic_pairs
        train_size = int(0.9 * len(all_pairs))

        return {
            'train': all_pairs[:train_size],
            'eval': all_pairs[train_size:]
        }

    def prepare_ppo_data(self) -> Dict[str, Any]:
        """
        Prepare environment and data for PPO training.

        Returns:
            Training environment and configuration
        """
        # Create RL environment for agent training
        # Define state space, action space, reward function
        pass

    def prepare_evaluation_data(self) -> List[Dict]:
        """
        Prepare held-out evaluation data.

        Returns:
            Evaluation dataset with ground truth
        """
        # Load test sets from CORAL
        # Add Bollywood-specific test queries
        # Format for evaluation
        pass
```

### 5.2 Complete Training Script

```python
# scripts/train_preference_model.py

import argparse
from src.utils.config import load_config
from src.training.data_pipeline import TrainingDataPipeline
from src.training.dpo_trainer import DPOTrainer
from src.training.ppo_trainer import PPOTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--stage', type=str, choices=['dpo', 'ppo', 'both'], default='both')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Prepare data
    print("Preparing training data...")
    data_pipeline = TrainingDataPipeline(config)

    # Stage 1: DPO Training
    if args.stage in ['dpo', 'both']:
        print("\n=== Stage 1: DPO Training ===")
        dpo_data = data_pipeline.prepare_dpo_data()

        dpo_trainer = DPOTrainer(config)
        dpo_trainer.train(
            train_dataset=dpo_data['train'],
            eval_dataset=dpo_data['eval']
        )

        print("✓ DPO training completed")

    # Stage 2: PPO Training
    if args.stage in ['ppo', 'both']:
        print("\n=== Stage 2: PPO Training ===")
        ppo_env = data_pipeline.prepare_ppo_data()

        ppo_trainer = PPOTrainer(config)
        ppo_trainer.train(env=ppo_env)

        print("✓ PPO training completed")

    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
```

---

## 6. Evaluation Metrics

### 6.1 Metric Categories

We'll evaluate the system across multiple dimensions:

#### **A. Recommendation Quality**
- **Hit Rate @K**: Percentage of correct recommendations in top K
- **NDCG @K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Precision @K**: Precision at K
- **Recall @K**: Recall at K

#### **B. Preference Alignment**
- **Preference Accuracy**: How well responses align with stated preferences
- **Contrasting Preference Score**: Avoidance of disliked items
- **Personalization Score**: Difference from generic recommendations

#### **C. Conversational Quality**
- **Response Coherence**: Using perplexity and ROUGE scores
- **Contextual Relevance**: How well responses consider conversation history
- **Fluency**: Language quality metrics

#### **D. Efficiency**
- **Planning Time**: Time to generate plans
- **Tool Selection Accuracy**: Correct tool usage rate
- **Execution Success Rate**: Successful tool executions

#### **E. User Satisfaction** (if interactive evaluation)
- **User Rating**: Direct feedback scores
- **Task Success Rate**: Percentage of successful query resolutions
- **Conversation Turns**: Efficiency in reaching solution

### 6.2 Evaluation Implementation

```python
# src/evaluation/metrics.py

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import ndcg_score

class EvaluationMetrics:
    """Comprehensive evaluation metrics for the agent system."""

    def __init__(self):
        pass

    # Recommendation Metrics
    def hit_rate_at_k(
        self,
        predictions: List[List[str]],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """
        Compute Hit Rate @K.

        Args:
            predictions: List of predicted item lists (ranked)
            ground_truth: List of ground truth items
            k: Top K to consider

        Returns:
            Hit rate @K
        """
        hits = 0
        for pred, truth in zip(predictions, ground_truth):
            if truth in pred[:k]:
                hits += 1

        return hits / len(predictions)

    def ndcg_at_k(
        self,
        predictions: List[List[str]],
        ground_truth: List[str],
        k: int = 10
    ) -> float:
        """Compute NDCG @K."""
        scores = []

        for pred, truth in zip(predictions, ground_truth):
            # Create relevance scores
            relevance = [1 if item == truth else 0 for item in pred[:k]]

            if sum(relevance) == 0:
                scores.append(0)
            else:
                ideal_relevance = sorted(relevance, reverse=True)
                ndcg = ndcg_score([ideal_relevance], [relevance])
                scores.append(ndcg)

        return np.mean(scores)

    def mrr(
        self,
        predictions: List[List[str]],
        ground_truth: List[str]
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        reciprocal_ranks = []

        for pred, truth in zip(predictions, ground_truth):
            try:
                rank = pred.index(truth) + 1
                reciprocal_ranks.append(1 / rank)
            except ValueError:
                reciprocal_ranks.append(0)

        return np.mean(reciprocal_ranks)

    # Preference Alignment Metrics
    def preference_accuracy(
        self,
        recommendations: List[str],
        liked_items: List[List[str]],
        disliked_items: List[List[str]]
    ) -> float:
        """
        Compute preference alignment accuracy.

        Measures how well recommendations align with likes
        and avoid dislikes.
        """
        scores = []

        for rec, liked, disliked in zip(recommendations, liked_items, disliked_items):
            # Positive score if rec is similar to liked
            # Negative score if rec is similar to disliked
            score = self._compute_preference_similarity(rec, liked, disliked)
            scores.append(score)

        return np.mean(scores)

    def contrasting_preference_score(
        self,
        recommendations: List[str],
        disliked_items: List[List[str]]
    ) -> float:
        """
        Measure how well system avoids disliked items.

        Returns percentage of recommendations that don't match dislikes.
        """
        avoid_rate = 0

        for rec, disliked in zip(recommendations, disliked_items):
            if rec not in disliked:
                avoid_rate += 1

        return avoid_rate / len(recommendations)

    # Conversational Quality Metrics
    def response_coherence(
        self,
        responses: List[str],
        contexts: List[str]
    ) -> float:
        """
        Measure response coherence with context.

        Uses perplexity and semantic similarity.
        """
        # Implementation using language models
        pass

    def contextual_relevance(
        self,
        responses: List[str],
        conversation_histories: List[List[str]]
    ) -> float:
        """
        Measure how well responses consider conversation history.
        """
        # Implementation using semantic similarity
        pass

    # Efficiency Metrics
    def tool_selection_accuracy(
        self,
        predicted_tools: List[List[str]],
        optimal_tools: List[List[str]]
    ) -> float:
        """Measure accuracy of tool selection."""
        correct = 0

        for pred, opt in zip(predicted_tools, optimal_tools):
            if set(pred) == set(opt):
                correct += 1

        return correct / len(predicted_tools)

    def execution_success_rate(
        self,
        execution_results: List[Dict]
    ) -> float:
        """Measure rate of successful tool executions."""
        successful = sum(1 for r in execution_results if r['status'] == 'success')
        return successful / len(execution_results)

    # Comprehensive Evaluation
    def evaluate_all(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth data

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Recommendation metrics
        metrics['hit_rate@5'] = self.hit_rate_at_k(
            predictions['recommendations'],
            ground_truth['true_items'],
            k=5
        )
        metrics['hit_rate@10'] = self.hit_rate_at_k(
            predictions['recommendations'],
            ground_truth['true_items'],
            k=10
        )
        metrics['ndcg@5'] = self.ndcg_at_k(
            predictions['recommendations'],
            ground_truth['true_items'],
            k=5
        )
        metrics['ndcg@10'] = self.ndcg_at_k(
            predictions['recommendations'],
            ground_truth['true_items'],
            k=10
        )
        metrics['mrr'] = self.mrr(
            predictions['recommendations'],
            ground_truth['true_items']
        )

        # Preference alignment
        metrics['preference_accuracy'] = self.preference_accuracy(
            predictions['recommendations'],
            ground_truth['liked_items'],
            ground_truth['disliked_items']
        )
        metrics['contrasting_score'] = self.contrasting_preference_score(
            predictions['recommendations'],
            ground_truth['disliked_items']
        )

        # Efficiency
        metrics['tool_accuracy'] = self.tool_selection_accuracy(
            predictions['tools_used'],
            ground_truth['optimal_tools']
        )
        metrics['execution_success'] = self.execution_success_rate(
            predictions['execution_results']
        )

        return metrics
```

### 6.3 Benchmark Comparison

```python
# src/evaluation/benchmark.py

class BenchmarkComparison:
    """Compare against baselines and other systems."""

    def __init__(self):
        self.baselines = {
            'random': self._random_baseline,
            'popularity': self._popularity_baseline,
            'no_preference': self._no_preference_baseline,
            'dpo_only': self._dpo_only_baseline,
            'ppo_only': self._ppo_only_baseline
        }

    def run_comparison(
        self,
        test_data: List[Dict],
        our_system: Any
    ) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive comparison.

        Returns:
            Results for each system including baselines
        """
        results = {}

        # Evaluate our system
        results['our_system'] = self._evaluate_system(our_system, test_data)

        # Evaluate baselines
        for name, baseline_fn in self.baselines.items():
            baseline_system = baseline_fn()
            results[name] = self._evaluate_system(baseline_system, test_data)

        return results

    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate markdown comparison report."""
        report = "# Benchmark Comparison Results\n\n"

        report += "## Recommendation Quality\n\n"
        report += "| System | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | MRR |\n"
        report += "|--------|-------|--------|--------|---------|-----|\n"

        for system, metrics in results.items():
            report += f"| {system} | "
            report += f"{metrics.get('hit_rate@5', 0):.4f} | "
            report += f"{metrics.get('hit_rate@10', 0):.4f} | "
            report += f"{metrics.get('ndcg@5', 0):.4f} | "
            report += f"{metrics.get('ndcg@10', 0):.4f} | "
            report += f"{metrics.get('mrr', 0):.4f} |\n"

        report += "\n## Preference Alignment\n\n"
        report += "| System | Preference Accuracy | Contrasting Score |\n"
        report += "|--------|--------------------|-----------------|\n"

        for system, metrics in results.items():
            report += f"| {system} | "
            report += f"{metrics.get('preference_accuracy', 0):.4f} | "
            report += f"{metrics.get('contrasting_score', 0):.4f} |\n"

        report += "\n## Efficiency\n\n"
        report += "| System | Tool Accuracy | Execution Success |\n"
        report += "|--------|--------------|------------------|\n"

        for system, metrics in results.items():
            report += f"| {system} | "
            report += f"{metrics.get('tool_accuracy', 0):.4f} | "
            report += f"{metrics.get('execution_success', 0):.4f} |\n"

        return report
```

---

## 7. Implementation Roadmap

### Phase 1: Data Preparation (Week 1-2)

- [ ] Download CORAL datasets from HuggingFace
- [ ] Analyze actual data structure
- [ ] Implement data preprocessing pipeline
- [ ] Create preference pairs for DPO
- [ ] Adapt data to Bollywood domain
- [ ] Generate synthetic preference data
- [ ] Split into train/eval/test sets

### Phase 2: Architecture Enhancement (Week 3-4)

- [ ] Implement Block 0: Preference Encoding
  - Preference encoder
  - Preference database/storage
  - Retrieval mechanisms
- [ ] Enhance Block 1: Context Fusion
  - Preference-aware encoding
  - Multi-modal fusion
- [ ] Enhance Block 3: Execution
  - Preference-aware tool selection
  - Feedback collection
- [ ] Add Block 5: Response Generation
  - Preference-aligned synthesis
  - Personalization layer

### Phase 3: DPO Training (Week 5-6)

- [ ] Implement DPO trainer
- [ ] Prepare policy model for training
- [ ] Train on CORAL preference pairs
- [ ] Evaluate on validation set
- [ ] Fine-tune hyperparameters
- [ ] Save best model

### Phase 4: PPO Training (Week 7-8)

- [ ] Implement PPO trainer
- [ ] Define RL environment
- [ ] Implement reward model
- [ ] Train value function
- [ ] Train policy with PPO
- [ ] Evaluate and iterate

### Phase 5: Evaluation & Metrics (Week 9-10)

- [ ] Implement all evaluation metrics
- [ ] Create test datasets
- [ ] Run comprehensive evaluation
- [ ] Compare against baselines
- [ ] Generate benchmark reports
- [ ] Analyze results

### Phase 6: Iteration & Refinement (Week 11-12)

- [ ] Analyze failure cases
- [ ] Improve weak areas
- [ ] Ablation studies
- [ ] Optimize performance
- [ ] Final evaluation
- [ ] Documentation

---

## 8. Expected Outcomes

### 8.1 Performance Targets

Based on CORAL paper and similar systems, we target:

**Recommendation Quality:**
- Hit Rate @5: > 0.60
- Hit Rate @10: > 0.75
- NDCG @5: > 0.55
- NDCG @10: > 0.65
- MRR: > 0.50

**Preference Alignment:**
- Preference Accuracy: > 0.70
- Contrasting Score: > 0.85 (high avoidance of dislikes)

**Efficiency:**
- Tool Selection Accuracy: > 0.80
- Execution Success Rate: > 0.90
- Average Planning Time: < 2 seconds

### 8.2 Comparison with Baselines

Expected improvements over baselines:

| Metric | Random | Popularity | No Pref | Our System |
|--------|--------|-----------|---------|------------|
| Hit@10 | 0.10 | 0.45 | 0.60 | **0.75** |
| NDCG@10 | 0.05 | 0.35 | 0.50 | **0.65** |
| Pref Acc | 0.20 | 0.30 | 0.40 | **0.70** |

### 8.3 Ablation Studies

We'll conduct ablation studies to understand component contributions:

1. **No Preferences**: Remove preference encoding
2. **DPO Only**: Train only with DPO, no PPO
3. **PPO Only**: Train only with PPO, no DPO
4. **No Tree of Thoughts**: Use simple planning
5. **Single Tool**: Restrict to one tool type

### 8.4 Visualizations

Create visualizations for:
- Training curves (loss over time)
- Metric improvements across training
- Preference space embeddings (t-SNE)
- Tool selection patterns
- Response quality heatmaps

---

## 9. File Structure

```
preference_guided_agent_system/
├── src/
│   ├── block0_preference_encoding/
│   │   ├── __init__.py
│   │   ├── preference_encoder.py
│   │   └── preference_storage.py
│   │
│   ├── block1_context_fusion/
│   │   ├── __init__.py
│   │   ├── text_encoder.py
│   │   └── preference_aware_encoder.py
│   │
│   ├── block5_response_generation/
│   │   ├── __init__.py
│   │   └── personalized_synthesis.py
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── download_coral.py
│   │   ├── preprocess_coral.py
│   │   └── bollywood_adapter.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py
│   │   ├── dpo_trainer.py
│   │   └── ppo_trainer.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── benchmark.py
│
├── data/
│   ├── coral/
│   │   ├── pearl/
│   │   ├── inspired/
│   │   └── redial/
│   └── processed/
│       ├── dpo_train.jsonl
│       ├── dpo_eval.jsonl
│       └── test.jsonl
│
├── scripts/
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── train_preference_model.py
│   └── evaluate.py
│
├── docs/
│   ├── CORAL_INTEGRATION_PLAN.md
│   └── EVALUATION_RESULTS.md
│
└── notebooks/
    ├── data_exploration.ipynb
    ├── preference_analysis.ipynb
    └── results_visualization.ipynb
```

---

## 10. Next Steps

### Immediate Actions

1. **Download CORAL Dataset**
   ```bash
   python scripts/download_data.py
   ```

2. **Explore Data**
   ```bash
   jupyter notebook notebooks/data_exploration.ipynb
   ```

3. **Implement Preprocessing**
   - Complete `preprocess_coral.py`
   - Adapt to Bollywood domain

4. **Set Up Training Infrastructure**
   - Implement DPO trainer
   - Prepare training data

### Questions to Resolve

1. **CORAL Data Format**: Need to inspect actual data structure once downloaded
2. **Compute Resources**: Determine GPU availability for training
3. **Model Choice**: Decide on specific model for DPO training (GPT-4o-mini or other)
4. **Evaluation Strategy**: Decide on human evaluation vs. automated only
5. **Deployment**: Plan for production deployment after training

---

## Conclusion

This comprehensive plan provides a roadmap for integrating CORAL dataset and enhancing the preference-guided agent system with DPO and PPO training. The key innovations are:

1. **Preference Encoding Layer**: Explicitly model user preferences
2. **DPO Training**: Align planning policy with preference data
3. **PPO Training**: Optimize execution strategy with RL
4. **Comprehensive Metrics**: Multi-dimensional evaluation
5. **Bollywood Adaptation**: Domain-specific enhancements

The expected outcome is a highly personalized agent system that:
- Understands and respects user preferences
- Avoids disliked recommendations
- Generates contextually relevant responses
- Optimizes tool usage for efficiency
- Achieves state-of-the-art performance on recommendation metrics

**Timeline**: 12 weeks
**Success Criteria**: > 15% improvement over no-preference baseline on key metrics
