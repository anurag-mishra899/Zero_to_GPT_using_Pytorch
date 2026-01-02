"""
Module 15: LLM Alignment - RLHF, DPO, and Beyond

This module covers:
1. Reward Model training
2. PPO for LLMs
3. DPO (Direct Preference Optimization)
4. KTO and other methods
5. Complete training examples
"""

import math
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Section 1: Data Structures
# ============================================================================

@dataclass
class PreferenceData:
    """Single preference comparison."""
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Rejected response


@dataclass
class RLHFBatch:
    """Batch for RLHF training."""
    prompts: torch.Tensor  # (batch, seq_len)
    responses: torch.Tensor  # (batch, response_len)
    rewards: torch.Tensor  # (batch,)
    old_logprobs: torch.Tensor  # (batch, response_len)
    values: torch.Tensor  # (batch,)
    advantages: torch.Tensor  # (batch,)


class PreferenceDataset(Dataset):
    """Dataset for preference-based training (DPO, reward modeling)."""

    def __init__(
        self,
        data: List[PreferenceData],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Tokenize prompt + chosen
        chosen_input = f"{item.prompt}{item.chosen}"
        chosen_tokens = self.tokenizer(
            chosen_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize prompt + rejected
        rejected_input = f"{item.prompt}{item.rejected}"
        rejected_tokens = self.tokenizer(
            rejected_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get prompt length for masking
        prompt_tokens = self.tokenizer(
            item.prompt,
            return_tensors='pt'
        )
        prompt_length = prompt_tokens['input_ids'].shape[1]

        return {
            'chosen_ids': chosen_tokens['input_ids'].squeeze(0),
            'chosen_mask': chosen_tokens['attention_mask'].squeeze(0),
            'rejected_ids': rejected_tokens['input_ids'].squeeze(0),
            'rejected_mask': rejected_tokens['attention_mask'].squeeze(0),
            'prompt_length': prompt_length
        }


# ============================================================================
# Section 2: Reward Model
# ============================================================================

class RewardModel(nn.Module):
    """
    Reward model for RLHF.

    Architecture:
    - LLM backbone (frozen or trainable)
    - Scalar reward head

    Training:
    - Bradley-Terry loss on preference pairs
    - L = -log σ(R_chosen - R_rejected)
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        # Reward head: hidden_size -> 1
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward for input sequence.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            rewards: (batch,) scalar reward for each sequence
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        # Get last hidden state
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs

        # Pool: use last non-padded token
        if attention_mask is not None:
            # Find last token position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_indices, seq_lengths]
        else:
            # Use last token
            pooled = hidden[:, -1, :]

        # Get scalar reward
        reward = self.reward_head(pooled).squeeze(-1)
        return reward


class RewardModelTrainer:
    """Trainer for reward model using Bradley-Terry loss."""

    def __init__(
        self,
        model: RewardModel,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Bradley-Terry loss.

        L = -log σ(R_chosen - R_rejected)
        """
        # Get rewards
        chosen_reward = self.model(chosen_ids, chosen_mask)
        rejected_reward = self.model(rejected_ids, rejected_mask)

        # Bradley-Terry loss
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        # Metrics
        accuracy = (chosen_reward > rejected_reward).float().mean()
        reward_diff = (chosen_reward - rejected_reward).mean()

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward': chosen_reward.mean().item(),
            'rejected_reward': rejected_reward.mean().item(),
            'reward_diff': reward_diff.item()
        }

        return loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move to device
        chosen_ids = batch['chosen_ids'].to(self.device)
        chosen_mask = batch['chosen_mask'].to(self.device)
        rejected_ids = batch['rejected_ids'].to(self.device)
        rejected_mask = batch['rejected_mask'].to(self.device)

        # Forward
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask
        )

        # Backward
        loss.backward()
        self.optimizer.step()

        return metrics


# ============================================================================
# Section 3: DPO (Direct Preference Optimization)
# ============================================================================

class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    DPO Loss:
    L = -log σ(β × (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

    Where:
    - y_w = chosen response
    - y_l = rejected response
    - π = policy model
    - π_ref = reference model (frozen SFT)
    - β = temperature parameter
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
        device: str = 'cpu'
    ):
        self.policy = policy_model.to(device)
        self.ref = ref_model.to(device)
        self.optimizer = optimizer
        self.beta = beta
        self.device = device

        # Freeze reference model
        for param in self.ref.parameters():
            param.requires_grad = False

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """
        Get log probabilities for response tokens.

        Args:
            model: Language model
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            prompt_length: Length of prompt (don't compute loss on prompt)

        Returns:
            log_probs: (batch,) summed log prob for each response
        """
        # Get model outputs
        outputs = model(input_ids, attention_mask=attention_mask)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        # Shift for autoregressive: predict next token
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out prompt tokens (only compute on response)
        response_mask = torch.zeros_like(shift_mask)
        response_mask[:, prompt_length-1:] = shift_mask[:, prompt_length-1:]

        # Sum log probs for response
        masked_log_probs = token_log_probs * response_mask
        total_log_prob = masked_log_probs.sum(dim=1)

        return total_log_prob

    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        prompt_length: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.

        L = -log σ(β × (log_ratio_chosen - log_ratio_rejected))
        where log_ratio = log(π/π_ref)
        """
        # Policy log probs
        pi_chosen = self.get_log_probs(
            self.policy, chosen_ids, chosen_mask, prompt_length
        )
        pi_rejected = self.get_log_probs(
            self.policy, rejected_ids, rejected_mask, prompt_length
        )

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_chosen = self.get_log_probs(
                self.ref, chosen_ids, chosen_mask, prompt_length
            )
            ref_rejected = self.get_log_probs(
                self.ref, rejected_ids, rejected_mask, prompt_length
            )

        # Log ratios
        chosen_log_ratio = pi_chosen - ref_chosen
        rejected_log_ratio = pi_rejected - ref_rejected

        # DPO loss
        loss = -F.logsigmoid(
            self.beta * (chosen_log_ratio - rejected_log_ratio)
        ).mean()

        # Metrics
        chosen_reward = self.beta * chosen_log_ratio
        rejected_reward = self.beta * rejected_log_ratio
        accuracy = (chosen_reward > rejected_reward).float().mean()

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward': chosen_reward.mean().item(),
            'rejected_reward': rejected_reward.mean().item(),
            'reward_margin': (chosen_reward - rejected_reward).mean().item()
        }

        return loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single DPO training step."""
        self.policy.train()

        # Move to device
        chosen_ids = batch['chosen_ids'].to(self.device)
        chosen_mask = batch['chosen_mask'].to(self.device)
        rejected_ids = batch['rejected_ids'].to(self.device)
        rejected_mask = batch['rejected_mask'].to(self.device)
        prompt_length = batch['prompt_length'][0].item()  # Assume same for batch

        # Compute loss
        self.optimizer.zero_grad()
        loss, metrics = self.compute_dpo_loss(
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask,
            prompt_length
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return metrics


# ============================================================================
# Section 4: PPO for Language Models
# ============================================================================

class ValueHead(nn.Module):
    """Value function head for PPO."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic model for PPO.

    Actor: LLM that generates responses
    Critic: Value function estimating expected reward
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        vocab_size: int
    ):
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(hidden_size)

        # LM head for action probabilities
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action logits and value estimates.

        Returns:
            logits: (batch, seq, vocab)
            values: (batch, seq)
        """
        hidden = self.backbone(input_ids, attention_mask=attention_mask)

        if hasattr(hidden, 'last_hidden_state'):
            hidden = hidden.last_hidden_state

        logits = self.lm_head(hidden)
        values = self.value_head(hidden)

        return logits, values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate response with action log probs and values.

        Returns:
            response_ids: Generated token IDs
            log_probs: Log probabilities of generated tokens
            values: Value estimates at each position
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        generated = input_ids
        all_log_probs = []
        all_values = []

        for _ in range(max_length):
            logits, values = self(generated)

            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)

            # Sample
            next_token = torch.multinomial(probs, num_samples=1)

            # Get log prob of sampled token
            log_prob = F.log_softmax(next_logits, dim=-1)
            token_log_prob = log_prob.gather(1, next_token).squeeze(-1)

            # Get value
            value = values[:, -1]

            all_log_probs.append(token_log_prob)
            all_values.append(value)

            # Append token
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS (simplified - would need actual EOS token)
            if (next_token == 0).all():
                break

        return (
            generated,
            torch.stack(all_log_probs, dim=1),
            torch.stack(all_values, dim=1)
        )


class PPOTrainer:
    """
    PPO trainer for RLHF.

    Objective: max E[R(x,y)] - β × KL(π || π_ref)

    PPO uses clipped objective:
    L = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        ref_model: nn.Module,
        reward_model: RewardModel,
        optimizer: torch.optim.Optimizer,
        kl_coef: float = 0.1,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = 'cpu'
    ):
        self.actor_critic = actor_critic.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        # Freeze reference and reward models
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

    def compute_rewards(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards with KL penalty.

        reward = R(x, y) - β × KL(π || π_ref)
        """
        # Full sequence for reward model
        full_seq = torch.cat([prompts, responses], dim=1)
        attention_mask = (full_seq != 0).long()

        # Get reward from reward model
        with torch.no_grad():
            base_reward = self.reward_model(full_seq, attention_mask)

            # Get reference log probs for KL
            ref_outputs = self.ref_model(full_seq, attention_mask=attention_mask)
            if hasattr(ref_outputs, 'logits'):
                ref_logits = ref_outputs.logits
            else:
                ref_logits = ref_outputs

            # Compute per-token KL
            # Simplified: use log prob difference as KL approximation
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # Get log probs for response tokens
            response_start = prompts.size(1)
            response_tokens = full_seq[:, response_start:]

            ref_token_log_probs = ref_log_probs[:, response_start-1:-1, :].gather(
                dim=-1,
                index=response_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # KL penalty (per token)
            kl = old_log_probs - ref_token_log_probs
            kl_penalty = self.kl_coef * kl.sum(dim=1)

        # Final reward
        reward = base_reward - kl_penalty

        return reward

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> torch.Tensor:
        """
        Compute GAE advantages.

        A_t = sum_{l=0}^{inf} (γλ)^l × δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)

        # Assume terminal reward at last step
        last_advantage = 0

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                # Terminal step
                delta = rewards - values[:, t]
            else:
                delta = gamma * values[:, t + 1] - values[:, t]

            advantages[:, t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[:, t]

        return advantages

    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss.

        L = L_clip + c1 × L_value - c2 × H(π)
        """
        # Policy ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (encourages exploration)
        # Simplified - would need actual distribution entropy
        entropy = -(log_probs.exp() * log_probs).mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
            'approx_kl': ((ratio - 1) - (ratio.log())).mean().item()
        }

        return loss, metrics

    def train_step(
        self,
        prompts: torch.Tensor,
        ppo_epochs: int = 4,
        max_response_len: int = 100
    ) -> Dict[str, float]:
        """
        Single PPO training iteration.

        1. Generate responses
        2. Compute rewards
        3. Compute advantages
        4. Update policy with PPO
        """
        prompts = prompts.to(self.device)

        # Generate responses
        with torch.no_grad():
            responses, old_log_probs, old_values = self.actor_critic.generate(
                prompts, max_length=max_response_len
            )
            response_tokens = responses[:, prompts.size(1):]

            # Compute rewards
            rewards = self.compute_rewards(prompts, response_tokens, old_log_probs)

            # Compute advantages
            advantages = self.compute_advantages(rewards.unsqueeze(1).expand_as(old_values), old_values)
            returns = advantages + old_values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        all_metrics = []
        for _ in range(ppo_epochs):
            # Get current policy outputs
            logits, values = self.actor_critic(responses)

            # Get log probs for response tokens
            response_start = prompts.size(1)
            response_logits = logits[:, response_start-1:-1, :]
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1,
                index=response_tokens.unsqueeze(-1)
            ).squeeze(-1)

            response_values = values[:, response_start:]

            # Compute loss
            self.optimizer.zero_grad()
            loss, metrics = self.ppo_loss(
                token_log_probs,
                old_log_probs,
                advantages,
                response_values,
                returns
            )

            # Update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
            self.optimizer.step()

            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0]
        }
        avg_metrics['mean_reward'] = rewards.mean().item()

        return avg_metrics


# ============================================================================
# Section 5: KTO (Kahneman-Tversky Optimization)
# ============================================================================

class KTOTrainer:
    """
    Kahneman-Tversky Optimization trainer.

    Unlike DPO, KTO works with unpaired data:
    (prompt, response, is_good) instead of (prompt, chosen, rejected)

    Loss based on prospect theory:
    - Different treatment for gains (good responses) vs losses (bad responses)
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        device: str = 'cpu'
    ):
        self.policy = policy_model.to(device)
        self.ref = ref_model.to(device)
        self.optimizer = optimizer
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.device = device

        # Freeze reference
        for param in self.ref.parameters():
            param.requires_grad = False

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """Get response log probabilities."""
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        response_mask = torch.zeros_like(shift_mask)
        response_mask[:, prompt_length-1:] = shift_mask[:, prompt_length-1:]

        return (token_log_probs * response_mask).sum(dim=1)

    def compute_kto_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_desirable: torch.Tensor,
        prompt_length: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute KTO loss.

        For desirable (good) responses:
        L = -sigmoid(β × log_ratio - KL_ref)

        For undesirable (bad) responses:
        L = -sigmoid(KL_ref - β × log_ratio)
        """
        # Policy log probs
        pi_logprobs = self.get_log_probs(
            self.policy, input_ids, attention_mask, prompt_length
        )

        # Reference log probs
        with torch.no_grad():
            ref_logprobs = self.get_log_probs(
                self.ref, input_ids, attention_mask, prompt_length
            )

        # Log ratio
        log_ratio = pi_logprobs - ref_logprobs

        # KL term (average log ratio as proxy)
        kl_ref = log_ratio.mean().detach()

        # Separate losses for desirable vs undesirable
        desirable_mask = is_desirable.bool()
        undesirable_mask = ~desirable_mask

        loss = torch.tensor(0.0, device=self.device)

        if desirable_mask.any():
            desirable_loss = -F.logsigmoid(
                self.beta * log_ratio[desirable_mask] - kl_ref
            ).mean()
            loss = loss + self.desirable_weight * desirable_loss

        if undesirable_mask.any():
            undesirable_loss = -F.logsigmoid(
                kl_ref - self.beta * log_ratio[undesirable_mask]
            ).mean()
            loss = loss + self.undesirable_weight * undesirable_loss

        metrics = {
            'loss': loss.item(),
            'kl': log_ratio.mean().item(),
            'desirable_ratio': desirable_mask.float().mean().item()
        }

        return loss, metrics


# ============================================================================
# Section 6: Simple Language Model for Testing
# ============================================================================

class SimpleTransformer(nn.Module):
    """Simple transformer for testing alignment algorithms."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 256
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        x = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(x)

        return type('Output', (), {'logits': logits, 'last_hidden_state': x})()


# ============================================================================
# Section 7: Demo Functions
# ============================================================================

def demo_reward_model():
    """Demonstrate reward model training."""
    print("=" * 60)
    print("Reward Model Training Demo")
    print("=" * 60)

    # Create models
    backbone = SimpleTransformer(vocab_size=100, d_model=64, num_layers=1)
    reward_model = RewardModel(backbone, hidden_size=64)

    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    trainer = RewardModelTrainer(reward_model, optimizer)

    # Simulate batch
    batch = {
        'chosen_ids': torch.randint(1, 100, (4, 32)),
        'chosen_mask': torch.ones(4, 32).long(),
        'rejected_ids': torch.randint(1, 100, (4, 32)),
        'rejected_mask': torch.ones(4, 32).long()
    }

    print("\nTraining reward model...")
    for step in range(5):
        metrics = trainer.train_step(batch)
        print(f"  Step {step+1}: Loss={metrics['loss']:.4f}, "
              f"Accuracy={metrics['accuracy']:.4f}, "
              f"Reward diff={metrics['reward_diff']:.4f}")


def demo_dpo():
    """Demonstrate DPO training."""
    print("\n" + "=" * 60)
    print("DPO Training Demo")
    print("=" * 60)

    # Create models
    policy = SimpleTransformer(vocab_size=100, d_model=64, num_layers=1)
    ref = SimpleTransformer(vocab_size=100, d_model=64, num_layers=1)

    # Copy weights to reference
    ref.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    trainer = DPOTrainer(policy, ref, optimizer, beta=0.1)

    # Simulate batch
    batch = {
        'chosen_ids': torch.randint(1, 100, (4, 32)),
        'chosen_mask': torch.ones(4, 32).long(),
        'rejected_ids': torch.randint(1, 100, (4, 32)),
        'rejected_mask': torch.ones(4, 32).long(),
        'prompt_length': torch.tensor([8, 8, 8, 8])
    }

    print("\nTraining with DPO...")
    for step in range(5):
        metrics = trainer.train_step(batch)
        print(f"  Step {step+1}: Loss={metrics['loss']:.4f}, "
              f"Accuracy={metrics['accuracy']:.4f}, "
              f"Reward margin={metrics['reward_margin']:.4f}")


def demo_ppo_components():
    """Demonstrate PPO components."""
    print("\n" + "=" * 60)
    print("PPO Components Demo")
    print("=" * 60)

    # Create actor-critic
    backbone = SimpleTransformer(vocab_size=100, d_model=64, num_layers=1)
    actor_critic = ActorCritic(backbone, hidden_size=64, vocab_size=100)

    print("\nActor-Critic forward pass:")
    input_ids = torch.randint(1, 100, (2, 16))
    logits, values = actor_critic(input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Values shape: {values.shape}")

    # Simulate PPO loss computation
    print("\nSimulating PPO loss:")
    old_log_probs = torch.randn(2, 10)
    new_log_probs = old_log_probs + torch.randn(2, 10) * 0.1
    advantages = torch.randn(2, 10)
    values = torch.randn(2, 10)
    returns = values + advantages

    # Compute ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    print(f"  Probability ratio mean: {ratio.mean():.4f}")

    # Clipped objective
    clip_eps = 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    print(f"  Policy loss: {policy_loss:.4f}")

    # Value loss
    value_loss = F.mse_loss(values, returns)
    print(f"  Value loss: {value_loss:.4f}")


def demo_alignment_comparison():
    """Compare different alignment methods."""
    print("\n" + "=" * 60)
    print("Alignment Methods Comparison")
    print("=" * 60)

    methods = [
        ("RLHF (PPO)", "Reward model + RL", "High", "Complex"),
        ("DPO", "Direct from preferences", "Medium", "Simple"),
        ("KTO", "Unpaired data", "Medium", "Simple"),
        ("IPO", "Margin-based DPO", "Medium", "Simple"),
        ("ORPO", "Single-stage", "Low", "Simple"),
    ]

    print(f"\n{'Method':<15} {'Data Format':<25} {'Compute':<10} {'Complexity':<10}")
    print("-" * 60)
    for method, data, compute, complexity in methods:
        print(f"{method:<15} {data:<25} {compute:<10} {complexity:<10}")

    print("\n" + "-" * 60)
    print("\nRecommendations:")
    print("  - Default choice: DPO (simple, effective)")
    print("  - Maximum quality: RLHF")
    print("  - Limited annotations: KTO")
    print("  - Single stage: ORPO")


def demo_kl_penalty():
    """Demonstrate KL penalty importance."""
    print("\n" + "=" * 60)
    print("KL Penalty Demonstration")
    print("=" * 60)

    # Simulate policy divergence
    ref_probs = F.softmax(torch.randn(5, 10), dim=-1)

    print("\nKL divergence as policy diverges from reference:")
    print(f"{'Temperature':<15} {'KL Divergence':<15} {'Status':<15}")
    print("-" * 45)

    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        # Simulate policy with different temperatures
        policy_probs = F.softmax(torch.randn(5, 10) * temp, dim=-1)

        # KL divergence
        kl = (policy_probs * (policy_probs.log() - ref_probs.log())).sum(dim=-1).mean()

        status = "OK" if kl < 1.0 else "WARNING" if kl < 5.0 else "DANGER"
        print(f"{temp:<15.1f} {kl.item():<15.4f} {status:<15}")

    print("\nNote: High KL indicates policy has drifted far from reference")
    print("This can lead to reward hacking and degraded language quality")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("Module 15: LLM Alignment - RLHF, DPO, and Beyond")
    print("=" * 60)

    demo_reward_model()
    demo_dpo()
    demo_ppo_components()
    demo_alignment_comparison()
    demo_kl_penalty()

    print("\n" + "=" * 60)
    print("Module 15 Complete!")
    print("=" * 60)
    print("\nKey concepts covered:")
    print("1. Reward model training with Bradley-Terry loss")
    print("2. DPO - Direct Preference Optimization")
    print("3. PPO components for RLHF")
    print("4. KTO for unpaired preference data")
    print("5. KL penalty to prevent reward hacking")
    print("6. Comparison of alignment methods")


if __name__ == "__main__":
    main()
