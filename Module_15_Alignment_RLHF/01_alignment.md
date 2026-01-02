# Module 15: LLM Alignment - RLHF, DPO, and Beyond

## Table of Contents
1. [Why Alignment?](#1-why-alignment)
2. [The RLHF Pipeline](#2-the-rlhf-pipeline)
3. [Reward Modeling](#3-reward-modeling)
4. [PPO for LLMs](#4-ppo-for-llms)
5. [Direct Preference Optimization (DPO)](#5-direct-preference-optimization-dpo)
6. [Other Alignment Methods](#6-other-alignment-methods)
7. [Practical Considerations](#7-practical-considerations)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Why Alignment?

### 1.1 The Problem with Pure Pretraining

Pretrained LLMs are good at predicting text but not at being helpful:
```
Problem behaviors:
- Hallucination (confident but wrong)
- Harmful content generation
- Refusal to help when appropriate
- Sycophancy (telling users what they want to hear)
- Following instructions poorly

Root cause:
  Pretraining objective: P(next_token | context)
  What we want: Helpful, harmless, honest responses
```

### 1.2 The Alignment Goal

Make LLMs:
```
1. Helpful: Follow instructions, provide useful information
2. Harmless: Refuse harmful requests, avoid generating toxic content
3. Honest: Acknowledge uncertainty, don't hallucinate

Challenge: These goals can conflict!
  - Being helpful vs refusing harmful requests
  - Being honest vs being helpful (uncertainty acknowledgment)
```

### 1.3 Alignment Approaches

```
1. RLHF (Reinforcement Learning from Human Feedback)
   - Train reward model on human preferences
   - Fine-tune with RL (PPO)
   - Used by: ChatGPT, Claude

2. DPO (Direct Preference Optimization)
   - Skip reward model training
   - Directly optimize from preferences
   - Simpler, similar results

3. Constitutional AI (CAI)
   - AI critiques and revises its own outputs
   - Uses principles/constitution as guide

4. RLAIF (RL from AI Feedback)
   - Use AI instead of humans for feedback
   - Scalable but potentially biased
```

---

## 2. The RLHF Pipeline

### 2.1 Three-Stage Pipeline

```
Stage 1: Supervised Fine-Tuning (SFT)
  Input: Pretrained LLM + demonstration data
  Output: LLM that follows instructions
  Method: Standard cross-entropy training

Stage 2: Reward Model Training
  Input: LLM responses + human preferences
  Output: Reward model R(prompt, response) → score
  Method: Train on pairwise comparisons

Stage 3: RL Fine-Tuning
  Input: SFT model + reward model
  Output: Aligned LLM
  Method: PPO with KL penalty
```

### 2.2 Stage 1: Supervised Fine-Tuning

Transform pretrained LLM into instruction-follower:
```
Training data:
  (instruction, high-quality response) pairs

Sources:
  - Human-written demonstrations
  - Existing datasets (FLAN, Dolly, etc.)
  - AI-generated with filtering

Training:
  Standard language modeling loss
  L = -log P(response | instruction)
```

### 2.3 Data Format for SFT

```
<system>You are a helpful assistant.</system>
<human>What is the capital of France?</human>
<assistant>The capital of France is Paris.</assistant>

Or:

### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

---

## 3. Reward Modeling

### 3.1 Preference Data Collection

```
Process:
1. Sample prompt from distribution
2. Generate multiple responses (usually 2)
3. Human labels which response is better
4. Store: (prompt, response_chosen, response_rejected)

Label options:
- A > B (A better)
- B > A (B better)
- A = B (tie, often discarded)
```

### 3.2 Bradley-Terry Model

Convert preferences to reward model:
```
Assumption: Human preference follows Bradley-Terry model
  P(A > B) = σ(R(A) - R(B))
           = exp(R(A)) / (exp(R(A)) + exp(R(B)))

Where:
  σ = sigmoid function
  R(x) = reward model score for response x
```

### 3.3 Reward Model Training

```
Loss function:
  L = -E[(log σ(r_w - r_l))]

Where:
  r_w = R(prompt, response_chosen)
  r_l = R(prompt, response_rejected)

Architecture:
  - Same as LLM but with scalar head
  - Often initialize from SFT model
  - Output: single scalar reward
```

### 3.4 Reward Model Architecture

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        # Use LLM backbone
        self.backbone = base_model

        # Replace LM head with scalar head
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids):
        # Get last hidden state
        hidden = self.backbone(input_ids).last_hidden_state

        # Pool (usually last token or mean)
        pooled = hidden[:, -1, :]  # Last token

        # Get scalar reward
        reward = self.reward_head(pooled)
        return reward
```

### 3.5 Challenges in Reward Modeling

```
1. Reward hacking: Model exploits reward model quirks
   Example: Longer responses score higher → model becomes verbose

2. Distribution shift: RM trained on one distribution,
   policy generates different distribution

3. Annotator disagreement: Humans disagree on preferences
   Solution: Use multiple annotators, measure agreement

4. Reward model overoptimization:
   Higher reward doesn't always mean better response
   Need KL penalty to prevent exploitation
```

---

## 4. PPO for LLMs

### 4.1 Why RL?

```
Why not just maximize reward directly?
  - Reward model is imperfect
  - Direct maximization leads to reward hacking
  - Need to stay close to SFT model (KL constraint)

PPO provides:
  - Stable training
  - KL penalty naturally incorporated
  - Clip prevents too large updates
```

### 4.2 RLHF Objective

```
Maximize:
  E_π[R(x, y)] - β × KL(π || π_ref)

Where:
  π = current policy (model being trained)
  π_ref = reference policy (SFT model, frozen)
  R(x, y) = reward model score
  β = KL coefficient (controls how far from ref)
  x = prompt
  y = response
```

### 4.3 PPO Algorithm for LLMs

```
1. Sample batch of prompts
2. Generate responses from current policy
3. Compute rewards R(prompt, response)
4. Compute KL penalty: KL(π || π_ref)
5. Compute advantages using GAE
6. PPO update with clipping

Repeat until convergence
```

### 4.4 PPO Clipping

Prevent too large policy updates:
```
L_CLIP = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]

Where:
  r_t = π(a|s) / π_old(a|s)  (probability ratio)
  A_t = advantage estimate
  ε = clip range (typically 0.1-0.2)

For LLMs:
  a = token
  s = context
  π(a|s) = token probability
```

### 4.5 Value Function

```
Critic network:
  V(s) estimates expected future reward

Architecture options:
1. Separate model (expensive but stable)
2. Head on policy model (cheaper but coupled)
3. Reward model as V function (common in practice)

Advantage:
  A_t = R_t - V(s_t)  (simple)
  Or use GAE for lower variance
```

### 4.6 KL Penalty Implementation

```python
# Log probabilities
logprobs = policy_model.log_prob(response)
ref_logprobs = ref_model.log_prob(response)  # Frozen

# Per-token KL divergence
kl = logprobs - ref_logprobs  # Approximation

# Add to reward
modified_reward = reward - beta * kl.sum()
```

---

## 5. Direct Preference Optimization (DPO)

### 5.1 Motivation

RLHF is complex:
```
RLHF requirements:
- Train separate reward model
- Complex PPO training loop
- Multiple models in memory
- Hyperparameter sensitive

DPO insight:
  Can we directly optimize from preferences?
  Yes! By reformulating the RL objective
```

### 5.2 DPO Derivation (Simplified)

```
RLHF objective:
  max E[R(x,y)] - β × KL(π || π_ref)

Optimal policy has closed form:
  π*(y|x) ∝ π_ref(y|x) × exp(R(x,y) / β)

Rearranging:
  R(x,y) = β × log(π*(y|x) / π_ref(y|x)) + const

Key insight:
  Reward is a function of policy ratio!
  We can substitute this into Bradley-Terry loss
```

### 5.3 DPO Loss Function

```
L_DPO = -E[log σ(β × (log π(y_w|x)/π_ref(y_w|x) -
                      log π(y_l|x)/π_ref(y_l|x)))]

Where:
  y_w = chosen (winning) response
  y_l = rejected (losing) response
  π = current policy
  π_ref = reference policy (SFT model)
  β = temperature parameter
```

### 5.4 DPO Implementation

```python
def dpo_loss(policy_model, ref_model, prompts, chosen, rejected, beta=0.1):
    # Get log probabilities
    pi_chosen = policy_model.log_prob(chosen, prompts)
    pi_rejected = policy_model.log_prob(rejected, prompts)

    ref_chosen = ref_model.log_prob(chosen, prompts)  # Frozen
    ref_rejected = ref_model.log_prob(rejected, prompts)

    # DPO loss
    chosen_rewards = beta * (pi_chosen - ref_chosen)
    rejected_rewards = beta * (pi_rejected - ref_rejected)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss
```

### 5.5 DPO vs RLHF Comparison

| Aspect | RLHF | DPO |
|--------|------|-----|
| Reward model | Required | Not needed |
| Training | Complex (PPO) | Simple (SL-like) |
| Memory | Multiple models | Two models |
| Stability | Can be unstable | Very stable |
| Quality | Slightly better | Very close |
| Complexity | High | Low |

### 5.6 When to Use Which?

```
DPO advantages:
- Simpler to implement
- More stable training
- Fewer hyperparameters
- Lower memory

RLHF advantages:
- Reward model can be reused
- Online sample generation
- Can iterate on reward model
- Slightly better results (sometimes)

Recommendation: Start with DPO, switch to RLHF if needed
```

---

## 6. Other Alignment Methods

### 6.1 Constitutional AI (CAI)

Self-critique and revision:
```
Process:
1. Model generates response
2. Model critiques own response using principles
3. Model revises based on critique
4. Train on revised responses

Constitution (principles):
- Be helpful and harmless
- Avoid deception
- Respect user autonomy
- etc.
```

### 6.2 RLAIF (RL from AI Feedback)

Use AI instead of humans:
```
Replace human labelers with LLM:
1. Generate response pairs
2. Ask LLM to choose better one
3. Train reward model on AI preferences
4. Standard RLHF pipeline

Benefits:
- Scalable (no human cost)
- Consistent (no annotator variance)

Risks:
- AI biases propagate
- May not capture human values
```

### 6.3 IPO (Identity Preference Optimization)

Addresses DPO limitations:
```
DPO issue: Can over-optimize, reducing diversity

IPO adds regularization:
  L_IPO = (log(π(y_w)/π_ref(y_w)) - log(π(y_l)/π_ref(y_l)) - margin)²

Pushes toward margin, not infinity
More stable, maintains diversity
```

### 6.4 KTO (Kahneman-Tversky Optimization)

Uses unpaired data:
```
DPO requires: (prompt, chosen, rejected) triplets

KTO allows: (prompt, response, is_good) pairs
  - More flexible data format
  - Easier to collect
  - Similar performance

Loss based on prospect theory:
  Different treatment for gains vs losses
```

### 6.5 ORPO (Odds Ratio Preference Optimization)

Combines SFT and preference optimization:
```
Traditional: SFT → DPO (two stages)
ORPO: Single stage

Loss = L_SFT + λ × L_OR

Where L_OR penalizes rejected responses
More efficient, competitive results
```

### 6.6 Comparison of Methods

| Method | Data Format | Stages | Complexity |
|--------|-------------|--------|------------|
| RLHF | Pairs | 3 | High |
| DPO | Pairs | 2 | Medium |
| IPO | Pairs | 2 | Medium |
| KTO | Singles | 2 | Low |
| ORPO | Pairs | 1 | Low |
| CAI | None (self) | 2+ | Medium |

---

## 7. Practical Considerations

### 7.1 Data Quality

```
Critical for alignment:
1. Diverse prompts (cover distribution)
2. Clear preferences (high annotator agreement)
3. Sufficient quantity (10K-100K pairs typically)
4. Representative of deployment

Red flags:
- All preferences for same style
- Low annotator agreement (<70%)
- Narrow prompt distribution
```

### 7.2 Hyperparameter Tuning

```
DPO:
  β (beta): 0.1 - 0.5
    Lower = stronger preference optimization
    Higher = closer to reference model

RLHF PPO:
  KL coefficient (β): 0.01 - 0.2
  Learning rate: 1e-6 to 5e-6
  PPO epochs: 2-4
  Batch size: 32-256

General:
  Start conservative (high β/KL penalty)
  Gradually decrease if undertrained
```

### 7.3 Evaluation

```
Metrics:
1. Win rate vs SFT model (human eval)
2. Reward model score (proxy)
3. Benchmarks (MT-Bench, AlpacaEval)
4. Safety evaluations

Challenges:
- Human eval expensive
- Reward model can be gamed
- Benchmarks saturate quickly
```

### 7.4 Common Pitfalls

```
1. Reward hacking:
   Symptom: High reward, low quality
   Solution: KL penalty, diverse eval

2. Mode collapse:
   Symptom: All responses similar
   Solution: Higher temperature, more KL

3. Forgetting:
   Symptom: Worse at pretrain tasks
   Solution: Mix in pretrain data, lower LR

4. Sycophancy:
   Symptom: Model agrees with everything
   Solution: Diverse annotators, CAI
```

### 7.5 Training Recipe

```
1. Start with strong SFT model
   - Use high-quality demonstrations
   - Don't overtrain (1-3 epochs)

2. Collect preference data
   - Clear guidelines for annotators
   - Multiple annotators per pair
   - Filter low-agreement samples

3. Choose alignment method
   - DPO for simplicity
   - RLHF for maximum control

4. Train with conservative hyperparameters
   - High β (DPO) or KL penalty (PPO)
   - Low learning rate
   - Monitor reward and KL

5. Evaluate thoroughly
   - Hold-out preference data
   - Human evaluation
   - Safety testing
```

---

## 8. Interview Questions

### Q1: Explain the RLHF pipeline.

**Answer**:

RLHF has three stages:

**Stage 1: Supervised Fine-Tuning (SFT)**
```
- Transform pretrained LLM to instruction-follower
- Train on (instruction, response) pairs
- Standard cross-entropy loss
```

**Stage 2: Reward Modeling**
```
- Collect preference data: (prompt, chosen, rejected)
- Train reward model using Bradley-Terry:
  L = -log σ(R(chosen) - R(rejected))
- Architecture: LLM backbone + scalar head
```

**Stage 3: RL Fine-Tuning (PPO)**
```
Objective: max E[R(x,y)] - β × KL(π || π_ref)

Process:
1. Sample prompts
2. Generate responses with current policy
3. Score with reward model
4. Update with PPO (clipped objective)
5. KL penalty keeps policy close to SFT
```

**Why KL penalty?**
- Prevents reward hacking
- Maintains language quality
- Balances reward vs staying in-distribution

### Q2: What is DPO and how does it compare to RLHF?

**Answer**:

**DPO** (Direct Preference Optimization) directly optimizes preferences without reward model.

**Key insight**:
```
Optimal RLHF policy has closed form:
  π*(y|x) ∝ π_ref(y|x) × exp(R(x,y) / β)

This means reward can be expressed as:
  R(x,y) = β × log(π*(y|x) / π_ref(y|x)) + const

Substitute into Bradley-Terry → DPO loss
```

**DPO Loss**:
```python
loss = -log σ(β × (log_ratio_chosen - log_ratio_rejected))

where log_ratio = log(π/π_ref)
```

**Comparison**:
| | RLHF | DPO |
|--|------|-----|
| Reward model | Yes | No |
| Training | Complex PPO | Simple gradient descent |
| Stability | Can diverge | Very stable |
| Memory | High | Lower |
| Results | Slightly better | Very close |

**When to use DPO**: Default choice for simplicity
**When to use RLHF**: Need reward model for other purposes, or marginal quality matters

### Q3: How do you handle reward hacking in RLHF?

**Answer**:

**Problem**: Model learns to exploit reward model quirks rather than being genuinely helpful.

**Examples**:
```
- Longer responses score higher → excessive verbosity
- Formal language scores higher → overly formal
- Agreeing with user scores higher → sycophancy
```

**Solutions**:

1. **KL Penalty**:
```
Objective: R(x,y) - β × KL(π || π_ref)

Prevents policy from drifting too far from SFT
Higher β = more constraint
```

2. **Diverse Reward Models**:
```
Train multiple RMs with different data
Use ensemble for scoring
Harder to hack all simultaneously
```

3. **Reward Model Regularization**:
```
Add penalty for reward magnitude
Prevent extreme scores
```

4. **Iterative Training**:
```
Retrain reward model on policy outputs
Closes distribution gap
Harder to hack moving target
```

5. **Human Evaluation**:
```
Regularly check outputs manually
Catch hacking early
Add to RM training data
```

### Q4: Explain the Bradley-Terry model for preference learning.

**Answer**:

**Bradley-Terry** models pairwise comparisons probabilistically.

**Assumption**:
```
Each item has latent "strength" or "quality" score
Probability of A > B depends on score difference
```

**Model**:
```
P(A > B) = σ(s_A - s_B) = exp(s_A) / (exp(s_A) + exp(s_B))

Where:
  s_A, s_B = strength scores
  σ = sigmoid function
```

**For reward modeling**:
```
s_A = R(prompt, response_A)
s_B = R(prompt, response_B)

P(A preferred) = σ(R(A) - R(B))
```

**Training loss**:
```
L = -log P(chosen > rejected)
  = -log σ(R(chosen) - R(rejected))

Maximize probability of observed preferences
```

**Properties**:
- Transitive (if A>B and B>C, tends to A>C)
- Handles ties poorly (often discarded)
- Assumes preferences depend only on pair

### Q5: What are the tradeoffs between different alignment methods?

**Answer**:

**RLHF (PPO)**:
```
Pros:
- Gold standard, proven results
- Reward model reusable
- Online generation during training

Cons:
- Complex implementation
- Multiple models in memory
- Unstable training
- Many hyperparameters
```

**DPO**:
```
Pros:
- Simple, stable training
- No reward model needed
- Similar results to RLHF

Cons:
- Offline (fixed preference data)
- Can over-optimize (reduce diversity)
- No reusable reward model
```

**KTO**:
```
Pros:
- Works with unpaired data
- Easier data collection

Cons:
- Newer, less proven
- May need more data
```

**Constitutional AI**:
```
Pros:
- Scalable (AI feedback)
- Explicit principles

Cons:
- AI biases propagate
- Limited by AI capability
```

**Recommendations**:
- **Default**: DPO (simplicity + quality)
- **Scale/iteration**: RLHF (reward model reuse)
- **Limited data**: KTO (flexible format)
- **Automated**: CAI + RLAIF

---

## 9. Summary

### RLHF Pipeline

```
Stage 1: SFT
  Pretrained → Instruction-following
  Loss: Cross-entropy on demonstrations

Stage 2: Reward Model
  Learn R(prompt, response)
  Loss: -log σ(R_chosen - R_rejected)

Stage 3: PPO
  Optimize: E[R] - β × KL
  Clip updates, value baseline
```

### DPO Formula

```
L = -log σ(β × (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))

No reward model, stable training, similar results
```

### Key Hyperparameters

| Method | Parameter | Typical Range |
|--------|-----------|---------------|
| DPO | β | 0.1 - 0.5 |
| PPO | KL coef | 0.01 - 0.2 |
| PPO | Clip ε | 0.1 - 0.2 |
| Both | Learning rate | 1e-6 - 5e-6 |

### Method Selection

| Scenario | Recommended |
|----------|-------------|
| Simple alignment | DPO |
| Maximum quality | RLHF |
| Limited annotations | KTO |
| Scalable | RLAIF + DPO |
| Safety focus | CAI |

### Key Takeaways

1. **RLHF is the standard** but complex
2. **DPO achieves similar results** with simpler training
3. **KL penalty is crucial** to prevent reward hacking
4. **Data quality matters more** than algorithm choice
5. **Start conservative** with hyperparameters
6. **Evaluate with humans** not just reward models
