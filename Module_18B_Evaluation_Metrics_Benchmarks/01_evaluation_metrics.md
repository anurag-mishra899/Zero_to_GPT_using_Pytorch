# Evaluation Metrics & Benchmarks for LLMs

## Overview

How do we know if a language model is "good"? This module covers the metrics, benchmarks, and evaluation frameworks used to assess LLM performance - from classical NLP metrics to modern LLM-as-judge approaches.

## Table of Contents
1. [Why Evaluation is Hard](#why-evaluation-is-hard)
2. [Perplexity - The Foundation](#perplexity---the-foundation)
3. [Generation Quality Metrics](#generation-quality-metrics)
4. [Task-Specific Metrics](#task-specific-metrics)
5. [LLM Benchmarks](#llm-benchmarks)
6. [LLM-as-Judge Evaluation](#llm-as-judge-evaluation)
7. [Human Evaluation](#human-evaluation)
8. [Practical Evaluation Strategies](#practical-evaluation-strategies)
9. [Interview Questions](#interview-questions)

---

## Why Evaluation is Hard

### The Core Challenge

LLMs are general-purpose - they can write poetry, solve math, code, and chat. No single metric captures all capabilities.

### Key Difficulties

1. **Subjectivity**: "Good" text depends on context
2. **Open-endedness**: Many valid answers exist
3. **Multi-dimensional**: Fluency ≠ factuality ≠ helpfulness
4. **Distribution Shift**: Training data ≠ real-world usage
5. **Gaming**: Models can overfit to benchmarks

### Evaluation Taxonomy

```
Evaluation Methods
├── Automatic Metrics
│   ├── Intrinsic (Perplexity, loss)
│   ├── Reference-based (BLEU, ROUGE)
│   └── Reference-free (LLM-as-judge)
├── Human Evaluation
│   ├── Absolute ratings
│   ├── Comparative (A/B testing)
│   └── Task completion
└── Benchmark Suites
    ├── Knowledge (MMLU, ARC)
    ├── Reasoning (GSM8K, BBH)
    └── Safety (TruthfulQA, HHH)
```

---

## Perplexity - The Foundation

### Definition

Perplexity measures how "surprised" the model is by test data:

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | context)\right)$$

### Interpretation

| PPL Value | Meaning |
|-----------|---------|
| 1 | Perfect prediction |
| 10 | As uncertain as 10 equally likely options |
| 100 | Very uncertain |
| V (vocab size) | Random guessing |

### Perplexity Pitfalls

**1. Vocabulary Dependence**
```
GPT-2 (50K vocab) PPL = 20
LLaMA (32K vocab) PPL = 15
→ Not directly comparable!
```

**2. Domain Sensitivity**
```
Model trained on news:
- Test on news: PPL = 15
- Test on code: PPL = 200
```

**3. Context Length Effects**
```
32 token context: PPL = 25
512 token context: PPL = 18
2048 token context: PPL = 15
```

### Bits Per Byte (BPB)

A more comparable metric:

$$\text{BPB} = \frac{\log_2(\text{PPL})}{\text{bytes per token}}$$

**Advantage**: Tokenizer-independent

---

## Generation Quality Metrics

### BLEU (Bilingual Evaluation Understudy)

Originally for machine translation, measures n-gram overlap with reference.

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $p_n$ = precision of n-grams
- $BP$ = brevity penalty
- $w_n$ = weights (typically uniform)

**BLEU Characteristics**:
- Range: 0-100 (often expressed as 0-1)
- Higher is better
- BLEU-4 most common (up to 4-grams)

**Limitations**:
- Ignores recall (doesn't penalize missing content)
- No semantic understanding
- Multiple references improve reliability

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Designed for summarization, focuses on recall.

**Variants**:
- **ROUGE-N**: N-gram recall
- **ROUGE-L**: Longest Common Subsequence
- **ROUGE-W**: Weighted LCS

$$\text{ROUGE-N} = \frac{\sum_{\text{ref}} \sum_{\text{gram}_n} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{\text{ref}} \sum_{\text{gram}_n} \text{Count}(\text{gram}_n)}$$

**Commonly Reported**:
- ROUGE-1, ROUGE-2, ROUGE-L
- Precision, Recall, F1 variants

### BERTScore

Uses BERT embeddings for semantic similarity:

$$\text{BERTScore} = \frac{1}{|x|} \sum_{x_i} \max_{y_j} \text{sim}(\mathbf{x}_i, \mathbf{y}_j)$$

**Advantages**:
- Captures paraphrasing
- Better correlation with human judgment
- Language-agnostic (with multilingual BERT)

**Disadvantages**:
- Computationally expensive
- Depends on embedding model quality

### METEOR

Addresses BLEU limitations with:
- Stemming and synonymy matching
- Explicit recall component
- Chunk-based penalty for fluency

### chrF

Character-level F-score, robust across languages:
- No tokenization needed
- Good for morphologically rich languages

### Comparison Table

| Metric | Focus | Semantic | Speed | Best For |
|--------|-------|----------|-------|----------|
| BLEU | Precision | No | Fast | Translation |
| ROUGE | Recall | No | Fast | Summarization |
| BERTScore | Similarity | Yes | Slow | General |
| METEOR | F1 | Partial | Medium | Translation |

---

## Task-Specific Metrics

### Classification Metrics

**Accuracy**: $\frac{\text{Correct}}{\text{Total}}$

**Precision**: $\frac{TP}{TP + FP}$

**Recall**: $\frac{TP}{TP + FN}$

**F1 Score**: $\frac{2 \cdot P \cdot R}{P + R}$

**Macro F1**: Average F1 across classes
**Micro F1**: Global TP/FP/FN, then compute F1

### Question Answering

**Exact Match (EM)**: Binary - is answer exactly correct?

**F1 Score**: Token-level overlap between prediction and gold

```python
def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()

    common = set(pred_tokens) & set(gold_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)
```

### Named Entity Recognition (NER)

**Entity-level F1**: Both boundary and type must match
**Token-level F1**: Each token scored independently

**Common Schemes**:
- BIO (Begin, Inside, Outside)
- BILOU (Begin, Inside, Last, Outside, Unit)

### Machine Translation

**BLEU**: Standard metric
**COMET**: Neural metric using cross-lingual embeddings
**chrF++**: Character F-score with word unigrams

### Summarization

**ROUGE-L**: Most common
**BERTScore**: For semantic faithfulness
**Factual Consistency**: Does summary contain only facts from source?

---

## LLM Benchmarks

### General Knowledge

**MMLU (Massive Multitask Language Understanding)**
- 57 subjects from STEM to humanities
- Multiple choice (4 options)
- Tests: World knowledge + reasoning
- **State-of-art**: GPT-4 ~87%, Human expert ~89%

**ARC (AI2 Reasoning Challenge)**
- Science questions from grade school
- Easy set + Challenge set
- **Focus**: Multi-step reasoning

**HellaSwag**
- Sentence completion task
- **Focus**: Commonsense reasoning
- Human: 95%, GPT-4: ~95%

### Reasoning

**GSM8K (Grade School Math)**
- 8.5K math word problems
- Tests: Multi-step reasoning, arithmetic
- **Challenge**: Chain-of-thought needed

```
Problem: If Maria has 3 apples and buys 2 more bags with
4 apples each, how many apples does she have?

Answer: 3 + (2 × 4) = 11 apples
```

**BBH (BIG-Bench Hard)**
- 23 challenging tasks from BIG-Bench
- Tests: Complex reasoning, symbolic manipulation
- **Includes**: Boolean expressions, navigation, etc.

**MATH**
- Competition mathematics problems
- 7 difficulty levels
- Tests: Mathematical reasoning

### Code Generation

**HumanEval**
- 164 Python programming problems
- Metric: pass@k (passes test cases)
- **Challenge**: Functional correctness

**MBPP (Mostly Basic Python Problems)**
- 974 programming tasks
- Simpler than HumanEval
- Entry-level programming

### Safety & Alignment

**TruthfulQA**
- Tests: Resistance to generating falsehoods
- Common misconceptions and conspiracy theories
- **Key**: Models often score worse than random!

**HHH (Helpful, Harmless, Honest)**
- Anthropic's alignment benchmark
- Measures: Helpfulness, harmlessness, honesty

**RealToxicityPrompts**
- Toxicity in open-ended generation
- 100K prompts designed to elicit toxic content

### Benchmark Aggregations

**HELM (Holistic Evaluation of Language Models)**
- Stanford's comprehensive evaluation
- Multiple metrics per scenario
- Calibration, robustness, fairness

**Open LLM Leaderboard (Hugging Face)**
- ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K
- Standardized evaluation
- Community submissions

### Benchmark Leaderboard Example (2024)

| Model | MMLU | GSM8K | HumanEval | HellaSwag |
|-------|------|-------|-----------|-----------|
| GPT-4 | 86.4 | 92.0 | 67.0 | 95.3 |
| Claude 3 Opus | 86.8 | 95.0 | 84.9 | - |
| LLaMA 3 70B | 82.0 | 93.0 | 81.7 | 88.0 |
| Mistral Large | 81.2 | 91.2 | - | 89.2 |

*Values approximate and evolving rapidly*

---

## LLM-as-Judge Evaluation

### The Paradigm Shift

**Problem**: Human evaluation is expensive and slow.
**Solution**: Use a strong LLM to judge outputs.

### Common Approaches

**1. Direct Scoring**
```
Prompt: "Rate this response on a scale of 1-10 for
helpfulness, accuracy, and clarity."

Response: "The capital of France is Paris, located
on the Seine River..."

Output: Helpfulness: 8, Accuracy: 10, Clarity: 9
```

**2. Pairwise Comparison**
```
Prompt: "Which response is better? A or B?"

Response A: [Model 1 output]
Response B: [Model 2 output]

Output: "Response A is better because..."
```

**3. Reference-Guided**
```
Prompt: "Compare this summary to the reference.
Rate faithfulness and completeness."

Reference: [Gold summary]
Generated: [Model output]
```

### Implementation Example

```python
def llm_judge_score(response: str, criteria: str, judge_model) -> dict:
    prompt = f"""
    Evaluate the following response on {criteria}.

    Response: {response}

    Provide scores from 1-10 for each criterion with brief justification.
    Format: {{"criterion": score, "reason": "..."}}
    """

    judgment = judge_model.generate(prompt)
    return parse_scores(judgment)
```

### Challenges

1. **Position Bias**: First response often preferred
2. **Length Bias**: Longer = better (not always true)
3. **Self-Preference**: Models prefer own outputs
4. **Sycophancy**: Agreeing with user even when wrong

### Mitigations

- **Swap positions**: Average A-B and B-A comparisons
- **Multiple judges**: Ensemble of different models
- **Calibration**: Use known good/bad examples
- **Structured rubrics**: Detailed scoring criteria

### Popular Judge Models

- GPT-4 (most common)
- Claude (good for nuanced evaluation)
- Prometheus (open-source, trained for judging)

---

## Human Evaluation

### When It's Necessary

- Final quality assessment
- Safety-critical applications
- Novel tasks without automatic metrics
- Validating automatic metrics

### Evaluation Designs

**1. Likert Scale Ratings**
```
Rate this response (1-5):
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent
```

**2. Pairwise Comparison (Elo Rating)**
```
Which is better?
[ ] Response A
[ ] Response B
[ ] Tie
```

Used by: Chatbot Arena, LMSYS

**3. Best-of-N Ranking**
```
Rank these responses from best to worst:
1. ___
2. ___
3. ___
```

### Quality Dimensions

| Dimension | Definition |
|-----------|------------|
| Fluency | Grammatically correct, natural |
| Coherence | Logically connected, on-topic |
| Relevance | Addresses the query |
| Factuality | Accurate information |
| Helpfulness | Actually useful to user |
| Harmlessness | No toxic/harmful content |

### Inter-Annotator Agreement

**Cohen's Kappa**:
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

- $\kappa > 0.8$: Almost perfect agreement
- $\kappa = 0.6-0.8$: Substantial
- $\kappa < 0.4$: Poor

### Crowdsourcing Platforms

- Amazon Mechanical Turk
- Scale AI
- Surge AI
- Prolific

### Best Practices

1. **Clear guidelines**: Detailed instructions with examples
2. **Qualification tests**: Filter low-quality annotators
3. **Attention checks**: Catch random clicking
4. **Multiple annotators**: 3-5 per example
5. **Pilot studies**: Test guidelines before scaling

---

## Practical Evaluation Strategies

### Development Workflow

```
1. Unit Tests
   └─ Specific behaviors (no hallucination on X)

2. Automatic Metrics
   └─ Fast feedback during development

3. LLM-as-Judge
   └─ Scalable quality assessment

4. Human Evaluation
   └─ Final validation, small sample
```

### Evaluation Sets

**Test Set Design**:
- Representative of real usage
- Diverse difficulty levels
- Include edge cases
- Stratified by category

**Sizes**:
- Development: 100-500 examples
- Test: 500-2000 examples
- Human eval: 50-200 examples

### Red Teaming

Actively trying to make the model fail:
- Jailbreaks
- Prompt injection
- Factual errors
- Harmful outputs

### Contamination Detection

**Problem**: Test data in training data.
**Detection**:
- Canary strings
- Membership inference
- Performance on variants

### Ablation Studies

Compare:
- With/without feature X
- Different model sizes
- Various hyperparameters

### Statistical Significance

Use bootstrap confidence intervals:
```python
def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, len(scores), replace=True)
        bootstrapped.append(np.mean(sample))

    lower = np.percentile(bootstrapped, (1-ci)/2 * 100)
    upper = np.percentile(bootstrapped, (1+ci)/2 * 100)
    return lower, upper
```

---

## Interview Questions

### Conceptual

**Q1: Why is perplexity alone insufficient for LLM evaluation?**

Perplexity measures prediction ability on held-out data but:
1. Doesn't capture task performance
2. Vocabulary/context dependent
3. Doesn't measure factuality, safety, or helpfulness
4. Can be gamed by memorization

**Q2: How would you evaluate a summarization model?**

Multi-pronged approach:
1. **ROUGE scores**: Quick, automated baseline
2. **BERTScore**: Semantic similarity
3. **Factual consistency**: NLI-based or QA-based check
4. **Human eval**: Fluency, coherence, informativeness
5. **LLM-as-judge**: Scalable quality assessment

**Q3: What is position bias in LLM-as-judge evaluation?**

When comparing two responses A and B, judges tend to prefer whichever is presented first (or sometimes second). Mitigation: Run both A-B and B-A orderings, average results.

**Q4: How do you detect benchmark contamination?**

1. **Canary strings**: Unique markers in test data
2. **Membership inference**: Check if model "remembers" exact examples
3. **Performance on variants**: Original vs paraphrased performance gap
4. **Training data search**: Check for overlap

### Coding

**Q5: Implement BLEU-4 from scratch.**

```python
from collections import Counter
import math

def compute_bleu(candidate: str, references: List[str], max_n: int = 4) -> float:
    """Compute BLEU score."""
    cand_tokens = candidate.lower().split()

    # Collect reference n-grams
    ref_ngrams = []
    for ref in references:
        ref_ngrams.append([ref.lower().split() for n in range(1, max_n + 1)])

    # Compute modified precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(zip(*[cand_tokens[i:] for i in range(n)]))

        max_ref_counts = Counter()
        for ref in references:
            ref_tokens = ref.lower().split()
            ref_ngram_counts = Counter(zip(*[ref_tokens[i:] for i in range(n)]))
            for ngram in ref_ngram_counts:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngram_counts[ngram])

        clipped = sum(min(cand_ngrams[ng], max_ref_counts[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())

        precisions.append(clipped / total if total > 0 else 0)

    # Geometric mean
    if any(p == 0 for p in precisions):
        return 0.0
    log_precision = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    ref_lens = [len(ref.split()) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - len(cand_tokens)))
    bp = 1 if len(cand_tokens) >= closest_ref_len else \
         math.exp(1 - closest_ref_len / len(cand_tokens))

    return bp * math.exp(log_precision)
```

**Q6: Design an LLM-as-judge evaluation pipeline.**

```python
class LLMJudge:
    def __init__(self, judge_model, criteria: List[str]):
        self.judge = judge_model
        self.criteria = criteria

    def score_response(self, query: str, response: str) -> Dict[str, float]:
        prompt = self._build_prompt(query, response)
        judgment = self.judge.generate(prompt)
        return self._parse_scores(judgment)

    def compare_responses(self, query: str, response_a: str, response_b: str) -> str:
        # Run both orderings
        ab_result = self._compare(query, response_a, response_b)
        ba_result = self._compare(query, response_b, response_a)

        # Aggregate
        return self._aggregate_comparisons(ab_result, ba_result)

    def _build_prompt(self, query: str, response: str) -> str:
        return f"""
        Evaluate this response to the query.

        Query: {query}
        Response: {response}

        Score each criterion 1-10:
        {', '.join(self.criteria)}

        Output JSON: {{"criterion": score, ...}}
        """
```

### System Design

**Q7: Design an evaluation pipeline for a production chatbot.**

```
Architecture:
┌─────────────────────────────────────────────────────┐
│                   Data Collection                   │
│  - Logged conversations (with consent)              │
│  - Synthetic test cases                             │
│  - Red team examples                                │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  Automatic Metrics                  │
│  - Latency, throughput                              │
│  - Safety classifier scores                         │
│  - Hallucination detection                          │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                   LLM-as-Judge                      │
│  - Helpfulness scoring                              │
│  - Factual accuracy                                 │
│  - Comparison to baseline model                     │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  Human Evaluation                   │
│  - Sample 5% for human review                       │
│  - Focus on failures and edge cases                 │
│  - Elo ratings via A/B testing                      │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│               Dashboard & Alerting                  │
│  - Track metrics over time                          │
│  - Alert on regressions                             │
│  - A/B test new models before deployment            │
└─────────────────────────────────────────────────────┘
```

---

## Summary

| Category | Key Metrics | Use Case |
|----------|------------|----------|
| Intrinsic | Perplexity, BPB | Model comparison (same tokenizer) |
| Generation | BLEU, ROUGE, BERTScore | Translation, summarization |
| Classification | Accuracy, F1 | Sentiment, NER, etc. |
| QA | EM, F1 | Question answering |
| Benchmarks | MMLU, GSM8K, HumanEval | General capability assessment |
| LLM-as-Judge | GPT-4 scores, pairwise | Scalable quality evaluation |
| Human | Likert, Elo | Final validation |

### Key Takeaways

1. **No single metric suffices** - use multiple perspectives
2. **Automatic metrics correlate imperfectly** with human judgment
3. **Benchmarks are gameable** - real-world performance may differ
4. **LLM-as-judge is powerful** but has biases
5. **Human evaluation is gold standard** but expensive
6. **Contamination is a real threat** - monitor carefully

---

## References

1. [BLEU: A Method for Automatic Evaluation](https://aclanthology.org/P02-1040/) - Papineni et al., 2002
2. [ROUGE: A Package for Automatic Evaluation](https://aclanthology.org/W04-1013/) - Lin, 2004
3. [BERTScore](https://arxiv.org/abs/1904.09675) - Zhang et al., 2019
4. [MMLU Benchmark](https://arxiv.org/abs/2009.03300) - Hendrycks et al., 2020
5. [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) - HELM Paper
6. [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) - Zheng et al., 2023
