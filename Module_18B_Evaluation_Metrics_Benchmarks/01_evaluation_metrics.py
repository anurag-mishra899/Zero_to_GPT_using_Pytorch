"""
Evaluation Metrics & Benchmarks - Hands-on Implementation
==========================================================

This module covers:
1. Perplexity calculation (various methods)
2. BLEU, ROUGE, BERTScore from scratch and with libraries
3. Task-specific metrics (F1, EM, etc.)
4. LLM-as-judge evaluation framework
5. Benchmark evaluation pipelines
6. Statistical significance testing

Author: Zero to GPT Course
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
from dataclasses import dataclass
import math
import re
import numpy as np
from abc import ABC, abstractmethod
import json


# =============================================================================
# Section 1: Perplexity Metrics
# =============================================================================

def compute_perplexity_from_loss(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.

    PPL = exp(CE_loss)

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    return math.exp(loss)


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int = 0
) -> float:
    """
    Compute perplexity from model logits.

    Args:
        logits: (batch, seq_len, vocab_size) model predictions
        targets: (batch, seq_len) ground truth token ids
        pad_token_id: Token ID to ignore in calculation

    Returns:
        Perplexity value
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()

    # Compute per-token loss
    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_targets.view(-1),
        reduction='none',
        ignore_index=pad_token_id
    )

    # Reshape and compute mean over non-padding tokens
    loss = loss.view(shift_targets.shape)
    mask = (shift_targets != pad_token_id).float()
    total_loss = (loss * mask).sum()
    total_tokens = mask.sum()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss.item())


def compute_perplexity_sliding_window(
    model,
    input_ids: torch.Tensor,
    stride: int = 512,
    max_length: int = 1024,
    device: torch.device = torch.device('cpu')
) -> float:
    """
    Compute perplexity with sliding window for long sequences.

    This is the standard way to evaluate perplexity on documents
    longer than the model's context window.

    Args:
        model: Language model with forward method
        input_ids: (1, total_length) full document tokens
        stride: How much to slide window each step
        max_length: Model's maximum context length
        device: Compute device

    Returns:
        Perplexity over entire document
    """
    model.eval()
    model = model.to(device)

    seq_len = input_ids.size(1)
    nlls = []  # Negative log likelihoods
    prev_end_loc = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # Number of new tokens

            input_chunk = input_ids[:, begin_loc:end_loc].to(device)

            # Forward pass
            outputs = model(input_chunk)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_chunk[:, 1:].contiguous()

            # Only compute loss on NEW tokens (not overlap)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Take only the last trg_len-1 losses (new tokens)
            loss = loss.view(shift_labels.shape)
            if prev_end_loc > 0:
                loss = loss[:, -(trg_len-1):]

            nlls.append(loss.sum().item())
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

    total_nll = sum(nlls)
    total_tokens = prev_end_loc - 1  # -1 because we predict next token
    ppl = math.exp(total_nll / total_tokens)

    return ppl


def bits_per_byte(perplexity: float, chars_per_token: float = 4.0) -> float:
    """
    Convert perplexity to bits per byte.

    More comparable across tokenizers.

    Args:
        perplexity: Model perplexity
        chars_per_token: Average characters per token (GPT-2 ~ 4)

    Returns:
        Bits per byte
    """
    bits_per_token = math.log2(perplexity)
    return bits_per_token / chars_per_token


# =============================================================================
# Section 2: BLEU Score
# =============================================================================

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Extract n-grams from token list.

    Args:
        tokens: List of tokens
        n: N-gram size

    Returns:
        Counter of n-gram tuples
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def compute_bleu(
    candidate: str,
    references: List[str],
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute BLEU score from scratch.

    BLEU = BP * exp(sum(w_n * log(p_n)))

    where:
    - BP = brevity penalty
    - p_n = modified n-gram precision
    - w_n = weights (typically uniform 1/n)

    Args:
        candidate: Generated text
        references: List of reference texts
        max_n: Maximum n-gram order (default 4 for BLEU-4)
        weights: Weights for each n-gram order

    Returns:
        Dict with 'bleu', 'precisions', 'bp', 'ratio'
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Tokenize
    cand_tokens = candidate.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]

    # Compute modified precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(cand_tokens, n)

        # Get max count for each n-gram across references
        max_ref_counts = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(
                    max_ref_counts[ngram],
                    ref_ngrams[ngram]
                )

        # Compute clipped count
        clipped_count = 0
        total_count = 0
        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, max_ref_counts.get(ngram, 0))
            total_count += count

        # Precision for this n
        if total_count > 0:
            precision = clipped_count / total_count
        else:
            precision = 0.0

        precisions.append(precision)

    # Brevity penalty
    cand_len = len(cand_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]

    # Find closest reference length
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - cand_len), x))

    if cand_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / cand_len)

    # Compute BLEU
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
        bleu = bp * math.exp(sum(log_precisions))

    return {
        'bleu': bleu,
        'precisions': precisions,
        'bp': bp,
        'ratio': cand_len / closest_ref_len,
        'candidate_len': cand_len,
        'reference_len': closest_ref_len
    }


def corpus_bleu(
    candidates: List[str],
    references_list: List[List[str]],
    max_n: int = 4
) -> float:
    """
    Compute corpus-level BLEU score.

    Aggregates counts across all sentences before computing precision.

    Args:
        candidates: List of generated texts
        references_list: List of reference lists for each candidate

    Returns:
        Corpus BLEU score
    """
    total_clipped = [0] * max_n
    total_count = [0] * max_n
    total_cand_len = 0
    total_ref_len = 0

    for candidate, references in zip(candidates, references_list):
        cand_tokens = candidate.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in references]

        total_cand_len += len(cand_tokens)

        # Closest reference length
        ref_lens = [len(ref) for ref in ref_tokens_list]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - len(cand_tokens)))
        total_ref_len += closest_ref_len

        for n in range(1, max_n + 1):
            cand_ngrams = get_ngrams(cand_tokens, n)

            max_ref_counts = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams = get_ngrams(ref_tokens, n)
                for ngram in ref_ngrams:
                    max_ref_counts[ngram] = max(
                        max_ref_counts[ngram],
                        ref_ngrams[ngram]
                    )

            for ngram, count in cand_ngrams.items():
                total_clipped[n-1] += min(count, max_ref_counts.get(ngram, 0))
                total_count[n-1] += count

    # Compute precisions
    precisions = []
    for n in range(max_n):
        if total_count[n] > 0:
            precisions.append(total_clipped[n] / total_count[n])
        else:
            precisions.append(0.0)

    # Brevity penalty
    if total_cand_len >= total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_cand_len)

    # Final score
    if any(p == 0 for p in precisions):
        return 0.0

    log_precisions = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_precisions)


# =============================================================================
# Section 3: ROUGE Score
# =============================================================================

def compute_rouge_n(
    candidate: str,
    reference: str,
    n: int = 1
) -> Dict[str, float]:
    """
    Compute ROUGE-N score.

    ROUGE-N measures n-gram recall between candidate and reference.

    Args:
        candidate: Generated summary
        reference: Reference summary
        n: N-gram order

    Returns:
        Dict with 'precision', 'recall', 'f1'
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    # Count overlapping n-grams
    overlap = 0
    for ngram in cand_ngrams:
        overlap += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))

    # Compute metrics
    total_cand = sum(cand_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    precision = overlap / total_cand if total_cand > 0 else 0.0
    recall = overlap / total_ref if total_ref > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def lcs_length(x: List[str], y: List[str]) -> int:
    """
    Compute length of longest common subsequence.

    Args:
        x: First sequence
        y: Second sequence

    Returns:
        LCS length
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def compute_rouge_l(
    candidate: str,
    reference: str
) -> Dict[str, float]:
    """
    Compute ROUGE-L score using Longest Common Subsequence.

    ROUGE-L captures sentence-level structure similarity.

    Args:
        candidate: Generated text
        reference: Reference text

    Returns:
        Dict with 'precision', 'recall', 'f1'
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    lcs_len = lcs_length(cand_tokens, ref_tokens)

    precision = lcs_len / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall > 0:
        # Using beta=1 for F1 (equal weight to P and R)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_rouge_scores(
    candidate: str,
    reference: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute all common ROUGE variants.

    Args:
        candidate: Generated text
        reference: Reference text

    Returns:
        Dict with 'rouge1', 'rouge2', 'rougeL' scores
    """
    return {
        'rouge1': compute_rouge_n(candidate, reference, n=1),
        'rouge2': compute_rouge_n(candidate, reference, n=2),
        'rougeL': compute_rouge_l(candidate, reference)
    }


# =============================================================================
# Section 4: BERTScore (Simplified)
# =============================================================================

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.

    Args:
        x: (n, d) tensor
        y: (m, d) tensor

    Returns:
        (n, m) similarity matrix
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return torch.mm(x_norm, y_norm.t())


class SimpleBERTScore:
    """
    Simplified BERTScore implementation.

    In practice, use the `bert_score` library for full implementation.
    This demonstrates the core concept.
    """

    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: Model that returns token embeddings
                            (or None for random embeddings demo)
        """
        self.model = embedding_model

    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        Get token embeddings for text.

        In production, use a real model like BERT.
        """
        if self.model is not None:
            # Use actual model
            return self.model.encode(text)
        else:
            # Demo: random embeddings
            tokens = text.lower().split()
            return torch.randn(len(tokens), 768)

    def compute_score(
        self,
        candidate: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Compute BERTScore.

        BERTScore = greedy matching between embeddings.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            Dict with 'precision', 'recall', 'f1'
        """
        cand_emb = self.get_embeddings(candidate)  # (n, d)
        ref_emb = self.get_embeddings(reference)   # (m, d)

        # Compute similarity matrix
        sim_matrix = cosine_similarity(cand_emb, ref_emb)  # (n, m)

        # Greedy matching
        # Precision: for each candidate token, find best match in reference
        precision = sim_matrix.max(dim=1).values.mean().item()

        # Recall: for each reference token, find best match in candidate
        recall = sim_matrix.max(dim=0).values.mean().item()

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# =============================================================================
# Section 5: Task-Specific Metrics
# =============================================================================

def exact_match(prediction: str, ground_truth: str) -> int:
    """
    Exact match for QA tasks.

    Returns 1 if normalized prediction matches ground truth, 0 otherwise.
    """
    def normalize(text: str) -> str:
        # Lowercase, remove punctuation and extra whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    return int(normalize(prediction) == normalize(ground_truth))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score for QA tasks.

    Used in SQuAD and similar benchmarks.

    Args:
        prediction: Model's answer
        ground_truth: Gold answer

    Returns:
        F1 score
    """
    def normalize_and_tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    pred_tokens = normalize_and_tokenize(prediction)
    gold_tokens = normalize_and_tokenize(ground_truth)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def classification_metrics(
    predictions: List[int],
    labels: List[int],
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        num_classes: Number of classes (inferred if None)

    Returns:
        Dict with accuracy, macro_f1, per-class metrics
    """
    if num_classes is None:
        num_classes = max(max(predictions), max(labels)) + 1

    # Compute confusion matrix
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(predictions, labels):
        confusion[label][pred] += 1

    # Compute per-class metrics
    class_metrics = []
    for c in range(num_classes):
        tp = confusion[c][c]
        fp = sum(confusion[i][c] for i in range(num_classes)) - tp
        fn = sum(confusion[c][i] for i in range(num_classes)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics.append({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Aggregate metrics
    accuracy = sum(confusion[i][i] for i in range(num_classes)) / len(predictions)
    macro_precision = sum(m['precision'] for m in class_metrics) / num_classes
    macro_recall = sum(m['recall'] for m in class_metrics) / num_classes
    macro_f1 = sum(m['f1'] for m in class_metrics) / num_classes

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': class_metrics
    }


# =============================================================================
# Section 6: LLM-as-Judge Framework
# =============================================================================

@dataclass
class JudgmentResult:
    """Result from LLM judge evaluation."""
    scores: Dict[str, float]
    reasoning: str
    raw_response: str


class LLMJudge(ABC):
    """Abstract base class for LLM-as-judge evaluation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from judge model."""
        pass

    def score_response(
        self,
        query: str,
        response: str,
        criteria: List[str] = None
    ) -> JudgmentResult:
        """
        Score a single response on given criteria.

        Args:
            query: Original user query
            response: Model's response to evaluate
            criteria: List of criteria to score (default: helpfulness, accuracy, clarity)

        Returns:
            JudgmentResult with scores and reasoning
        """
        if criteria is None:
            criteria = ['helpfulness', 'accuracy', 'clarity']

        prompt = self._build_scoring_prompt(query, response, criteria)
        raw_response = self.generate(prompt)
        scores, reasoning = self._parse_scores(raw_response, criteria)

        return JudgmentResult(
            scores=scores,
            reasoning=reasoning,
            raw_response=raw_response
        )

    def compare_responses(
        self,
        query: str,
        response_a: str,
        response_b: str,
        swap_positions: bool = True
    ) -> Dict[str, Union[str, float]]:
        """
        Compare two responses using pairwise comparison.

        Args:
            query: Original user query
            response_a: First response
            response_b: Second response
            swap_positions: Also run B vs A to mitigate position bias

        Returns:
            Dict with winner and confidence
        """
        # Run A vs B
        result_ab = self._single_comparison(query, response_a, response_b, "A", "B")

        if swap_positions:
            # Run B vs A
            result_ba = self._single_comparison(query, response_b, response_a, "B", "A")
            return self._aggregate_comparisons(result_ab, result_ba)

        return result_ab

    def _build_scoring_prompt(
        self,
        query: str,
        response: str,
        criteria: List[str]
    ) -> str:
        """Build prompt for scoring evaluation."""
        criteria_str = '\n'.join([f"- {c}: Rate 1-10" for c in criteria])

        return f"""You are an expert evaluator. Rate the following response to the query.

Query: {query}

Response: {response}

Evaluate on these criteria:
{criteria_str}

Provide your evaluation in JSON format:
{{
    "scores": {{"criterion": score, ...}},
    "reasoning": "Brief explanation of scores"
}}

Be objective and critical. Only output the JSON."""

    def _parse_scores(
        self,
        response: str,
        criteria: List[str]
    ) -> Tuple[Dict[str, float], str]:
        """Parse scores from judge response."""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                scores = {c: float(data.get('scores', {}).get(c, 5)) for c in criteria}
                reasoning = data.get('reasoning', '')
                return scores, reasoning
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: return default scores
        return {c: 5.0 for c in criteria}, "Failed to parse response"

    def _single_comparison(
        self,
        query: str,
        response_1: str,
        response_2: str,
        label_1: str,
        label_2: str
    ) -> Dict[str, Union[str, float]]:
        """Run single pairwise comparison."""
        prompt = f"""Compare these two responses to the query.

Query: {query}

Response {label_1}:
{response_1}

Response {label_2}:
{response_2}

Which response is better overall? Consider helpfulness, accuracy, and clarity.

Output ONLY one of: "{label_1}", "{label_2}", or "TIE"
Then briefly explain why."""

        result = self.generate(prompt)

        # Parse winner
        if label_1 in result.upper()[:20]:
            winner = label_1
        elif label_2 in result.upper()[:20]:
            winner = label_2
        else:
            winner = "TIE"

        return {
            'winner': winner,
            'reasoning': result
        }

    def _aggregate_comparisons(
        self,
        result_ab: Dict,
        result_ba: Dict
    ) -> Dict[str, Union[str, float]]:
        """Aggregate results from swapped comparisons."""
        # Map results to consistent labels
        ab_winner = result_ab['winner']
        # In BA comparison, "B" means original A won, "A" means original B won
        ba_winner = 'A' if result_ba['winner'] == 'B' else ('B' if result_ba['winner'] == 'A' else 'TIE')

        # Determine final winner
        if ab_winner == ba_winner:
            final_winner = ab_winner
            confidence = 1.0
        elif ab_winner == 'TIE' or ba_winner == 'TIE':
            final_winner = ab_winner if ba_winner == 'TIE' else ba_winner
            confidence = 0.75
        else:
            final_winner = 'TIE'
            confidence = 0.5

        return {
            'winner': final_winner,
            'confidence': confidence,
            'ab_result': result_ab,
            'ba_result': result_ba
        }


class MockLLMJudge(LLMJudge):
    """Mock judge for demonstration (replace with real API in production)."""

    def generate(self, prompt: str) -> str:
        """Generate mock response."""
        # In production, call your LLM API here
        import random
        mock_scores = {
            'helpfulness': random.randint(6, 9),
            'accuracy': random.randint(7, 10),
            'clarity': random.randint(6, 9)
        }
        return json.dumps({
            'scores': mock_scores,
            'reasoning': 'Mock evaluation for demonstration.'
        })


# =============================================================================
# Section 7: Benchmark Evaluation Pipeline
# =============================================================================

@dataclass
class BenchmarkExample:
    """Single benchmark example."""
    id: str
    input_text: str
    expected_output: str
    metadata: Optional[Dict] = None


@dataclass
class BenchmarkResult:
    """Result for a benchmark example."""
    example_id: str
    prediction: str
    score: float
    details: Optional[Dict] = None


class BenchmarkEvaluator:
    """
    Framework for evaluating models on benchmarks.

    Supports multiple-choice, generation, and classification tasks.
    """

    def __init__(
        self,
        model_fn: Callable[[str], str],
        metric_fn: Callable[[str, str], float]
    ):
        """
        Args:
            model_fn: Function that takes input and returns prediction
            metric_fn: Function that takes (prediction, expected) and returns score
        """
        self.model_fn = model_fn
        self.metric_fn = metric_fn

    def evaluate_example(self, example: BenchmarkExample) -> BenchmarkResult:
        """Evaluate single example."""
        prediction = self.model_fn(example.input_text)
        score = self.metric_fn(prediction, example.expected_output)

        return BenchmarkResult(
            example_id=example.id,
            prediction=prediction,
            score=score,
            details={'expected': example.expected_output}
        )

    def evaluate_benchmark(
        self,
        examples: List[BenchmarkExample],
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate full benchmark.

        Args:
            examples: List of benchmark examples
            verbose: Print progress

        Returns:
            Dict with aggregate scores
        """
        results = []

        for i, example in enumerate(examples):
            result = self.evaluate_example(example)
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                avg_score = sum(r.score for r in results) / len(results)
                print(f"Progress: {i+1}/{len(examples)}, Avg Score: {avg_score:.4f}")

        # Aggregate
        scores = [r.score for r in results]
        return {
            'mean': sum(scores) / len(scores),
            'std': np.std(scores),
            'min': min(scores),
            'max': max(scores),
            'num_examples': len(examples),
            'results': results
        }


class MultipleChoiceEvaluator(BenchmarkEvaluator):
    """Evaluator for multiple-choice benchmarks like MMLU."""

    def __init__(self, model_fn: Callable[[str], str], choices: List[str] = None):
        if choices is None:
            choices = ['A', 'B', 'C', 'D']
        self.choices = choices

        def mc_metric(pred: str, expected: str) -> float:
            # Extract choice letter from prediction
            pred_upper = pred.upper().strip()
            for choice in self.choices:
                if pred_upper.startswith(choice):
                    return 1.0 if choice == expected.upper() else 0.0
            return 0.0

        super().__init__(model_fn, mc_metric)


# =============================================================================
# Section 8: Statistical Significance
# =============================================================================

def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean score.

    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    scores = np.array(scores)
    bootstrapped_means = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped_means.append(np.mean(sample))

    # Compute percentiles
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrapped_means, alpha * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha) * 100)
    mean = np.mean(scores)

    return mean, lower, upper


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000
) -> float:
    """
    Paired bootstrap test for comparing two systems.

    Tests if system A is significantly better than system B.

    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B (same examples)
        n_bootstrap: Number of bootstrap iterations

    Returns:
        p-value (probability that A is not better than B)
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diff = scores_a - scores_b

    observed_diff = np.mean(diff)
    count = 0

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(diff, size=len(diff), replace=True)
        if np.mean(sample) <= 0:  # A not better than B
            count += 1

    return count / n_bootstrap


def cohens_kappa(
    annotations_1: List[int],
    annotations_2: List[int]
) -> float:
    """
    Compute Cohen's Kappa for inter-annotator agreement.

    Args:
        annotations_1: First annotator's labels
        annotations_2: Second annotator's labels

    Returns:
        Kappa coefficient (-1 to 1, higher is better agreement)
    """
    assert len(annotations_1) == len(annotations_2)

    n = len(annotations_1)
    labels = set(annotations_1) | set(annotations_2)

    # Observed agreement
    observed_agreement = sum(a == b for a, b in zip(annotations_1, annotations_2)) / n

    # Expected agreement (by chance)
    expected_agreement = 0
    for label in labels:
        p1 = sum(a == label for a in annotations_1) / n
        p2 = sum(a == label for a in annotations_2) / n
        expected_agreement += p1 * p2

    # Kappa
    if expected_agreement == 1:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


# =============================================================================
# Section 9: Demonstrations
# =============================================================================

def demo_bleu_rouge():
    """Demonstrate BLEU and ROUGE calculation."""
    print("=" * 60)
    print("Demo: BLEU and ROUGE Scores")
    print("=" * 60)

    candidate = "The cat sat on the mat in the living room"
    references = [
        "The cat was sitting on the mat",
        "A cat sat on the mat in the room"
    ]

    # BLEU
    bleu_result = compute_bleu(candidate, references)
    print(f"\nCandidate: {candidate}")
    print(f"References: {references}")
    print(f"\nBLEU Score: {bleu_result['bleu']:.4f}")
    print(f"  Precisions: {[f'{p:.3f}' for p in bleu_result['precisions']]}")
    print(f"  Brevity Penalty: {bleu_result['bp']:.4f}")

    # ROUGE
    rouge_scores = compute_rouge_scores(candidate, references[0])
    print(f"\nROUGE Scores (vs first reference):")
    for name, scores in rouge_scores.items():
        print(f"  {name}: P={scores['precision']:.3f}, R={scores['recall']:.3f}, F1={scores['f1']:.3f}")


def demo_qa_metrics():
    """Demonstrate QA evaluation metrics."""
    print("\n" + "=" * 60)
    print("Demo: Question Answering Metrics")
    print("=" * 60)

    test_cases = [
        ("Paris", "Paris"),
        ("the Eiffel Tower", "Eiffel Tower"),
        ("It is located in Paris, France", "Paris"),
        ("I don't know", "Paris")
    ]

    print("\nPrediction vs Ground Truth:")
    for pred, gold in test_cases:
        em = exact_match(pred, gold)
        f1 = qa_f1_score(pred, gold)
        print(f"  '{pred}' vs '{gold}': EM={em}, F1={f1:.3f}")


def demo_llm_judge():
    """Demonstrate LLM-as-judge evaluation."""
    print("\n" + "=" * 60)
    print("Demo: LLM-as-Judge Evaluation")
    print("=" * 60)

    judge = MockLLMJudge()

    query = "What is the capital of France?"
    response = "The capital of France is Paris. It's located on the Seine River."

    result = judge.score_response(query, response)
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"\nJudge Scores:")
    for criterion, score in result.scores.items():
        print(f"  {criterion}: {score}")
    print(f"Reasoning: {result.reasoning}")


def demo_benchmark_evaluation():
    """Demonstrate benchmark evaluation pipeline."""
    print("\n" + "=" * 60)
    print("Demo: Benchmark Evaluation")
    print("=" * 60)

    # Create mock examples
    examples = [
        BenchmarkExample(
            id="1",
            input_text="Q: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6",
            expected_output="B"
        ),
        BenchmarkExample(
            id="2",
            input_text="Q: What is the capital of Japan?\nA) Seoul\nB) Beijing\nC) Tokyo\nD) Bangkok",
            expected_output="C"
        ),
        BenchmarkExample(
            id="3",
            input_text="Q: Which planet is closest to the sun?\nA) Venus\nB) Mercury\nC) Mars\nD) Earth",
            expected_output="B"
        ),
    ]

    # Mock model (random guessing)
    import random
    def mock_model(input_text: str) -> str:
        return random.choice(['A', 'B', 'C', 'D'])

    # Evaluate
    evaluator = MultipleChoiceEvaluator(mock_model)
    results = evaluator.evaluate_benchmark(examples)

    print(f"\nBenchmark Results (random baseline):")
    print(f"  Mean Accuracy: {results['mean']:.4f}")
    print(f"  Std: {results['std']:.4f}")
    print(f"  Examples: {results['num_examples']}")


def demo_statistical_tests():
    """Demonstrate statistical significance testing."""
    print("\n" + "=" * 60)
    print("Demo: Statistical Significance Tests")
    print("=" * 60)

    # Simulate scores from two systems
    np.random.seed(42)
    scores_a = np.random.normal(0.75, 0.1, 100).clip(0, 1)
    scores_b = np.random.normal(0.70, 0.1, 100).clip(0, 1)

    # Bootstrap confidence interval
    mean_a, lower_a, upper_a = bootstrap_confidence_interval(scores_a.tolist())
    mean_b, lower_b, upper_b = bootstrap_confidence_interval(scores_b.tolist())

    print(f"\nSystem A: {mean_a:.4f} [{lower_a:.4f}, {upper_a:.4f}]")
    print(f"System B: {mean_b:.4f} [{lower_b:.4f}, {upper_b:.4f}]")

    # Paired bootstrap test
    p_value = paired_bootstrap_test(scores_a.tolist(), scores_b.tolist())
    print(f"\nPaired Bootstrap Test (A > B):")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at 0.05? {'Yes' if p_value < 0.05 else 'No'}")

    # Cohen's Kappa
    ann1 = [1, 1, 2, 2, 1, 2, 1, 2, 1, 1]
    ann2 = [1, 2, 2, 2, 1, 2, 1, 1, 1, 1]
    kappa = cohens_kappa(ann1, ann2)
    print(f"\nInter-annotator Agreement:")
    print(f"  Annotator 1: {ann1}")
    print(f"  Annotator 2: {ann2}")
    print(f"  Cohen's Kappa: {kappa:.4f}")


def demo_perplexity():
    """Demonstrate perplexity calculation."""
    print("\n" + "=" * 60)
    print("Demo: Perplexity Metrics")
    print("=" * 60)

    # Simulate different loss values
    losses = [2.0, 3.0, 4.0, 5.0]

    print("\nPerplexity from Cross-Entropy Loss:")
    for loss in losses:
        ppl = compute_perplexity_from_loss(loss)
        bpb = bits_per_byte(ppl)
        print(f"  CE Loss={loss:.1f} -> PPL={ppl:.1f}, BPB={bpb:.2f}")


if __name__ == "__main__":
    print("Evaluation Metrics & Benchmarks - Hands-on Implementation")
    print("=" * 60)

    demo_perplexity()
    demo_bleu_rouge()
    demo_qa_metrics()
    demo_llm_judge()
    demo_benchmark_evaluation()
    demo_statistical_tests()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
