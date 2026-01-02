# Module 5.1: Word Embeddings - From One-Hot to Word2Vec

## Table of Contents
1. [The Representation Problem](#1-the-representation-problem)
2. [One-Hot Encoding](#2-one-hot-encoding)
3. [Distributed Representations](#3-distributed-representations)
4. [Word2Vec](#4-word2vec)
5. [Training Word Embeddings](#5-training-word-embeddings)
6. [Embedding Properties](#6-embedding-properties)
7. [Modern Embeddings](#7-modern-embeddings)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. The Representation Problem

### 1.1 Why Representation Matters

Neural networks operate on numbers, not words. We need to convert words to numerical representations.

**Goals for word representations**:
1. Capture semantic meaning
2. Preserve relationships (king-queen ≈ man-woman)
3. Enable mathematical operations
4. Be efficient computationally

### 1.2 Evolution of Word Representations

```
1. One-hot encoding (sparse, no semantics)
   ↓
2. TF-IDF (statistical, document-level)
   ↓
3. Word2Vec/GloVe (dense, semantic)
   ↓
4. Contextual (BERT, GPT - different per context)
```

---

## 2. One-Hot Encoding

### 2.1 Definition

Each word = vector of vocabulary size with single 1:

```
Vocabulary: [the, cat, sat, on, mat]

"the" → [1, 0, 0, 0, 0]
"cat" → [0, 1, 0, 0, 0]
"sat" → [0, 0, 1, 0, 0]
"on"  → [0, 0, 0, 1, 0]
"mat" → [0, 0, 0, 0, 1]
```

### 2.2 Problems with One-Hot

| Problem | Explanation |
|---------|-------------|
| **High dimensionality** | Vector size = vocabulary size (50K+) |
| **Sparse** | Mostly zeros, inefficient |
| **No semantics** | All words equally distant |
| **No similarity** | cos(cat, dog) = cos(cat, car) = 0 |

### 2.3 Why One-Hot Fails

**Example**: "cat" and "dog" have no more in common than "cat" and "democracy":

```
Cosine similarity:
  sim(cat, dog) = 0  (orthogonal)
  sim(cat, car) = 0  (orthogonal)

All one-hot vectors are orthogonal!
```

---

## 3. Distributed Representations

### 3.1 The Key Insight

Represent words as **dense vectors** in lower-dimensional space where:
- Similar words have similar vectors
- Relationships are preserved as directions
- Semantic meaning encoded in continuous space

### 3.2 Distributional Hypothesis

**"You shall know a word by the company it keeps"** - J.R. Firth

Words appearing in similar contexts have similar meanings:
```
"The cat sat on the mat"
"The dog sat on the mat"
→ "cat" and "dog" appear in similar contexts
→ Should have similar representations
```

### 3.3 Dense vs Sparse

```
One-Hot (Sparse):
  "cat" → [0, 0, 1, 0, 0, ...] (50,000 dimensions)

Dense Embedding:
  "cat" → [0.2, -0.4, 0.1, 0.8, -0.3, ...] (300 dimensions)
```

**Benefits of dense**:
- Much smaller (300 vs 50,000)
- Captures similarity (nearby vectors = similar words)
- Generalizes better (similar words share features)

---

## 4. Word2Vec

### 4.1 Overview

Word2Vec (Mikolov et al., 2013) learns word embeddings by predicting:
- **Skip-gram**: Predict context words from center word
- **CBOW**: Predict center word from context words

### 4.2 Skip-gram Model

**Idea**: Given center word, predict surrounding context words.

```
Sentence: "The quick brown fox jumps"
Window size: 2

Center: "brown"
Context: ["quick", "fox"]

Training pairs:
  (brown, quick)
  (brown, fox)
```

**Architecture**:
```
Input (one-hot) → Embedding Layer → Softmax → Output (probability distribution)
     |              |                           |
  [0,0,1,0]    W_embed (V×D)               P(word|center)
```

**Objective**: Maximize probability of context words given center word:
```
L = Σ_t Σ_{j∈context} log P(w_{t+j} | w_t)
```

### 4.3 CBOW (Continuous Bag of Words)

**Idea**: Predict center word from average of context words.

```
Context: ["quick", "fox"]
Predict: "brown"
```

**Architecture**:
```
Context words → Average embeddings → Softmax → Predict center
```

### 4.4 Skip-gram vs CBOW

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Predicts | Context from center | Center from context |
| Speed | Slower | Faster |
| Rare words | Better | Worse |
| Common use | More popular | Less common |

### 4.5 The Softmax Problem

Full softmax over vocabulary is expensive:
```
P(w_o | w_i) = exp(v'_{w_o} · v_{w_i}) / Σ_{w∈V} exp(v'_w · v_{w_i})

Denominator requires V computations (50K+ words)!
```

**Solutions**:
1. **Negative Sampling**: Sample few negative words instead of all
2. **Hierarchical Softmax**: Binary tree structure

---

## 5. Training Word Embeddings

### 5.1 Negative Sampling

Instead of full softmax, train binary classifier:
- Is (center, context) a real pair? (positive)
- Is (center, random) a real pair? (negative)

**Objective**:
```
L = log σ(v'_{w_o} · v_{w_i}) + Σ_{k=1}^K E_{w_k~P_n} [log σ(-v'_{w_k} · v_{w_i})]

where:
  K = number of negative samples (typically 5-20)
  P_n = noise distribution (typically unigram^0.75)
```

**Key idea**: Learn to distinguish real context pairs from random pairs.

### 5.2 Negative Sampling Distribution

Sample negatives according to unigram distribution raised to 0.75:

```
P_n(w) ∝ count(w)^0.75

Why 0.75?
  - Raises probability of rare words
  - Reduces dominance of very common words
  - Empirically works well
```

### 5.3 Subsampling Frequent Words

Very common words ("the", "a", "is") provide little information.

**Subsampling**: Discard frequent words with probability:
```
P(discard) = 1 - √(t / f(w))

where:
  f(w) = frequency of word w
  t = threshold (typically 1e-5)
```

High frequency words have higher discard probability.

### 5.4 Training Details

**Hyperparameters**:
```
Embedding dimension: 100-300
Window size: 5-10
Negative samples: 5-20
Minimum count: 5 (discard rare words)
Learning rate: 0.025 (decaying)
Iterations: 5-15 epochs
```

---

## 6. Embedding Properties

### 6.1 Semantic Relationships

Word2Vec captures semantic relationships as directions in space:

```
king - man + woman ≈ queen
paris - france + italy ≈ rome
```

**Why this works**:
- Gender direction: man → woman
- Same direction applies: king → queen
- Royal + gender shift = correct answer

### 6.2 Analogy Tasks

**Format**: A is to B as C is to ?
```
man : woman :: king : ?
→ Find: argmax_d cos(d, b - a + c)
→ Answer: queen
```

**Types of analogies**:
- Semantic: king:queen::man:woman
- Syntactic: walk:walked::run:?
- Geographic: paris:france::tokyo:?

### 6.3 Similarity and Clustering

**Similar words cluster together**:
```
animals: cat, dog, horse, elephant
colors: red, blue, green, yellow
verbs: run, walk, jump, swim
```

**Measure similarity**: Cosine similarity
```
sim(a, b) = (a · b) / (||a|| × ||b||)
```

### 6.4 Limitations

| Limitation | Explanation |
|------------|-------------|
| **One embedding per word** | "bank" (river) = "bank" (financial) |
| **No context** | Same embedding regardless of usage |
| **Out-of-vocabulary** | Can't handle unseen words |
| **Static** | Doesn't update with new data |

---

## 7. Modern Embeddings

### 7.1 GloVe (Global Vectors)

Combines:
- Global statistics (like LSA)
- Local context (like Word2Vec)

**Objective**: Weighted least squares on co-occurrence counts:
```
L = Σ_{i,j} f(X_{ij}) (w_i · w_j + b_i + b_j - log(X_{ij}))²

where X_{ij} = co-occurrence count
```

### 7.2 FastText

Extends Word2Vec with subword information:
```
"where" → "<where>" + "<wh" + "whe" + "her" + "ere" + "re>"

Embedding = average of n-gram embeddings
```

**Benefits**:
- Handles out-of-vocabulary words
- Better for morphologically rich languages
- Captures word structure (un-happy, re-play)

### 7.3 Contextual Embeddings (BERT, GPT)

Different embedding for each context:
```
"bank of the river" → embedding₁
"bank account" → embedding₂
```

**Key difference**: Embedding depends on entire sentence context.

This is the foundation for transformers - covered in later modules.

### 7.4 Comparison

| Method | Static/Contextual | Subwords | Year |
|--------|------------------|----------|------|
| Word2Vec | Static | No | 2013 |
| GloVe | Static | No | 2014 |
| FastText | Static | Yes | 2016 |
| ELMo | Contextual | No | 2018 |
| BERT | Contextual | Yes | 2018 |
| GPT | Contextual | Yes | 2018+ |

---

## 8. Interview Questions

### Q1: Explain the Skip-gram model in Word2Vec.

**Answer**:

**Goal**: Learn word embeddings by predicting context words from center word.

**Process**:
1. Take a center word from sentence
2. Define context window (e.g., ±2 words)
3. Create (center, context) training pairs
4. Train network to predict context from center

**Example**:
```
"The quick brown fox jumps"
Center: "brown", Window: 2
Pairs: (brown, quick), (brown, fox)
```

**Architecture**:
```
One-hot(center) → Embedding → Output layer → P(context|center)
```

**Key insight**: The embedding layer weights become the word vectors.

### Q2: What is negative sampling and why is it used?

**Answer**:

**Problem**: Softmax over entire vocabulary is expensive:
```
P(w|center) = exp(score) / Σ_all_words exp(score)
```
Computing denominator requires V calculations (50K+).

**Negative Sampling Solution**:
- Instead of full softmax, train binary classifier
- Positive: Real (center, context) pairs
- Negative: Random (center, random_word) pairs

**Objective**:
```
Maximize: log σ(v_context · v_center)           # Real pair
Minimize: log σ(v_negative · v_center)          # Fake pairs
```

**Sampling**:
- Sample K negative words (typically 5-20)
- Use unigram^0.75 distribution (boosts rare words)

### Q3: Explain the king-queen analogy property.

**Answer**:

Word2Vec captures relationships as **directions** in embedding space:

```
king - man + woman ≈ queen
```

**Why it works**:
1. The vector (king - man) captures "royalty without gender"
2. Adding woman adds "female gender"
3. Result is closest to "queen"

**Mathematical view**:
- man → woman defines a "gender" direction
- king → queen is parallel to this direction
- Vector arithmetic follows these parallel relationships

**Finding analogies**:
```
A : B :: C : ?
Answer = argmax_d cos(d, B - A + C)
```

### Q4: What are the limitations of Word2Vec?

**Answer**:

1. **One embedding per word**:
   - "bank" (river) and "bank" (financial) have same embedding
   - Can't handle polysemy

2. **Context-independent**:
   - Same embedding regardless of surrounding words
   - Misses contextual nuances

3. **Out-of-vocabulary**:
   - Can't handle words not seen during training
   - No way to embed new/rare words

4. **Fixed**:
   - Once trained, doesn't adapt to new data
   - Domain-specific usage not captured

5. **Window-based**:
   - Only captures local context
   - Long-range dependencies missed

**Modern solutions**: BERT, GPT use contextual embeddings that vary by context.

### Q5: How do contextual embeddings (BERT) differ from Word2Vec?

**Answer**:

| Aspect | Word2Vec | BERT/GPT |
|--------|----------|----------|
| Embedding | One per word | Different per context |
| Polysemy | Not handled | Handled |
| Training | Predict neighbors | Masked LM / Next token |
| Output | Fixed vector | Context-dependent |

**Word2Vec**:
```
embed("bank") = [0.1, 0.3, -0.2, ...]  # Always same
```

**BERT**:
```
embed("bank of river") = [0.1, 0.5, ...]
embed("bank account") = [0.3, -0.2, ...]  # Different!
```

**How BERT achieves this**:
- Processes entire sequence through transformer
- Each word attends to all other words
- Output embedding incorporates full context

---

## 9. Summary

### Quick Reference

| Method | Description | Key Feature |
|--------|-------------|-------------|
| One-hot | Sparse, vocabulary-sized | No semantics |
| Word2Vec | Dense, learned from context | Semantic similarity |
| GloVe | Dense, global statistics | Combines local + global |
| FastText | Dense, subword-based | Handles OOV |
| BERT/GPT | Contextual, transformer | Context-dependent |

### Key Equations

**Skip-gram Objective**:
```
L = Σ_t Σ_{j∈context} log P(w_{t+j} | w_t)
```

**Negative Sampling**:
```
L = log σ(v'_o · v_i) + Σ_k E[log σ(-v'_k · v_i)]
```

**Analogy**:
```
A : B :: C : ?
Answer = argmax_d cos(d, B - A + C)
```

### Key Takeaways

1. **Dense > Sparse**: Embeddings capture semantics
2. **Context predicts meaning**: Distributional hypothesis
3. **Relationships as directions**: king - man + woman = queen
4. **Negative sampling**: Makes training tractable
5. **Limitations**: Static embeddings miss context (BERT solves this)
