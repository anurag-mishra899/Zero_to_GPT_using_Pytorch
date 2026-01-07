# Module 8B: BERT & Encoder Models

## Table of Contents
1. [BERT Overview](#1-bert-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Pre-training Objectives](#3-pre-training-objectives)
4. [Fine-tuning BERT](#4-fine-tuning-bert)
5. [BERT Variants](#5-bert-variants)
6. [BERT vs GPT](#6-bert-vs-gpt)
7. [Practical Applications](#7-practical-applications)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. BERT Overview

### 1.1 What is BERT?

**BERT** = Bidirectional Encoder Representations from Transformers

BERT (Devlin et al., 2018) was a breakthrough in NLP that introduced:
- **Bidirectional context**: Each token sees ALL other tokens (not just left context)
- **Pre-training + Fine-tuning paradigm**: Train once on large corpus, fine-tune for specific tasks
- **Transfer learning for NLP**: Like ImageNet for vision

```
Pre-training:                    Fine-tuning:
Large unlabeled corpus           Small labeled dataset
      ↓                                ↓
   [BERT]  ---------------→     [BERT + Task Head]
      ↓                                ↓
Language understanding           Task-specific predictions
```

### 1.2 Why BERT Was Revolutionary

**Before BERT** (2018):
- Train models from scratch for each task
- Small models due to data limitations
- Word embeddings (Word2Vec) were static

**After BERT**:
- Pre-train once, fine-tune everywhere
- Transfer learning brought huge gains
- Contextual embeddings capture meaning

### 1.3 Key Innovations

| Innovation | Description | Impact |
|------------|-------------|--------|
| **Bidirectional** | See all tokens, not just left | Better understanding |
| **MLM** | Predict masked tokens | Learn context both directions |
| **NSP** | Predict sentence relationships | Understand document structure |
| **[CLS] token** | Special token for classification | Easy fine-tuning |

### 1.4 BERT Timeline

```
2018: BERT released
2019: RoBERTa (improved pre-training)
      ALBERT (parameter efficient)
      DistilBERT (smaller, faster)
2020: DeBERTa (disentangled attention)
      ELECTRA (replaced token detection)
```

---

## 2. Architecture Deep Dive

### 2.1 Overall Architecture

BERT is an **encoder-only transformer** (no decoder):

```
Input: [CLS] The cat sat [SEP] on the mat [SEP]
         ↓
   Token Embedding
         +
   Segment Embedding
         +
   Position Embedding
         ↓
   ┌─────────────────────┐
   │  Transformer Block  │ × 12 (base) or 24 (large)
   │  - Multi-head Attn  │
   │  - Add & Norm       │
   │  - FFN              │
   │  - Add & Norm       │
   └─────────────────────┘
         ↓
   Contextual Embeddings
         ↓
   Task-specific Head
```

### 2.2 Model Configurations

| Model | Layers | Hidden | Heads | Params |
|-------|--------|--------|-------|--------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

### 2.3 Input Representation

BERT's input is the sum of three embeddings:

```
Input:      [CLS]    The      cat     [SEP]    sat     [SEP]
              ↓       ↓        ↓        ↓       ↓        ↓
Token:      E_CLS   E_The   E_cat   E_SEP   E_sat   E_SEP
              +       +        +        +       +        +
Segment:    E_A     E_A      E_A     E_A     E_B     E_B
              +       +        +        +       +        +
Position:   E_0     E_1      E_2     E_3     E_4     E_5
              =       =        =        =       =        =
Final:      [...]   [...]   [...]   [...]   [...]   [...]
```

**Token Embeddings**: WordPiece vocabulary (30,522 tokens)
**Segment Embeddings**: Which sentence (A or B)
**Position Embeddings**: Learned absolute positions (0 to 511)

### 2.4 Special Tokens

| Token | Purpose | Position |
|-------|---------|----------|
| `[CLS]` | Classification token | Start of input |
| `[SEP]` | Separator | Between/after sentences |
| `[PAD]` | Padding | Fill to max length |
| `[MASK]` | Masked token | For MLM pre-training |
| `[UNK]` | Unknown token | OOV words |

### 2.5 Bidirectional Attention

Unlike GPT (causal attention), BERT uses **full attention**:

```
GPT (Causal):              BERT (Bidirectional):
Position 3 sees: 0,1,2,3   Position 3 sees: 0,1,2,3,4,5,...

Attention Matrix:          Attention Matrix:
[1 0 0 0 0]               [1 1 1 1 1]
[1 1 0 0 0]               [1 1 1 1 1]
[1 1 1 0 0]               [1 1 1 1 1]
[1 1 1 1 0]               [1 1 1 1 1]
[1 1 1 1 1]               [1 1 1 1 1]
```

**Why bidirectional matters**:
```
"The bank by the river"
"The bank handles money"

With bidirectional context:
  "bank" in sentence 1 → attends to "river" → financial institution unlikely
  "bank" in sentence 2 → attends to "money" → financial institution likely
```

### 2.6 Architecture Code

```python
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position, type_vocab_size=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = (self.word_embeddings(input_ids) +
                     self.position_embeddings(position_ids) +
                     self.token_type_embeddings(token_type_ids))

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
```

---

## 3. Pre-training Objectives

### 3.1 Masked Language Model (MLM)

**The key innovation**: Randomly mask tokens and predict them.

```
Input:  "The [MASK] sat on the [MASK]"
Target: "The  cat   sat on the  mat"

Model predicts: P(cat | context), P(mat | context)
```

**Masking strategy** (for 15% of tokens):
- 80%: Replace with [MASK]
- 10%: Replace with random token
- 10%: Keep unchanged

**Why this mixture?**
- 80% [MASK]: Primary signal for learning
- 10% random: Prevent model from ignoring non-[MASK] tokens
- 10% unchanged: Fine-tuning doesn't have [MASK], need robustness

### 3.2 MLM Implementation

```python
def create_mlm_data(tokens, tokenizer, mlm_probability=0.15):
    """Create masked language model training data."""
    labels = tokens.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # Don't mask special tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), 0.0)

    # Create mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Only predict masked tokens
    labels[~masked_indices] = -100

    # 80% [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    tokens[indices_replaced] = tokenizer.mask_token_id

    # 10% random
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    tokens[indices_random] = random_words[indices_random]

    # 10% unchanged (already done - original tokens)

    return tokens, labels
```

### 3.3 Next Sentence Prediction (NSP)

**Task**: Given two sentences, is sentence B the actual next sentence after A?

```
Positive (50%):
  A: "The cat sat on the mat."
  B: "It was a fluffy cat."
  Label: IsNext

Negative (50%):
  A: "The cat sat on the mat."
  B: "The stock market crashed today."
  Label: NotNext
```

**Implementation**:
```
[CLS] Sentence A [SEP] Sentence B [SEP]
  ↓
BERT
  ↓
[CLS] hidden state → Binary classifier → IsNext / NotNext
```

### 3.4 Why NSP? (And Why It Was Controversial)

**Original motivation**:
- Help with tasks requiring sentence pairs (QA, NLI)
- Learn document structure

**Later findings** (RoBERTa paper):
- NSP may actually hurt performance
- Easy to solve using topic overlap
- Doesn't generalize well

**Modern practice**: Most successors remove NSP.

### 3.5 Pre-training Loss

```
Total Loss = MLM Loss + NSP Loss

MLM Loss = CrossEntropy(predicted_tokens, masked_tokens)
NSP Loss = CrossEntropy(is_next_prediction, is_next_label)
```

### 3.6 Pre-training Data and Scale

| Aspect | BERT |
|--------|------|
| **Data** | BooksCorpus (800M words) + English Wikipedia (2.5B words) |
| **Steps** | 1M steps |
| **Batch size** | 256 sequences |
| **Sequence length** | 512 tokens (90% of training at 128) |
| **Hardware** | 4 TPU pods (16 TPU chips) |
| **Time** | 4 days |

---

## 4. Fine-tuning BERT

### 4.1 The Fine-tuning Paradigm

```
Pre-trained BERT (general language understanding)
         ↓
    + Task Head
         ↓
Fine-tune on task-specific data (small, labeled)
         ↓
Task-specific model
```

**Key benefits**:
- Much less data needed (100s to 10,000s examples)
- Much faster training (hours, not days)
- Better performance than training from scratch

### 4.2 Classification Tasks

**Single sentence classification**:
```
Input: [CLS] This movie is great! [SEP]
         ↓
       BERT
         ↓
      [CLS] hidden → Linear → Sentiment (pos/neg)
```

**Sentence pair classification**:
```
Input: [CLS] Sentence A [SEP] Sentence B [SEP]
         ↓
       BERT
         ↓
      [CLS] hidden → Linear → Entailment / Contradiction / Neutral
```

### 4.3 Token Classification (NER, POS Tagging)

```
Input: [CLS] John works at Google [SEP]
         ↓
       BERT
         ↓
       hidden_0, hidden_1, hidden_2, hidden_3, hidden_4
                    ↓        ↓         ↓         ↓
              Linear    Linear   Linear    Linear
                    ↓        ↓         ↓         ↓
                  O     B-PER     O     B-ORG    O
```

### 4.4 Question Answering (Extractive)

```
Input: [CLS] Question [SEP] Context with the answer in it [SEP]
         ↓
       BERT
         ↓
     Start classifier → Position of answer start
     End classifier → Position of answer end

Answer = Context[start:end]
```

### 4.5 Fine-tuning Hyperparameters

| Hyperparameter | Recommended Value |
|----------------|-------------------|
| Learning rate | 2e-5, 3e-5, 5e-5 |
| Batch size | 16, 32 |
| Epochs | 2-4 |
| Warmup | 10% of steps |
| Max length | 128 or 512 |

**Key insight**: BERT is sensitive to learning rate. Too high → catastrophic forgetting.

### 4.6 Fine-tuning Code

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy='epoch',
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

---

## 5. BERT Variants

### 5.1 RoBERTa (Robustly Optimized BERT)

**Key changes** (Liu et al., 2019):
- Remove NSP (just MLM)
- Larger batches (8K)
- More data (160GB)
- Dynamic masking (different masks each epoch)
- Longer training

| Change | BERT | RoBERTa |
|--------|------|---------|
| NSP | Yes | No |
| Batch size | 256 | 8,000 |
| Data | 16GB | 160GB |
| Masking | Static | Dynamic |

**Result**: +3-5% on most benchmarks.

### 5.2 ALBERT (A Lite BERT)

**Key changes** (Lan et al., 2019):
- Factorized embedding (vocab_size × E, E × hidden)
- Cross-layer parameter sharing
- SOP instead of NSP (Sentence Order Prediction)

```
BERT embedding: vocab_size × hidden_size = 30K × 768 = 23M params
ALBERT embedding: 30K × 128 + 128 × 768 = 3.8M + 0.1M = 3.9M params
```

| Model | Layers | Hidden | Params |
|-------|--------|--------|--------|
| BERT-large | 24 | 1024 | 340M |
| ALBERT-xxlarge | 12 | 4096 | 235M |

### 5.3 DistilBERT

**Key changes** (Sanh et al., 2019):
- Knowledge distillation from BERT
- 6 layers (vs 12)
- 40% smaller, 60% faster
- 97% of BERT performance

**Distillation loss**:
```
L = α × L_ce + β × L_mlm + γ × L_cos

L_ce: Soft cross-entropy with teacher
L_mlm: Masked LM loss
L_cos: Cosine embedding loss
```

### 5.4 DeBERTa (Disentangled BERT)

**Key changes** (He et al., 2020):
- Disentangled attention (separate content and position)
- Enhanced mask decoder
- Virtual adversarial training

```
Standard attention:
  Attention(H) where H = content + position

DeBERTa:
  Attention(content, position) separately, then combine
```

### 5.5 ELECTRA

**Key changes** (Clark et al., 2020):
- Replace MLM with Replaced Token Detection (RTD)
- Generator produces fake tokens
- Discriminator detects which tokens are fake

```
Generator: "The [MASK] sat" → "The dog sat"
Discriminator: Which tokens are replaced?
  "The" → original
  "dog" → replaced (was "cat")
  "sat" → original
```

**Benefit**: All tokens provide signal, not just 15%.

### 5.6 Comparison Table

| Model | Params | Speed | GLUE | Key Innovation |
|-------|--------|-------|------|----------------|
| BERT-base | 110M | 1x | 79.6 | MLM + NSP |
| RoBERTa | 125M | 1x | 83.2 | Better pre-training |
| ALBERT-xxlarge | 235M | 0.3x | 84.6 | Parameter sharing |
| DistilBERT | 66M | 1.6x | 77.0 | Distillation |
| DeBERTa-large | 400M | 0.7x | 88.1 | Disentangled attention |
| ELECTRA-large | 335M | 1x | 85.1 | RTD |

---

## 6. BERT vs GPT

### 6.1 Fundamental Differences

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal (left-to-right) |
| Pre-training | MLM + NSP | Next token prediction |
| Primary use | Understanding | Generation |
| Fine-tuning | Add task head | Prompt/instruction tuning |

### 6.2 When to Use Which?

**Use BERT (encoder) for**:
- Classification (sentiment, topic)
- Named Entity Recognition
- Question Answering (extractive)
- Sentence similarity
- When you need full context understanding

**Use GPT (decoder) for**:
- Text generation
- Summarization
- Translation
- Code generation
- Chatbots / conversational AI
- When output is free-form text

### 6.3 The Key Trade-off

```
BERT: Sees everything → Better understanding
      "The [MASK] sat on the mat. It was fluffy."
      [MASK] can see "fluffy" → easier to predict "cat"

GPT: Sees only left → Can generate
     "The cat sat on the" → generates "mat"
     Can produce new text autoregressively
```

### 6.4 Why GPT "Won" for LLMs

Despite BERT's advantages for understanding:

1. **Generation is more versatile**: Any task can be framed as generation
2. **Scaling**: Decoder-only scales better to 100B+ params
3. **In-context learning**: GPT-3 showed few-shot works at scale
4. **Instruction following**: ChatGPT showed conversational capability

**Modern approach**: Use GPT-style for most tasks, BERT-style for specific needs (search, classification at scale).

---

## 7. Practical Applications

### 7.1 Text Classification

```python
from transformers import pipeline

# Zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "This movie was fantastic and entertaining!",
    candidate_labels=["positive", "negative", "neutral"]
)

# Fine-tuned classification
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
result = classifier("I love this product!")
```

### 7.2 Named Entity Recognition

```python
from transformers import pipeline

ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
result = ner("Apple CEO Tim Cook announced new products in Cupertino.")

# Output:
# [{'entity': 'B-ORG', 'word': 'Apple'},
#  {'entity': 'B-PER', 'word': 'Tim'},
#  {'entity': 'I-PER', 'word': 'Cook'},
#  {'entity': 'B-LOC', 'word': 'Cupertino'}]
```

### 7.3 Question Answering

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
result = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is the capital of France."
)
# {'answer': 'Paris', 'score': 0.98, 'start': 39, 'end': 44}
```

### 7.4 Sentence Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["This is a sentence", "This is another sentence"]
embeddings = model.encode(sentences)

# Compare similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
```

### 7.5 Semantic Search

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Index documents
documents = ["Doc 1 content...", "Doc 2 content...", ...]
doc_embeddings = model.encode(documents)

# Search
query = "What is machine learning?"
query_embedding = model.encode(query)

# Find most similar
scores = util.cos_sim(query_embedding, doc_embeddings)[0]
top_results = scores.argsort(descending=True)[:5]
```

---

## 8. Interview Questions

### Q1: Explain the BERT architecture and how it differs from GPT.

**Answer**:

**BERT Architecture**:
- Encoder-only transformer (no decoder)
- 12 layers (base) or 24 layers (large)
- Bidirectional attention (each token sees all others)
- Input: [CLS] + tokens + [SEP] + (optional second sentence + [SEP])
- Uses token + segment + position embeddings

**Key Differences from GPT**:

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal (left-to-right) |
| Pre-training | Masked LM | Next token prediction |
| Use case | Understanding | Generation |

**Why bidirectional matters**:
```
"The bank by the river"
BERT: "bank" sees "river" → understands it's a riverbank
GPT: "bank" only sees "The" → ambiguous until later
```

### Q2: Explain Masked Language Modeling (MLM) and why it works.

**Answer**:

**MLM Procedure**:
1. Randomly select 15% of tokens
2. Of selected tokens:
   - 80%: Replace with [MASK]
   - 10%: Replace with random token
   - 10%: Keep unchanged
3. Model predicts original tokens

**Example**:
```
Input:  "The [MASK] sat on the [MASK]"
Target: "The  cat   sat on the  mat"
```

**Why it works**:
1. **Forces context understanding**: Must use surrounding words to predict
2. **Bidirectional**: Can use left AND right context
3. **Self-supervised**: No labels needed, just text

**Why the 80-10-10 split**:
- 80% [MASK]: Primary learning signal
- 10% random: Teaches model that non-[MASK] tokens might be wrong too
- 10% unchanged: Bridges gap to fine-tuning (no [MASK] at inference)

### Q3: What is the purpose of the [CLS] token?

**Answer**:

**[CLS] (Classification) token**:
- Special token added at the start of every input
- Its final hidden state is used for sequence-level tasks

**How it works**:
```
Input: [CLS] The movie was great [SEP]
         ↓
       BERT
         ↓
   h_CLS, h_The, h_movie, h_was, h_great, h_SEP
     ↓
   h_CLS → Linear layer → Classification output
```

**Why it works**:
- Attends to all tokens in the sequence
- Aggregates information from entire input
- Provides fixed-size representation for any length input

**Use cases**:
- Sentiment classification: [CLS] → positive/negative
- Next Sentence Prediction: [CLS] → IsNext/NotNext
- Sentence similarity: Compare two [CLS] embeddings

### Q4: Compare RoBERTa, ALBERT, and DistilBERT.

**Answer**:

**RoBERTa** (Robustly Optimized BERT):
- Removes NSP (just MLM)
- Dynamic masking (different mask each epoch)
- Larger batches (8K vs 256)
- More data (160GB vs 16GB)
- Longer training
- **Result**: Same size, better performance (+3-5%)

**ALBERT** (A Lite BERT):
- Factorized embeddings (smaller embedding matrix)
- Cross-layer parameter sharing (one layer, repeated)
- Sentence Order Prediction (SOP) instead of NSP
- **Result**: Fewer parameters, similar performance

**DistilBERT** (Distilled BERT):
- 6 layers (vs 12)
- Knowledge distillation from BERT-base
- 40% smaller, 60% faster
- **Result**: 97% performance with much faster inference

| Model | Params | Speed | Best For |
|-------|--------|-------|----------|
| RoBERTa | Same | Same | Best accuracy |
| ALBERT | Fewer | Slower | Memory constraints |
| DistilBERT | 60% | 1.6x | Production speed |

### Q5: How does fine-tuning BERT work for different tasks?

**Answer**:

**Classification** (single or pair):
```
[CLS] sentence(s) [SEP]
         ↓
       BERT
         ↓
   [CLS] hidden → Linear(768, num_classes) → output
```

**Token Classification** (NER, POS):
```
[CLS] token1 token2 token3 [SEP]
         ↓
       BERT
         ↓
   h_1, h_2, h_3 → Linear(768, num_tags) per token
```

**Question Answering** (extractive):
```
[CLS] question [SEP] context [SEP]
         ↓
       BERT
         ↓
   Start: Linear(768, 1) → position scores
   End: Linear(768, 1) → position scores
   Answer = context[argmax(start):argmax(end)]
```

**Fine-tuning tips**:
- Lower learning rate (2e-5 to 5e-5)
- Fewer epochs (2-4)
- Warmup (10% of steps)
- Don't freeze BERT (full fine-tuning usually better)

### Q6: Why did GPT-style models become dominant over BERT-style?

**Answer**:

**BERT advantages**:
- Better contextual understanding (bidirectional)
- Excellent for classification, NER, QA
- More sample-efficient for specific tasks

**Why GPT "won" for LLMs**:

1. **Versatility of generation**:
   - Any task can be framed as text generation
   - BERT can't generate new text naturally

2. **Scaling properties**:
   - Decoder-only scales better to 100B+ params
   - Causal attention is simpler to train

3. **In-context learning** (GPT-3):
   - Few-shot learning without fine-tuning
   - Works better with scale

4. **Instruction following**:
   - ChatGPT showed conversation works
   - Users prefer interactive generation

5. **Unified paradigm**:
   - One model for all tasks
   - No task-specific heads needed

**Modern best practices**:
- GPT-style for general purpose, generation, chat
- BERT-style for specific tasks (search ranking, classification at scale)

---

## 9. Summary

### BERT Recipe

```
Architecture:
- Encoder-only transformer
- 12/24 layers, 768/1024 hidden
- Bidirectional attention
- [CLS] + [SEP] special tokens

Pre-training:
- MLM (15% masking: 80%/10%/10%)
- NSP (sentence pair prediction)
- BookCorpus + Wikipedia

Fine-tuning:
- Add task head on [CLS] or token outputs
- Low learning rate (2e-5)
- Few epochs (2-4)
```

### Key Equations

**MLM Loss**:
```
L_MLM = -Σ log P(x_masked | x_context)
```

**NSP Loss**:
```
L_NSP = -log P(IsNext | [CLS] representation)
```

**Attention** (bidirectional, no mask):
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### When to Use BERT

| Task | Use BERT? | Alternative |
|------|-----------|-------------|
| Classification | Yes | GPT with prompting |
| NER | Yes | - |
| Extractive QA | Yes | - |
| Sentence similarity | Yes | - |
| Text generation | No | GPT |
| Summarization | Partial | GPT, BART |
| Translation | No | Encoder-decoder |

### Key Takeaways

1. **BERT = Bidirectional encoder** for understanding tasks
2. **MLM is the key innovation** - learn from context both ways
3. **[CLS] token** provides sentence-level representation
4. **Fine-tuning is efficient** - small data, big gains
5. **Variants optimize for different needs**: RoBERTa (accuracy), DistilBERT (speed), ALBERT (size)
6. **GPT complements BERT** - generation vs understanding
