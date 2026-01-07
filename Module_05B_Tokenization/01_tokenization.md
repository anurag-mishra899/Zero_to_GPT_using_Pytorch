# Module 5B: Tokenization - From Text to Tokens

## Table of Contents
1. [Why Tokenization Matters](#1-why-tokenization-matters)
2. [Tokenization Strategies](#2-tokenization-strategies)
3. [Byte Pair Encoding (BPE)](#3-byte-pair-encoding-bpe)
4. [WordPiece](#4-wordpiece)
5. [Unigram Language Model](#5-unigram-language-model)
6. [SentencePiece](#6-sentencepiece)
7. [Modern Tokenizers](#7-modern-tokenizers)
8. [Special Tokens](#8-special-tokens)
9. [Practical Considerations](#9-practical-considerations)
10. [Interview Questions](#10-interview-questions)
11. [Summary](#11-summary)

---

## 1. Why Tokenization Matters

### 1.1 The Bridge Between Text and Numbers

Neural networks operate on numbers, not text. Tokenization is the critical first step that converts raw text into numerical representations.

```
Text: "Hello, world!"
         ↓ Tokenization
Tokens: ["Hello", ",", " world", "!"]
         ↓ Vocabulary Lookup
IDs: [15496, 11, 995, 0]
         ↓ Embedding Layer
Vectors: [[0.1, -0.2, ...], [0.3, 0.1, ...], ...]
```

### 1.2 The Tokenization Trilemma

Every tokenization strategy must balance three competing goals:

```
            Vocabulary Size
                 /\
                /  \
               /    \
              /      \
             /________\
    Coverage          Meaning
```

| Goal | Description | Trade-off |
|------|-------------|-----------|
| **Small Vocabulary** | Fewer embeddings to learn | May lose semantic meaning |
| **Good Coverage** | Handle any text (no UNK) | May need more tokens |
| **Meaningful Units** | Tokens carry semantic info | Larger vocabulary |

### 1.3 Why Not Just Use Words?

**Word-level tokenization problems**:

1. **Huge vocabulary**: English has 170,000+ words
2. **Out-of-vocabulary (OOV)**: New words, typos, names become `<UNK>`
3. **Morphology blindness**: "run", "running", "runs" are unrelated
4. **Multilingual nightmare**: Each language explodes vocabulary

```
Word-level:
  "unhappiness" → ["unhappiness"] (1 token, or UNK if rare)

Subword-level:
  "unhappiness" → ["un", "happiness"] (2 tokens, compositional!)
```

### 1.4 Why Not Just Use Characters?

**Character-level tokenization problems**:

1. **Very long sequences**: 5x-10x more tokens than words
2. **Harder to learn**: Model must learn spelling, then words, then meaning
3. **Computational cost**: O(n²) attention on longer sequences
4. **Lost word boundaries**: Harder for model to identify units

```
Character-level:
  "Hello" → ["H", "e", "l", "l", "o"] (5 tokens)

Subword-level:
  "Hello" → ["Hello"] (1 token)
```

### 1.5 The Subword Solution

**Key insight**: Use a vocabulary of subword units that balances:
- Common words as single tokens ("the", "and", "hello")
- Rare words as multiple subwords ("un" + "believ" + "able")

**Benefits**:
- Manageable vocabulary (32K-100K tokens)
- Zero OOV tokens (can represent any text)
- Morphological awareness ("un-" prefix, "-ing" suffix)
- Cross-lingual sharing ("inter-" used in many languages)

---

## 2. Tokenization Strategies

### 2.1 Overview of Methods

```
                  Tokenization Methods
                          |
        +-----------------+-----------------+
        |                 |                 |
   Character           Word             Subword
        |                 |                 |
   [a,b,c,...]     [word1,word2,...]        |
                                   +--------+--------+
                                   |        |        |
                                  BPE   WordPiece  Unigram
                                   |        |        |
                                 GPT-2    BERT     T5/XLNet
```

### 2.2 Method Comparison

| Method | Used By | Vocabulary | Approach |
|--------|---------|------------|----------|
| **BPE** | GPT-2, GPT-3, RoBERTa, LLaMA | 50K-100K | Merge frequent pairs |
| **WordPiece** | BERT, DistilBERT | 30K | Merge by likelihood |
| **Unigram** | T5, XLNet, ALBERT | 32K | Remove by likelihood |
| **SentencePiece** | Library | Any | Language-agnostic BPE/Unigram |

### 2.3 Training vs Inference

**Training a tokenizer** (done once):
1. Collect large corpus
2. Apply algorithm (BPE/WordPiece/Unigram)
3. Build vocabulary
4. Save tokenizer

**Using a tokenizer** (every time):
1. Load pre-trained tokenizer
2. Encode: text → token IDs
3. Decode: token IDs → text

---

## 3. Byte Pair Encoding (BPE)

### 3.1 History

Originally a data compression algorithm (Gage, 1994). Adapted for NLP by Sennrich et al. (2016) for neural machine translation.

### 3.2 The Algorithm

**Core idea**: Iteratively merge the most frequent pair of tokens.

**Training algorithm**:
```
1. Initialize vocabulary with all characters + end-of-word marker
2. Count frequency of all adjacent pairs
3. Merge the most frequent pair into a new token
4. Repeat steps 2-3 until vocabulary size reached
```

### 3.3 BPE Example

**Corpus**: "low lower lowest" (with word frequencies)
```
Initial vocabulary: {l, o, w, e, r, s, t, _}  (_ = end-of-word)

Word frequencies:
  "low_": 5
  "lower_": 2
  "lowest_": 1

Step 1: Character-level
  l o w _     (5 times)
  l o w e r _ (2 times)
  l o w e s t _ (1 time)

Count pairs:
  (l, o): 8
  (o, w): 8
  (w, _): 5
  (w, e): 3
  (e, r): 2
  (e, s): 1
  (s, t): 1
  (t, _): 1
  (r, _): 2

Most frequent: (l, o) and (o, w) both have 8
Merge (l, o) → "lo"

Step 2: After first merge
  lo w _       (5 times)
  lo w e r _   (2 times)
  lo w e s t _ (1 time)

Count pairs:
  (lo, w): 8     ← Most frequent
  (w, _): 5
  ...

Merge (lo, w) → "low"

Step 3: After second merge
  low _       (5 times)
  low e r _   (2 times)
  low e s t _ (1 time)

Count pairs:
  (low, _): 5    ← Most frequent
  (low, e): 3
  ...

Merge (low, _) → "low_"

Continue until desired vocabulary size...
```

### 3.4 BPE Encoding (Inference)

**Greedy encoding**: Apply merge rules in learned order.

```python
def bpe_encode(text, merges):
    tokens = list(text)  # Start with characters

    for (a, b), merged in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i+1] == b:
                tokens[i:i+2] = [merged]
            else:
                i += 1

    return tokens
```

### 3.5 Byte-Level BPE (GPT-2)

**Problem with character BPE**: Unicode has 150,000+ characters.

**Solution**: Work at byte level (256 possible values).

```
GPT-2 Byte-Level BPE:
1. Encode text as UTF-8 bytes
2. Map bytes to printable characters (for readability)
3. Apply BPE on these characters

Benefit: Can tokenize ANY text (any language, emoji, code)
```

**Example**:
```
Text: "Hello"
UTF-8 bytes: [72, 101, 108, 108, 111]
Mapped to: ['H', 'e', 'l', 'l', 'o']
After BPE: ['Hello'] (if "Hello" is in vocab)
```

### 3.6 BPE Properties

| Property | Value |
|----------|-------|
| Deterministic | Yes - same text always produces same tokens |
| Greedy | Yes - always takes longest possible match |
| Handles OOV | Yes - falls back to bytes/characters |
| Vocabulary growth | Linear with merges |

---

## 4. WordPiece

### 4.1 Overview

Developed by Google for Google's speech recognition (2012), later used in BERT.

**Key difference from BPE**: Instead of merging most *frequent* pair, merge pair that maximizes *likelihood* of the training data.

### 4.2 The Algorithm

```
BPE: Merge argmax count(AB)

WordPiece: Merge argmax P(AB) / (P(A) × P(B))
           = Merge argmax count(AB) / (count(A) × count(B))
```

**Intuition**: Merge pairs that occur together more than expected by chance.

### 4.3 WordPiece Example

```
Corpus: "hug", "hugs", "hugger", "huggers" with frequencies

BPE would merge:
  Most frequent pair overall

WordPiece would merge:
  Pair with highest: P(XY) / (P(X) × P(Y))

  If "##er" appears mostly after "hugg", it gets merged early
  Even if "##s" appears more frequently overall
```

### 4.4 The ## Prefix

WordPiece uses `##` to indicate continuation:

```
"hugging" → ["hug", "##ging"]
"unhappy" → ["un", "##happy"]

The ## means "this attaches to the previous token"
```

**Why this matters**:
- "play" (start of word) vs "##play" (continuation) are different tokens
- Helps model distinguish word boundaries

### 4.5 WordPiece Encoding

Unlike BPE (which applies merges in order), WordPiece uses **maximum matching**:

```python
def wordpiece_encode(word, vocab):
    tokens = []
    start = 0

    while start < len(word):
        end = len(word)
        found = False

        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr

            if substr in vocab:
                tokens.append(substr)
                found = True
                break
            end -= 1

        if not found:
            tokens.append("[UNK]")
            start += 1
        else:
            start = end

    return tokens
```

### 4.6 BPE vs WordPiece

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| Merge criterion | Frequency | Likelihood ratio |
| Encoding | Apply merges in order | Greedy longest match |
| Subword marker | End (Ġ for space) | Start (##) |
| OOV handling | Never (byte-level) | [UNK] token |
| Used by | GPT, RoBERTa, LLaMA | BERT, DistilBERT |

---

## 5. Unigram Language Model

### 5.1 Overview

Proposed by Kudo (2018). Takes opposite approach from BPE:
- **BPE**: Start small, add tokens by merging
- **Unigram**: Start large, remove tokens by pruning

### 5.2 The Algorithm

```
1. Initialize with large vocabulary (all substrings up to length N)
2. Train unigram language model: P(token)
3. For each token, compute loss if removed
4. Remove tokens with smallest loss increase (keep x%)
5. Repeat until desired vocabulary size
```

### 5.3 Unigram Probability Model

**Model**: Each token has probability P(x_i), tokenizations are independent.

```
P(tokenization) = ∏ P(token_i)

P("unbelievable") with tokenization ["un", "believ", "able"]:
  = P("un") × P("believ") × P("able")
```

### 5.4 Encoding with Unigram

**Key difference**: Multiple valid tokenizations exist!

```
"unbelievable" could be:
  ["un", "believable"]      P = 0.001 × 0.0001 = 1e-7
  ["un", "believ", "able"]  P = 0.001 × 0.0003 × 0.002 = 6e-10
  ["unbelievable"]          P = 0.00001 = 1e-5  ← Most probable!
```

**Viterbi algorithm** finds most probable tokenization in O(n²).

### 5.5 Subword Regularization

**Unique feature**: Can sample different tokenizations during training!

```python
# Instead of always using most probable:
tokens = tokenize_best("unbelievable")  # ["unbelievable"]

# Can sample proportional to probability:
tokens = tokenize_sample("unbelievable")  # Maybe ["un", "believable"]
```

**Benefits**:
- Data augmentation (same text → different tokens)
- More robust model
- Better handling of rare words

### 5.6 Comparison

| Aspect | BPE | Unigram |
|--------|-----|---------|
| Building | Bottom-up (merge) | Top-down (prune) |
| Encoding | Deterministic | Probabilistic (can sample) |
| Tokenizations | One per text | Multiple possible |
| Regularization | No | Yes (subword sampling) |
| Used by | GPT, LLaMA | T5, ALBERT, XLNet |

---

## 6. SentencePiece

### 6.1 What is SentencePiece?

**SentencePiece** is a library (not an algorithm) that implements BPE and Unigram with key advantages:

1. **Language-agnostic**: Treats text as raw bytes/unicode
2. **No pre-tokenization**: Doesn't require word segmentation
3. **Reversible**: Can perfectly reconstruct original text

### 6.2 The Pre-tokenization Problem

**Traditional pipeline**:
```
"Hello, world!"
    ↓ Pre-tokenize (language-specific!)
["Hello", ",", "world", "!"]
    ↓ Subword tokenize
["Hello", ",", "world", "!"]
```

**Problem**: Pre-tokenization is language-dependent:
- English: split on spaces and punctuation
- Chinese: no spaces between words
- German: compound words ("Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz")

### 6.3 SentencePiece Solution

**Treat everything as a sequence of characters**:
```
"Hello, world!"
    ↓ Direct to subword (no pre-tokenization)
["▁Hello", ",", "▁world", "!"]

"▁" (U+2581) represents space/word boundary
```

**Benefits**:
- Works identically for all languages
- Spaces are explicitly represented
- Perfect reconstruction: tokens → original text

### 6.4 SentencePiece Training

```python
import sentencepiece as spm

# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',  # or 'unigram'
    character_coverage=0.9995,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
```

### 6.5 SentencePiece Usage

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('my_tokenizer.model')

# Encode
text = "Hello, world!"
tokens = sp.encode_as_pieces(text)  # ['▁Hello', ',', '▁world', '!']
ids = sp.encode_as_ids(text)        # [1234, 45, 678, 90]

# Decode (perfect reconstruction)
decoded = sp.decode_pieces(tokens)   # "Hello, world!"
decoded = sp.decode_ids(ids)         # "Hello, world!"
```

---

## 7. Modern Tokenizers

### 7.1 Hugging Face Tokenizers

The **tokenizers** library provides fast, Rust-based implementations:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Use
output = tokenizer.encode("Hello, world!")
print(output.tokens)  # ['Hello', ',', 'world', '!']
print(output.ids)     # [1234, 45, 678, 90]
```

### 7.2 tiktoken (OpenAI)

GPT-3.5/GPT-4 use **tiktoken**, an efficient BPE implementation:

```python
import tiktoken

# Load GPT-4 tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

# Encode
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]

# Decode
text = enc.decode(tokens)
print(text)  # "Hello, world!"

# Token count (useful for API limits)
print(len(tokens))  # 4
```

### 7.3 Tokenizer Comparison by Model

| Model | Library | Type | Vocab Size | Special Features |
|-------|---------|------|------------|------------------|
| GPT-2 | HF/tiktoken | Byte-BPE | 50,257 | Byte-level, no UNK |
| GPT-3/4 | tiktoken | Byte-BPE | 100,277 | cl100k_base |
| BERT | HF | WordPiece | 30,522 | [CLS], [SEP], [MASK] |
| T5 | SentencePiece | Unigram | 32,000 | </s> for EOS |
| LLaMA | SentencePiece | BPE | 32,000 | Language-agnostic |
| LLaMA-2 | SentencePiece | BPE | 32,000 | Same as LLaMA |
| Mistral | SentencePiece | BPE | 32,000 | Same as LLaMA |

### 7.4 Vocabulary Size Considerations

```
Smaller vocab (8K-16K):
  + Smaller embedding matrix
  + Faster softmax
  - Longer sequences (more tokens per text)
  - Less semantic tokens

Larger vocab (64K-100K):
  + Shorter sequences
  + More whole-word tokens
  - Larger embedding matrix
  - More memory

Sweet spot: 32K-50K for most models
```

---

## 8. Special Tokens

### 8.1 Common Special Tokens

| Token | Used By | Purpose |
|-------|---------|---------|
| `[PAD]` / `<pad>` | All | Padding sequences to same length |
| `[UNK]` / `<unk>` | BERT, T5 | Unknown/OOV token |
| `[CLS]` | BERT | Classification token (sentence repr) |
| `[SEP]` | BERT | Separator between sentences |
| `[MASK]` | BERT | Masked token for MLM |
| `<s>` | GPT, LLaMA | Beginning of sequence (BOS) |
| `</s>` | GPT, LLaMA, T5 | End of sequence (EOS) |
| `<|endoftext|>` | GPT-2/3 | Document boundary |
| `<|im_start|>` | ChatGPT | Chat message start |
| `<|im_end|>` | ChatGPT | Chat message end |

### 8.2 Special Tokens in Different Models

**BERT**:
```
Input: "Hello world"
Tokenized: [CLS] Hello world [SEP]
IDs: [101, 7592, 2088, 102]

For sentence pairs:
[CLS] sentence A [SEP] sentence B [SEP]
```

**GPT-2**:
```
Input: "Hello world"
Tokenized: Hello world
IDs: [15496, 995]

For multiple documents:
Document 1 <|endoftext|> Document 2 <|endoftext|>
```

**LLaMA / Chat Models**:
```
<s>[INST] User message [/INST] Assistant response </s>
```

### 8.3 Adding Special Tokens

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add new special tokens
special_tokens = {"additional_special_tokens": ["<CUSTOM>", "<ANOTHER>"]}
tokenizer.add_special_tokens(special_tokens)

# IMPORTANT: Resize model embeddings after adding tokens!
model.resize_token_embeddings(len(tokenizer))
```

---

## 9. Practical Considerations

### 9.1 Tokenization Pitfalls

**1. Tokenization mismatch**:
```python
# WRONG: Different tokenizers for train/inference
train_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
infer_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # Different!

# RIGHT: Same tokenizer always
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Save and load the same tokenizer
```

**2. Max length truncation**:
```python
# Text longer than model's max length
long_text = "..." * 10000

# Option 1: Truncate (loses information!)
tokens = tokenizer(long_text, truncation=True, max_length=512)

# Option 2: Sliding window
tokens = tokenizer(long_text, return_overflowing_tokens=True,
                   max_length=512, stride=128)
```

**3. Padding issues**:
```python
# Padding side matters for generation!
tokenizer.padding_side = "left"  # For decoder-only (GPT)
tokenizer.padding_side = "right" # For encoder (BERT)
```

### 9.2 Token Counting for API Limits

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Check before API call
text = "Your long prompt..."
if count_tokens(text) > 8000:
    print("Warning: May exceed context limit!")
```

### 9.3 Debugging Tokenization

```python
# See exactly how text is tokenized
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, I'm learning tokenization!"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print("Tokens:", tokens)
# ['Hello', ',', 'ĠI', "'m", 'Ġlearning', 'Ġtoken', 'ization', '!']

print("IDs:", ids)
# [15496, 11, 314, 1101, 4673, 11241, 1634, 0]

# Decode each token separately to see what it represents
for id in ids:
    print(f"{id} → '{tokenizer.decode([id])}'")
```

### 9.4 Multilingual Tokenization

```python
# Same tokenizer, different languages
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

texts = [
    "Hello world",           # English
    "Bonjour le monde",      # French
    "Hallo Welt",            # German
    "こんにちは世界",          # Japanese
    "مرحبا بالعالم",          # Arabic
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    print(f"{text}: {len(tokens)} tokens → {tokens}")
```

### 9.5 Tokenization Speed

```python
import time
from tokenizers import Tokenizer

# Rust-based tokenizer (fast)
fast_tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# Python-based (slower)
from transformers import BertTokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "This is a test. " * 1000

start = time.time()
fast_tokenizer.encode(text)
print(f"Fast: {time.time() - start:.4f}s")

start = time.time()
slow_tokenizer.encode(text)
print(f"Slow: {time.time() - start:.4f}s")

# Fast is typically 10-100x faster for batches
```

---

## 10. Interview Questions

### Q1: Explain the difference between BPE and WordPiece.

**Answer**:

Both are subword tokenization algorithms, but they differ in merge criterion:

**BPE (Byte Pair Encoding)**:
```
Merge criterion: Most frequent pair
merge = argmax count(A, B)

Example: If "th" appears 1000 times and "he" appears 800 times,
BPE merges "th" first.
```

**WordPiece**:
```
Merge criterion: Highest likelihood increase
merge = argmax P(AB) / (P(A) × P(B))
      = argmax count(AB) × N / (count(A) × count(B))

This prefers pairs that co-occur more than expected by chance.
```

**Key differences**:
| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| Criterion | Frequency | Likelihood ratio |
| Bias | Favors common pairs | Favors meaningful merges |
| Encoding | Apply merges in order | Greedy longest match |
| Marker | End (Ġ) | Start (##) |

### Q2: Why do modern LLMs use subword tokenization instead of word-level?

**Answer**:

**Word-level problems**:

1. **Vocabulary explosion**:
   - English has 170K+ words
   - Add names, technical terms, typos → millions
   - Embedding matrix becomes huge

2. **Out-of-vocabulary (OOV)**:
   - Any unseen word → `<UNK>`
   - "TensorFlow" might be UNK in older models
   - Loses information

3. **No morphological awareness**:
   - "happy", "unhappy", "happiness" are unrelated
   - Can't generalize patterns

**Subword benefits**:

1. **Fixed vocabulary**: 32K-100K tokens cover any text
2. **No OOV**: Fall back to characters/bytes
3. **Compositional**: "un" + "happy" shares meaning
4. **Efficient**: Common words = 1 token, rare = multiple

### Q3: What is byte-level BPE and why does GPT-2 use it?

**Answer**:

**Standard BPE problem**: Unicode has 150K+ characters. Initial vocabulary would be huge.

**Byte-level BPE solution**:
```
1. Encode text as UTF-8 bytes (only 256 possible values)
2. Map bytes to printable characters for readability
3. Apply BPE on these 256 base tokens
```

**Benefits**:
- Initial vocab = 256 (not 150K)
- Can tokenize ANY text: all languages, emoji, code
- Never produces `<UNK>` (any text = sequence of bytes)
- No pre-tokenization needed

**GPT-2 example**:
```
Text: "Hello 🌍"
UTF-8: [72, 101, 108, 108, 111, 32, 240, 159, 140, 141]
Tokens: ["Hello", " ", "🌍"] (if emoji in vocab)
    or: ["Hello", " ", byte tokens for emoji]
```

### Q4: How does SentencePiece differ from other tokenizers?

**Answer**:

**Key differences**:

1. **No pre-tokenization**:
   ```
   Traditional: "Hello world" → ["Hello", "world"] → subwords
   SentencePiece: "Hello world" → subwords directly
   ```

2. **Whitespace is a token**:
   ```
   Regular: "hello world" → ["hello", "world"]
   SentencePiece: "hello world" → ["▁hello", "▁world"]

   ▁ (U+2581) explicitly represents the space
   ```

3. **Language agnostic**:
   - No language-specific rules
   - Works identically for English, Chinese, Arabic, etc.

4. **Reversible**:
   - Can perfectly reconstruct original text
   - No information loss from spaces

**Used by**: T5, ALBERT, XLNet, LLaMA, Mistral

### Q5: What are special tokens and why are they important?

**Answer**:

**Special tokens** are reserved tokens with specific meanings:

| Token | Purpose | Example |
|-------|---------|---------|
| `<PAD>` | Padding | Making sequences equal length |
| `<BOS>`/`<s>` | Beginning | Signal sequence start |
| `<EOS>`/`</s>` | End | Signal sequence end |
| `<UNK>` | Unknown | OOV words |
| `[CLS]` | Classification | Sentence representation (BERT) |
| `[SEP]` | Separator | Between segments (BERT) |
| `[MASK]` | Mask | MLM training (BERT) |

**Why important**:

1. **Model knows sequence boundaries**: BOS/EOS
2. **Batch processing**: PAD enables variable-length batching
3. **Task-specific**: CLS for classification, MASK for MLM
4. **Conversation structure**: Chat models use special tokens for turns

**Critical**: Must use same special tokens for training and inference!

### Q6: How would you handle a new domain with many OOV words?

**Answer**:

**Options** (in order of preference):

1. **Use byte-level tokenizer** (GPT-2, LLaMA):
   - No OOV by design
   - Domain words become multiple tokens
   - No changes needed

2. **Add special tokens**:
   ```python
   tokenizer.add_special_tokens({"additional_special_tokens": ["<GENE>", "<PROTEIN>"]})
   model.resize_token_embeddings(len(tokenizer))
   # Fine-tune model on domain data
   ```

3. **Train domain-specific tokenizer**:
   ```python
   # Train new tokenizer on domain corpus
   new_tokenizer = Tokenizer(BPE())
   new_tokenizer.train(["domain_corpus.txt"])
   # Retrain model from scratch
   ```

4. **Extend existing vocabulary**:
   ```python
   # Add domain words to existing tokenizer
   new_tokens = ["CRISPR", "mRNA", "epitope"]
   tokenizer.add_tokens(new_tokens)
   model.resize_token_embeddings(len(tokenizer))
   ```

**Best practice**: Start with byte-level tokenizer, only customize if needed.

---

## 11. Summary

### Algorithm Comparison

| Algorithm | Approach | Encoding | Used By |
|-----------|----------|----------|---------|
| **BPE** | Bottom-up merge by frequency | Deterministic | GPT, LLaMA |
| **WordPiece** | Bottom-up merge by likelihood | Greedy longest | BERT |
| **Unigram** | Top-down prune by likelihood | Probabilistic | T5, XLNet |

### Key Equations

**BPE merge**:
```
next_merge = argmax_{(A,B)} count(A, B)
```

**WordPiece merge**:
```
next_merge = argmax_{(A,B)} count(A,B) / (count(A) × count(B))
```

**Unigram probability**:
```
P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ)
```

### Quick Reference

```python
# Hugging Face (most common)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model_name")
tokens = tokenizer.encode("text")

# tiktoken (OpenAI models)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("text")

# SentencePiece (T5, LLaMA)
import sentencepiece as spm
sp = spm.SentencePieceProcessor("model.model")
tokens = sp.encode_as_ids("text")
```

### Key Takeaways

1. **Subword tokenization** is the standard for modern NLP
2. **BPE** is most common (GPT, LLaMA) - simple and effective
3. **Byte-level** eliminates OOV entirely (GPT-2+)
4. **Special tokens** are critical for model behavior
5. **Same tokenizer** must be used for training and inference
6. **Vocabulary size** is a trade-off (32K-100K typical)
7. **SentencePiece** enables language-agnostic tokenization
