# Module 5C: Text Preprocessing & Data Pipeline

## Table of Contents
1. [Why Preprocessing Matters](#1-why-preprocessing-matters)
2. [Text Cleaning](#2-text-cleaning)
3. [Building NLP Datasets](#3-building-nlp-datasets)
4. [Collation Functions](#4-collation-functions)
5. [Efficient Data Loading](#5-efficient-data-loading)
6. [Data Augmentation](#6-data-augmentation)
7. [Production Pipelines](#7-production-pipelines)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Why Preprocessing Matters

### 1.1 The Garbage In, Garbage Out Principle

Your model is only as good as your data. Preprocessing transforms raw text into clean, consistent input that models can learn from effectively.

```
Raw Web Data:
  "<p>Hello   WORLD!!!  </p>\n\n\t  Click HERE: http://spam.com"

After Preprocessing:
  "Hello world! Click here."

Model Performance: Dramatically different!
```

### 1.2 Preprocessing Pipeline Overview

```
Raw Text
    ↓
[1. Cleaning] → Remove HTML, fix encoding, normalize whitespace
    ↓
[2. Normalization] → Lowercase, expand contractions, standardize
    ↓
[3. Tokenization] → Convert to tokens (Module 5B)
    ↓
[4. Numericalization] → Convert tokens to IDs
    ↓
[5. Batching] → Group sequences, pad/truncate
    ↓
Model Input
```

### 1.3 When to Preprocess

| Stage | What to Do | When |
|-------|------------|------|
| **Data Collection** | Remove duplicates, filter quality | Once |
| **Dataset Creation** | Clean, normalize, tokenize | Once (save results) |
| **Training** | Batch, pad, augment | Every epoch |
| **Inference** | Same as training (minus augmentation) | Every request |

### 1.4 Critical Rule: Consistency

**The preprocessing at inference MUST match training exactly.**

```python
# WRONG: Different preprocessing
def train_preprocess(text):
    return text.lower().strip()

def inference_preprocess(text):
    return text.strip()  # Forgot lowercase!

# RIGHT: Same function
def preprocess(text):
    return text.lower().strip()
```

---

## 2. Text Cleaning

### 2.1 Common Cleaning Operations

```python
def clean_text(text: str) -> str:
    # 1. Fix encoding issues
    text = fix_encoding(text)

    # 2. Remove HTML tags
    text = remove_html(text)

    # 3. Normalize whitespace
    text = normalize_whitespace(text)

    # 4. Remove/replace URLs
    text = handle_urls(text)

    # 5. Handle special characters
    text = handle_special_chars(text)

    return text
```

### 2.2 Encoding Issues

Real-world text often has encoding problems:

```
Problem: "cafÃ©" (UTF-8 interpreted as Latin-1)
Solution: "café"

Problem: "â€™" (curly quote as bytes)
Solution: "'"
```

**Using ftfy (fixes text for you)**:
```python
import ftfy

broken = "The Mona Lisa doesnÃ¢â‚¬â„¢t have eyebrows."
fixed = ftfy.fix_text(broken)
# "The Mona Lisa doesn't have eyebrows."
```

### 2.3 HTML and Markup Removal

```python
import re
from html import unescape

def remove_html(text: str) -> str:
    # Decode HTML entities
    text = unescape(text)  # &amp; → &

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    return text

# Example
html = "<p>Hello &amp; <b>World</b>!</p>"
clean = remove_html(html)  # "Hello & World!"
```

### 2.4 Whitespace Normalization

```python
def normalize_whitespace(text: str) -> str:
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove tabs
    text = text.replace('\t', ' ')

    return text
```

### 2.5 URL Handling

Options for URLs:

```python
# Option 1: Remove completely
text = re.sub(r'https?://\S+', '', text)

# Option 2: Replace with token
text = re.sub(r'https?://\S+', '<URL>', text)

# Option 3: Keep as is (model might learn from URLs)
# No change

# Best practice: Depends on task
# - Classification: Usually remove
# - Generation: Keep or replace with <URL>
```

### 2.6 Special Characters

```python
def handle_special_chars(text: str) -> str:
    # Normalize unicode quotes and dashes
    replacements = {
        '"': '"', '"': '"',  # Curly quotes
        ''': "'", ''': "'",  # Curly apostrophes
        '–': '-', '—': '-',  # Dashes
        '…': '...',          # Ellipsis
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text
```

### 2.7 Case Normalization

**When to lowercase**:
- Classification tasks (usually)
- When vocabulary is limited
- When case doesn't matter for task

**When to preserve case**:
- Named Entity Recognition (NER)
- Generation tasks
- When case carries meaning ("WHO" vs "who")

```python
# BERT uncased: Always lowercase
text = text.lower()

# BERT cased: Preserve case
# No change

# Mixed approach: First letter of sentences
import re
def sentence_case(text):
    return re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), text.lower())
```

---

## 3. Building NLP Datasets

### 3.1 PyTorch Dataset Basics

```python
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """Basic text classification dataset."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',  # Or handle in collate
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }
```

### 3.2 Language Modeling Dataset

```python
class LanguageModelDataset(Dataset):
    """Dataset for causal language modeling (GPT-style)."""

    def __init__(self, text, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Tokenize entire text at once
        tokens = tokenizer.encode(text)

        # Split into blocks
        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(tokens[i:i + block_size + 1])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]

        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        return {
            'input_ids': torch.tensor(tokens[:-1]),
            'labels': torch.tensor(tokens[1:])
        }
```

### 3.3 Sequence-to-Sequence Dataset

```python
class Seq2SeqDataset(Dataset):
    """Dataset for encoder-decoder models (translation, summarization)."""

    def __init__(self, sources, targets, tokenizer, max_source_len=512, max_target_len=128):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]

        # Tokenize source (encoder input)
        source_encoding = self.tokenizer(
            source,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors='pt'
        )

        # Tokenize target (decoder input/output)
        target_encoding = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_target_len,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }
```

### 3.4 Instruction Tuning Dataset

```python
class InstructionDataset(Dataset):
    """Dataset for instruction-following models."""

    def __init__(self, data, tokenizer, max_length=2048):
        """
        data: List of dicts with 'instruction', 'input', 'output' keys
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format prompt
        if item.get('input'):
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"

        full_text = prompt + item['output']

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Create labels (mask prompt tokens with -100)
        labels = encoding['input_ids'].clone()
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[0, :prompt_len] = -100  # Don't compute loss on prompt

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
```

---

## 4. Collation Functions

### 4.1 Why Collation Matters

Batching requires all sequences to have the same length. Collation functions handle:
- Padding sequences to equal length
- Creating attention masks
- Handling variable-length inputs efficiently

### 4.2 Basic Collate Function

```python
def collate_fn(batch):
    """Basic collation with padding."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Find max length in batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad sequences
    padded_input_ids = []
    attention_masks = []

    for ids in input_ids:
        padding_len = max_len - len(ids)
        padded_input_ids.append(
            torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)])
        )
        attention_masks.append(
            torch.cat([torch.ones(len(ids)), torch.zeros(padding_len)])
        )

    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }
```

### 4.3 Dynamic Padding (Efficient)

```python
from transformers import DataCollatorWithPadding

# Hugging Face provides optimized collators
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,           # Pad to longest in batch
    max_length=512,         # Maximum sequence length
    return_tensors='pt'
)

# Use with DataLoader
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)
```

### 4.4 Collation for Language Modeling

```python
from transformers import DataCollatorForLanguageModeling

# For causal LM (GPT-style)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Not masked LM
)

# For masked LM (BERT-style)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,  # Mask 15% of tokens
)
```

### 4.5 Left Padding for Decoder Models

```python
def left_pad_collate(batch, tokenizer):
    """Left padding for decoder-only models (GPT-style generation)."""
    input_ids = [item['input_ids'] for item in batch]
    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    attention_masks = []

    for ids in input_ids:
        padding_len = max_len - len(ids)
        # Left padding (pad at the beginning)
        padded_input_ids.append(
            torch.cat([torch.full((padding_len,), tokenizer.pad_token_id), ids])
        )
        attention_masks.append(
            torch.cat([torch.zeros(padding_len), torch.ones(len(ids))])
        )

    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks)
    }
```

**Why left padding for generation?**
```
Right padding (wrong for generation):
  ["Hello", "world", "<PAD>", "<PAD>"]
  Model generates after PAD tokens

Left padding (correct for generation):
  ["<PAD>", "<PAD>", "Hello", "world"]
  Model generates after actual content
```

---

## 5. Efficient Data Loading

### 5.1 DataLoader Best Practices

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # Shuffle for training
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    collate_fn=collator,
    prefetch_factor=2,      # Prefetch batches
    persistent_workers=True # Keep workers alive
)
```

### 5.2 Memory-Efficient Loading

For large datasets that don't fit in memory:

```python
class StreamingDataset(torch.utils.data.IterableDataset):
    """Memory-efficient dataset that streams from disk."""

    def __init__(self, file_path, tokenizer, block_size=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            buffer = []
            for line in f:
                tokens = self.tokenizer.encode(line.strip())
                buffer.extend(tokens)

                while len(buffer) >= self.block_size + 1:
                    yield {
                        'input_ids': torch.tensor(buffer[:self.block_size]),
                        'labels': torch.tensor(buffer[1:self.block_size + 1])
                    }
                    buffer = buffer[self.block_size:]
```

### 5.3 Using Hugging Face Datasets

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("imdb")

# Memory-mapped (efficient for large datasets)
dataset = load_dataset("wikipedia", "20220301.en", streaming=True)

# Preprocess with map
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert to PyTorch format
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
```

### 5.4 Bucketing by Length

Group similar-length sequences to minimize padding:

```python
from torch.utils.data import Sampler
import numpy as np

class BucketBatchSampler(Sampler):
    """Sample batches with similar-length sequences."""

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Sort indices by length
        indices = np.argsort(self.lengths)

        # Create batches of similar lengths
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]

        # Shuffle batch order (not within batches)
        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch.tolist()

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
```

---

## 6. Data Augmentation

### 6.1 Why Augment Text Data?

- Increase effective dataset size
- Improve model robustness
- Handle rare cases and variations

### 6.2 Simple Augmentation Techniques

```python
import random
import nltk
from nltk.corpus import wordnet

# Download required data
# nltk.download('wordnet')
# nltk.download('punkt')

def random_deletion(words, p=0.1):
    """Randomly delete words with probability p."""
    if len(words) == 1:
        return words
    return [w for w in words if random.random() > p]

def random_swap(words, n=1):
    """Randomly swap n pairs of words."""
    words = words.copy()
    for _ in range(n):
        if len(words) >= 2:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
    return words

def synonym_replacement(words, n=1):
    """Replace n words with synonyms."""
    words = words.copy()
    candidates = [i for i, w in enumerate(words) if wordnet.synsets(w)]

    for _ in range(min(n, len(candidates))):
        idx = random.choice(candidates)
        word = words[idx]
        syns = wordnet.synsets(word)
        if syns:
            lemmas = syns[0].lemmas()
            if lemmas:
                words[idx] = lemmas[0].name()
        candidates.remove(idx)

    return words
```

### 6.3 Back-Translation

```python
def back_translate(text, src_lang='en', pivot_lang='de'):
    """
    Augment by translating to another language and back.

    "The cat sat" → "Die Katze saß" → "The cat was sitting"
    """
    # Using transformers translation models
    from transformers import pipeline

    # Translate to pivot language
    forward = pipeline('translation', model=f'Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}')
    translated = forward(text)[0]['translation_text']

    # Translate back
    backward = pipeline('translation', model=f'Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}')
    back = backward(translated)[0]['translation_text']

    return back
```

### 6.4 EDA (Easy Data Augmentation)

```python
def eda(text, alpha_sr=0.1, alpha_rd=0.1, alpha_rs=0.1, num_aug=4):
    """
    Easy Data Augmentation (Wei & Zou, 2019)

    Combines:
    - Synonym Replacement (SR)
    - Random Deletion (RD)
    - Random Swap (RS)
    """
    words = text.split()
    n = len(words)

    augmented = []
    for _ in range(num_aug):
        new_words = words.copy()

        # Synonym replacement
        n_sr = max(1, int(alpha_sr * n))
        new_words = synonym_replacement(new_words, n_sr)

        # Random deletion
        new_words = random_deletion(new_words, alpha_rd)

        # Random swap
        n_rs = max(1, int(alpha_rs * n))
        new_words = random_swap(new_words, n_rs)

        augmented.append(' '.join(new_words))

    return augmented
```

### 6.5 Augmentation for Language Modeling

```python
def random_token_masking(tokens, tokenizer, mask_prob=0.15):
    """
    BERT-style random masking for pre-training.

    80% [MASK], 10% random, 10% unchanged
    """
    labels = tokens.clone()
    mask = torch.rand(len(tokens)) < mask_prob

    for i, should_mask in enumerate(mask):
        if should_mask:
            r = random.random()
            if r < 0.8:
                tokens[i] = tokenizer.mask_token_id
            elif r < 0.9:
                tokens[i] = random.randint(0, tokenizer.vocab_size - 1)
            # else: keep original (10%)
        else:
            labels[i] = -100  # Don't compute loss

    return tokens, labels
```

---

## 7. Production Pipelines

### 7.1 Complete Training Pipeline

```python
def create_training_pipeline(
    train_texts,
    train_labels,
    tokenizer,
    batch_size=32,
    max_length=512,
    num_workers=4
):
    """Create complete training data pipeline."""

    # 1. Preprocessing
    processed_texts = [preprocess(text) for text in train_texts]

    # 2. Create dataset
    dataset = TextDataset(
        texts=processed_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # 3. Create collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    return dataloader
```

### 7.2 Preprocessing Cache

```python
import hashlib
import pickle
from pathlib import Path

class PreprocessingCache:
    """Cache preprocessed data to avoid recomputation."""

    def __init__(self, cache_dir='./cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, text, config):
        """Generate unique key for text + config combination."""
        content = f"{text}_{config}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text, config):
        """Get cached result if exists."""
        key = self._get_cache_key(text, config)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, text, config, result):
        """Cache result."""
        key = self._get_cache_key(text, config)
        cache_file = self.cache_dir / f"{key}.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

### 7.3 Inference Pipeline

```python
class InferencePipeline:
    """Production inference pipeline with batching."""

    def __init__(self, model, tokenizer, device='cuda', max_length=512):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def preprocess(self, text):
        """Same preprocessing as training!"""
        text = text.strip()
        # Add any other preprocessing steps used during training
        return text

    def predict(self, texts, batch_size=32):
        """Batch prediction."""
        # Preprocess all texts
        processed = [self.preprocess(t) for t in texts]

        all_predictions = []

        for i in range(0, len(processed), batch_size):
            batch_texts = processed[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors='pt'
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)

            all_predictions.extend(predictions.cpu().tolist())

        return all_predictions
```

---

## 8. Interview Questions

### Q1: What are the key steps in a text preprocessing pipeline?

**Answer**:

A complete text preprocessing pipeline includes:

1. **Text Cleaning**:
   - Fix encoding issues (UTF-8 problems)
   - Remove HTML/markup tags
   - Normalize whitespace (multiple spaces → single)
   - Handle URLs (remove or replace)
   - Normalize special characters (curly quotes → straight)

2. **Text Normalization**:
   - Case normalization (lowercase for classification)
   - Expand contractions ("don't" → "do not")
   - Handle numbers (normalize or remove)

3. **Tokenization**:
   - Apply tokenizer (BPE, WordPiece, etc.)
   - Handle special tokens

4. **Numericalization**:
   - Convert tokens to IDs

5. **Batching**:
   - Pad/truncate to equal lengths
   - Create attention masks

**Critical**: Inference preprocessing MUST match training exactly!

### Q2: Explain padding and attention masks. Why are they necessary?

**Answer**:

**Padding**: Making all sequences in a batch the same length.

```
Batch: ["Hello", "Hi there friend"]
Without padding: Different lengths (1, 3) - can't batch!
With padding: [["Hello", "<PAD>", "<PAD>"], ["Hi", "there", "friend"]]
```

**Attention mask**: Tells model which tokens are real vs padding.

```
Tokens:         ["Hello", "<PAD>", "<PAD>"]
Attention mask: [1,       0,       0      ]

In attention: mask ensures PAD tokens get zero attention weight
```

**Why necessary**:
1. **Batching requires equal lengths** - GPU operations need fixed shapes
2. **Model shouldn't attend to padding** - Attention mask prevents this
3. **Loss shouldn't count padding** - Labels set to -100 for padding

**Padding direction matters**:
- **Right padding (encoder)**: ["Hello", "world", "<PAD>"]
- **Left padding (decoder/generation)**: ["<PAD>", "Hello", "world"]

### Q3: What is dynamic padding and why is it better?

**Answer**:

**Static padding**: Pad all sequences to a fixed max length.
```python
tokenizer(texts, padding='max_length', max_length=512)
# All sequences become 512 tokens, regardless of actual length
```

**Dynamic padding**: Pad to longest sequence in batch.
```python
tokenizer(texts, padding='longest')
# Batch 1: max_len=47 → all padded to 47
# Batch 2: max_len=123 → all padded to 123
```

**Why dynamic is better**:
1. **Less computation**: Attention is O(n²), shorter = much faster
2. **Less memory**: Fewer padding tokens = smaller batches fit
3. **Faster training**: ~2-3x speedup in practice

**Implementation**:
```python
from transformers import DataCollatorWithPadding
collator = DataCollatorWithPadding(tokenizer, padding=True)
```

### Q4: How do you handle variable-length sequences efficiently?

**Answer**:

**1. Dynamic Padding** (most common):
- Pad to longest in batch
- Use attention masks

**2. Bucketing**:
- Group similar-length sequences
- Minimize padding waste

```python
# Sort by length, batch similar lengths together
sorted_indices = sorted(range(len(data)), key=lambda i: len(data[i]))
batches = [sorted_indices[i:i+batch_size] for i in range(0, len(data), batch_size)]
```

**3. Packing** (advanced):
- Concatenate multiple sequences into one
- Use special tokens as separators
- Maximizes GPU utilization

```python
# Pack multiple short sequences into one
packed = [SEP].join(short_sequences)
# Track sequence boundaries for loss computation
```

**4. Truncation strategies**:
- `max_length` truncation for very long sequences
- Sliding window for preserving all content

### Q5: What are common data augmentation techniques for NLP?

**Answer**:

**1. Lexical Augmentation**:
- **Synonym replacement**: "good" → "excellent"
- **Random deletion**: Remove random words
- **Random swap**: Swap word positions
- **Random insertion**: Insert synonyms

**2. Back-translation**:
```
English → French → English
"The cat sat" → "Le chat s'assit" → "The cat was sitting"
```
Creates paraphrases while preserving meaning.

**3. For Language Models**:
- **Token masking**: BERT-style [MASK] tokens
- **Token dropping**: Drop random tokens
- **Sentence shuffling**: Reorder sentences

**4. For Classification**:
- **Label-preserving noise**: Typos, case changes
- **Mixup**: Interpolate embeddings + labels

**When to use**:
- Small datasets benefit most
- Diminishing returns with large datasets
- Back-translation best for paraphrasing

### Q6: How do you build a production-ready data pipeline?

**Answer**:

**Key components**:

1. **Preprocessing consistency**:
```python
# Save preprocessing config with model
config = {'lowercase': True, 'max_length': 512, ...}
# Apply SAME config at inference
```

2. **Efficient data loading**:
```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # Parallel loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch batches
)
```

3. **Caching**:
```python
# Cache tokenized data to disk
tokenized.save_to_disk('cached_data/')
# Load without re-tokenizing
dataset = load_from_disk('cached_data/')
```

4. **Memory efficiency**:
```python
# For large datasets: memory mapping
dataset = load_dataset(..., streaming=True)

# Process in chunks
for chunk in dataset.iter(batch_size=1000):
    process(chunk)
```

5. **Monitoring**:
```python
# Log data statistics
log.info(f"Batch lengths: min={min_len}, max={max_len}, mean={mean_len}")
# Detect data issues early
```

---

## 9. Summary

### Preprocessing Checklist

```
□ Fix encoding issues (ftfy)
□ Remove HTML/markup
□ Normalize whitespace
□ Handle URLs consistently
□ Normalize special characters
□ Apply consistent casing
□ Use same tokenizer as training
□ Handle max length (truncation)
□ Create attention masks
□ Verify preprocessing matches training
```

### Key Code Patterns

**DataLoader setup**:
```python
DataLoader(dataset, batch_size=32, collate_fn=collator, num_workers=4, pin_memory=True)
```

**Tokenization**:
```python
tokenizer(texts, truncation=True, padding='longest', return_tensors='pt')
```

**Collation**:
```python
DataCollatorWithPadding(tokenizer, padding=True)  # Dynamic padding
```

### Key Takeaways

1. **Preprocessing consistency is critical**: Same steps for train and inference
2. **Dynamic padding saves compute**: 2-3x faster than fixed padding
3. **Bucketing reduces padding waste**: Group similar lengths
4. **Augmentation helps small datasets**: EDA, back-translation
5. **Cache tokenized data**: Avoid re-tokenizing every epoch
6. **Left pad for generation**: Right pad for encoding
7. **Use Hugging Face collators**: Optimized and tested
