# Module 1.4: Datasets and DataLoaders

## Table of Contents
1. [Overview](#1-overview)
2. [Custom Dataset Class](#2-custom-dataset-class)
3. [DataLoader Deep Dive](#3-dataloader-deep-dive)
4. [Samplers](#4-samplers)
5. [Data Transforms](#5-data-transforms)
6. [Efficient Data Loading](#6-efficient-data-loading)
7. [Common Patterns for NLP](#7-common-patterns-for-nlp)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Overview

### 1.1 Why Datasets and DataLoaders?

Training neural networks requires:
- **Iteration**: Loop through data multiple times (epochs)
- **Batching**: Group samples for efficient GPU utilization
- **Shuffling**: Randomize order to prevent learning data order
- **Parallelism**: Load data in background while GPU trains
- **Memory efficiency**: Load only what fits in memory

### 1.2 Two Core Abstractions

```
Dataset: Defines HOW to access individual samples
         - __len__(): Total number of samples
         - __getitem__(idx): Get sample at index

DataLoader: Defines HOW to iterate through Dataset
            - Batching
            - Shuffling
            - Parallel loading
            - Collating
```

---

## 2. Custom Dataset Class

### 2.1 Basic Structure

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
```

### 2.2 Map-Style vs Iterable-Style

**Map-style** (most common):
- Implements `__getitem__` and `__len__`
- Random access by index
- Supports shuffling

**Iterable-style**:
- Implements `__iter__`
- Sequential access only
- For streaming data (too large for memory)

```python
# Iterable-style dataset
class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path) as f:
            for line in f:
                yield process(line)
```

### 2.3 Dataset for Text (NLP)

```python
class TextDataset(Dataset):
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
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }
```

### 2.4 Dataset for Language Modeling

```python
class LMDataset(Dataset):
    """Dataset for causal language modeling (GPT-style)"""

    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        # Input: tokens[idx:idx+block_size]
        # Target: tokens[idx+1:idx+block_size+1] (shifted by 1)
        x = torch.tensor(self.token_ids[idx:idx + self.block_size])
        y = torch.tensor(self.token_ids[idx + 1:idx + self.block_size + 1])
        return x, y
```

---

## 3. DataLoader Deep Dive

### 3.1 Basic Usage

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

for batch in loader:
    inputs, labels = batch
    # Train...
```

### 3.2 Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `batch_size` | Samples per batch | 16-512 |
| `shuffle` | Randomize order each epoch | True for train |
| `num_workers` | Parallel data loading processes | 4-8 |
| `pin_memory` | Use pinned memory for faster GPU transfer | True if GPU |
| `drop_last` | Drop incomplete final batch | True for training |
| `collate_fn` | Function to combine samples into batch | Custom for NLP |
| `sampler` | Custom sampling strategy | WeightedRandomSampler |

### 3.3 Understanding num_workers

```
num_workers=0: Main process loads data (slowest)
num_workers=1: One subprocess loads data
num_workers=4: Four subprocesses load in parallel

Rule of thumb: num_workers = 4 * num_gpus
              or num_cpus // 2
```

**Gotchas**:
- On Windows, use `if __name__ == '__main__':` guard
- Workers inherit dataset state at fork time
- Large num_workers increases memory usage
- Set `persistent_workers=True` to avoid respawning each epoch

### 3.4 Collate Functions

Collate functions combine samples into batches. Default works for simple tensors:

```python
# Default collate: stack tensors along new dimension
samples = [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]
batch = default_collate(samples)  # tensor([[1, 2], [3, 4], [5, 6]])
```

Custom collate for variable-length sequences:

```python
def collate_fn(batch):
    """Pad sequences to max length in batch"""
    inputs, labels = zip(*batch)

    # Pad sequences
    inputs_padded = nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=0
    )

    labels = torch.stack(labels)
    return inputs_padded, labels
```

### 3.5 Collate for Transformers

```python
def transformer_collate_fn(batch):
    """Collate function for transformer inputs"""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    # Pad to max length in batch
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }
```

---

## 4. Samplers

### 4.1 Built-in Samplers

```python
from torch.utils.data import (
    SequentialSampler,    # [0, 1, 2, 3, ...]
    RandomSampler,        # Random permutation
    SubsetRandomSampler,  # Random from subset of indices
    WeightedRandomSampler # Sample with replacement based on weights
)
```

### 4.2 Handling Imbalanced Classes

```python
# Count samples per class
class_counts = torch.bincount(labels)

# Inverse frequency as weights
class_weights = 1.0 / class_counts.float()

# Weight for each sample
sample_weights = class_weights[labels]

# Weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Note: Can't use shuffle=True with sampler
```

### 4.3 Custom Sampler

```python
class BucketSampler(torch.utils.data.Sampler):
    """Sample sequences of similar length together for efficiency"""

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

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.lengths)
```

---

## 5. Data Transforms

### 5.1 Transform Pipeline

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.RandomHorizontalFlip(p=0.5),
])

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
```

### 5.2 Text Transforms

```python
class TextTransforms:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text):
        # Lowercase
        text = text.lower()

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_length
        )

        return torch.tensor(tokens)
```

---

## 6. Efficient Data Loading

### 6.1 Memory Mapping for Large Files

```python
import numpy as np

class MemmapDataset(Dataset):
    """Memory-mapped dataset for large files"""

    def __init__(self, file_path, dtype=np.float32, shape=None):
        self.data = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())
```

### 6.2 Pre-tokenized Data (LLM Training)

```python
class PreTokenizedDataset(Dataset):
    """Load pre-tokenized data for fast training"""

    def __init__(self, token_file, block_size):
        # Load all tokens as memory-mapped array
        self.tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.tokens[idx + 1:idx + self.block_size + 1].astype(np.int64))
        return x, y
```

### 6.3 Pin Memory and Non-Blocking Transfer

```python
# Enable pin_memory for faster CPU->GPU transfer
loader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=4)

for batch_x, batch_y in loader:
    # Non-blocking transfer
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)

    # Forward pass (data transfer happens in parallel)
    output = model(batch_x)
```

### 6.4 Prefetching with CUDA Streams

```python
class CUDAPrefetcher:
    """Prefetch data to GPU in background"""

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = tuple(
                t.to(self.device, non_blocking=True)
                for t in self.next_batch
            )

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
```

---

## 7. Common Patterns for NLP

### 7.1 Instruction Tuning Dataset

```python
class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning (like Alpaca)"""

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format: instruction + input -> output
        prompt = f"### Instruction:\n{item['instruction']}\n\n"
        if item.get('input'):
            prompt += f"### Input:\n{item['input']}\n\n"
        prompt += "### Response:\n"

        response = item['output']

        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt)
        response_tokens = self.tokenizer.encode(response)

        # Combine and truncate
        input_ids = prompt_tokens + response_tokens
        input_ids = input_ids[:self.max_length]

        # Labels: -100 for prompt tokens (don't compute loss)
        labels = [-100] * len(prompt_tokens) + response_tokens
        labels = labels[:self.max_length]

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels)
        }
```

### 7.2 Dynamic Padding Collator

```python
class DynamicPaddingCollator:
    """Pad to max length in batch, not global max"""

    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, batch):
        # Find max length in this batch
        max_len = max(len(item['input_ids']) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            # Pad to max_len
            padding_length = max_len - len(item['input_ids'])

            input_ids.append(
                torch.cat([item['input_ids'], torch.zeros(padding_length, dtype=torch.long)])
            )
            attention_masks.append(
                torch.cat([torch.ones(len(item['input_ids'])), torch.zeros(padding_length)])
            )
            if 'labels' in item:
                labels.append(
                    torch.cat([item['labels'], torch.full((padding_length,), -100)])
                )

        result = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
        }
        if labels:
            result['labels'] = torch.stack(labels)

        return result
```

### 7.3 Multi-Dataset Training

```python
from torch.utils.data import ConcatDataset, ChainDataset

# Concatenate datasets (map-style)
combined = ConcatDataset([dataset1, dataset2, dataset3])

# Chain datasets (iterable-style)
chained = ChainDataset([iter_dataset1, iter_dataset2])

# Interleaving with probability
class InterleavedDataset(Dataset):
    def __init__(self, datasets, probs):
        self.datasets = datasets
        self.probs = probs / np.sum(probs)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        # Sample dataset according to probability
        dataset_idx = np.random.choice(len(self.datasets), p=self.probs)
        # Sample from that dataset
        sample_idx = np.random.randint(len(self.datasets[dataset_idx]))
        return self.datasets[dataset_idx][sample_idx]
```

---

## 8. Interview Questions

### Q1: What's the difference between `num_workers=0` and `num_workers>0`?

**Answer**:
- `num_workers=0`: Data loading happens in main process, blocking training
- `num_workers>0`: Data loading in separate processes, can overlap with GPU computation

Gotchas with num_workers>0:
- Higher memory usage (each worker has copy of dataset)
- Windows requires `if __name__ == '__main__':` guard
- Random state is inherited from fork time (set worker_init_fn for different seeds)

### Q2: Why use `pin_memory=True`?

**Answer**: Pinned (page-locked) memory enables faster CPU→GPU transfer:
- Normal memory can be swapped to disk
- Pinned memory is always in RAM, enabling DMA transfer
- GPU can copy directly without CPU involvement

Use when:
- Training on GPU
- Data loading is not the bottleneck
- You have enough RAM (pinned memory can't be swapped)

### Q3: How do you handle variable-length sequences?

**Answer**:
1. **Padding**: Pad all sequences to same length
   - Fixed max length (simpler)
   - Dynamic padding per batch (more efficient)
2. **Custom collate_fn**: Combine samples and add padding
3. **Bucket sampling**: Group similar lengths together
4. **Pack sequences**: Use `pack_padded_sequence` for RNNs

### Q4: What is a collate_fn and when do you need a custom one?

**Answer**: `collate_fn` combines individual samples into a batch. Default works when:
- All samples are tensors of same shape
- Or simple tuples of tensors

Need custom when:
- Variable-length sequences (need padding)
- Complex return types (dicts, nested structures)
- Special batching logic (e.g., negative sampling)

### Q5: How would you handle a dataset too large for memory?

**Answer**:
1. **Memory mapping**: `np.memmap` for numpy arrays
2. **Iterable dataset**: Stream from disk
3. **Lazy loading**: Load samples on-demand in `__getitem__`
4. **Sharded files**: Split data into chunks, load one at a time
5. **HuggingFace Datasets**: Built-in memory mapping and streaming

---

## 9. Summary

### Key Concepts

1. **Dataset**: Defines data access (`__len__`, `__getitem__`)
2. **DataLoader**: Handles batching, shuffling, parallelism
3. **Collate function**: Combines samples into batches
4. **Samplers**: Control sampling order (weighted, bucket, etc.)
5. **Efficiency**: pin_memory, num_workers, prefetching

### Best Practices

```python
# Training loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,  # For consistent batch sizes
    persistent_workers=True  # Avoid respawning each epoch
)

# Validation loader
val_loader = DataLoader(
    val_dataset,
    batch_size=64,  # Can be larger (no gradients)
    shuffle=False,  # Reproducible evaluation
    num_workers=4,
    pin_memory=True,
    drop_last=False  # Evaluate all samples
)
```

### Training Loop Pattern

```python
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
```
