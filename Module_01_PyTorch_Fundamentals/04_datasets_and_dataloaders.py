"""
Module 1.4: Datasets and DataLoaders
====================================

This module covers:
1. Custom Dataset classes
2. DataLoader configuration
3. Samplers for imbalanced data
4. Collate functions for NLP
5. Efficient data loading
6. Patterns for LLM training

Run this file to see all examples in action.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data import SequentialSampler, RandomSampler, WeightedRandomSampler
import numpy as np

print("=" * 70)
print("MODULE 1.4: DATASETS AND DATALOADERS")
print("=" * 70)


# =============================================================================
# SECTION 1: BASIC DATASET
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: BASIC DATASET")
print("=" * 70)

# 1.1 Simple dataset
print("\n--- 1.1 Basic Custom Dataset ---")

class SimpleDataset(Dataset):
    """Basic dataset structure"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create sample data
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randint(0, 2, (100,))  # Binary labels

dataset = SimpleDataset(X, y)
print(f"Dataset size: {len(dataset)}")
print(f"Sample 0: X shape={dataset[0][0].shape}, y={dataset[0][1]}")


# 1.2 Dataset with transforms
print("\n--- 1.2 Dataset with Transforms ---")

class TransformDataset(Dataset):
    """Dataset with optional transform"""

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

# Example transform: normalize
def normalize_transform(x):
    return (x - x.mean()) / (x.std() + 1e-8)

dataset_with_transform = TransformDataset(X, y, transform=normalize_transform)
sample, label = dataset_with_transform[0]
print(f"After transform - mean: {sample.mean():.4f}, std: {sample.std():.4f}")


# =============================================================================
# SECTION 2: NLP DATASETS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: NLP DATASETS")
print("=" * 70)

# 2.1 Text Classification Dataset
print("\n--- 2.1 Text Classification Dataset ---")

class TextClassificationDataset(Dataset):
    """Dataset for text classification with simple tokenization"""

    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab  # word -> index mapping
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        """Simple whitespace tokenization"""
        words = text.lower().split()
        indices = [self.vocab.get(w, self.vocab['<UNK>']) for w in words]
        return indices

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        tokens = self.tokenize(text)

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_length - len(tokens))

        return torch.tensor(tokens), torch.tensor(label)

# Example usage
texts = ["hello world", "this is a test", "machine learning is great"]
labels = [0, 1, 1]
vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3, 'this': 4, 'is': 5,
         'a': 6, 'test': 7, 'machine': 8, 'learning': 9, 'great': 10}

text_dataset = TextClassificationDataset(texts, labels, vocab, max_length=10)
tokens, label = text_dataset[0]
print(f"Text: '{texts[0]}'")
print(f"Tokens: {tokens.tolist()}")
print(f"Label: {label.item()}")


# 2.2 Language Modeling Dataset
print("\n--- 2.2 Language Modeling Dataset (GPT-style) ---")

class LanguageModelingDataset(Dataset):
    """
    Dataset for causal language modeling.
    Input: tokens[i:i+block_size]
    Target: tokens[i+1:i+block_size+1] (shifted by 1)
    """

    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

# Example
token_ids = list(range(100))  # Simulated token IDs
lm_dataset = LanguageModelingDataset(token_ids, block_size=8)
x, y = lm_dataset[0]
print(f"Block size: 8")
print(f"Input (x):  {x.tolist()}")
print(f"Target (y): {y.tolist()}")
print("Note: Target is input shifted by 1 position")


# =============================================================================
# SECTION 3: DATALOADER
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DATALOADER")
print("=" * 70)

# 3.1 Basic DataLoader
print("\n--- 3.1 Basic DataLoader ---")

dataset = SimpleDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,  # Use 0 for this demo
    drop_last=False
)

print(f"Number of batches: {len(loader)}")
for i, (batch_x, batch_y) in enumerate(loader):
    if i == 0:
        print(f"First batch - X shape: {batch_x.shape}, y shape: {batch_y.shape}")
    if i == len(loader) - 1:
        print(f"Last batch - X shape: {batch_x.shape}, y shape: {batch_y.shape}")

# 3.2 DataLoader parameters
print("\n--- 3.2 DataLoader Parameters ---")
print("""
Key parameters:
  batch_size:       Samples per batch (16-512 typical)
  shuffle:          Randomize order each epoch (True for training)
  num_workers:      Parallel data loading (4-8 typical)
  pin_memory:       Faster CPU->GPU transfer (True if using GPU)
  drop_last:        Drop incomplete last batch (True for training)
  collate_fn:       Custom function to combine samples
  sampler:          Custom sampling strategy

Example:
  train_loader = DataLoader(
      train_dataset,
      batch_size=32,
      shuffle=True,
      num_workers=4,
      pin_memory=True,
      drop_last=True,
      persistent_workers=True  # Keep workers alive between epochs
  )
""")


# =============================================================================
# SECTION 4: COLLATE FUNCTIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: COLLATE FUNCTIONS")
print("=" * 70)

# 4.1 Why custom collate?
print("\n--- 4.1 Why Custom Collate Functions? ---")
print("""
Default collate stacks tensors along new dimension.
Works when all samples have same shape.

Need custom collate when:
- Variable length sequences (need padding)
- Complex data structures (dicts, nested)
- Special batching logic
""")

# 4.2 Padding collate for variable length
print("\n--- 4.2 Padding Collate Function ---")

class VariableLengthDataset(Dataset):
    """Dataset returning variable length sequences"""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random length sequence
        length = np.random.randint(5, 15)
        sequence = torch.randn(length, 4)  # (seq_len, features)
        label = torch.tensor(idx % 2)
        return sequence, label

def padding_collate_fn(batch):
    """Pad sequences to max length in batch"""
    sequences, labels = zip(*batch)

    # Get lengths
    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)

    # Pad sequences
    padded = torch.zeros(len(sequences), max_len, sequences[0].size(-1))
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    # Stack labels
    labels = torch.stack(labels)

    # Return lengths too (useful for pack_padded_sequence)
    lengths = torch.tensor(lengths)

    return padded, labels, lengths

var_dataset = VariableLengthDataset(10)
var_loader = DataLoader(var_dataset, batch_size=4, collate_fn=padding_collate_fn)

for padded, labels, lengths in var_loader:
    print(f"Padded batch shape: {padded.shape}")
    print(f"Original lengths: {lengths.tolist()}")
    break


# 4.3 Dictionary collate (for transformers)
print("\n--- 4.3 Dictionary Collate (Transformer Style) ---")

class DictDataset(Dataset):
    """Dataset returning dictionaries"""

    def __init__(self, num_samples, vocab_size=100, max_len=20):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        length = np.random.randint(5, self.max_len)
        input_ids = torch.randint(0, self.vocab_size, (length,))
        attention_mask = torch.ones(length)
        label = torch.tensor(idx % 2)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

def dict_collate_fn(batch):
    """Collate dictionaries with padding"""
    # Find max length
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        length = item['input_ids'].size(0)
        padding = max_len - length

        # Pad input_ids with 0
        input_ids.append(
            torch.cat([item['input_ids'], torch.zeros(padding, dtype=torch.long)])
        )
        # Pad attention_mask with 0
        attention_masks.append(
            torch.cat([item['attention_mask'], torch.zeros(padding)])
        )
        labels.append(item['label'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }

dict_dataset = DictDataset(10)
dict_loader = DataLoader(dict_dataset, batch_size=4, collate_fn=dict_collate_fn)

for batch in dict_loader:
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    break


# =============================================================================
# SECTION 5: SAMPLERS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: SAMPLERS")
print("=" * 70)

# 5.1 Built-in samplers
print("\n--- 5.1 Built-in Samplers ---")

data = torch.randn(20, 5)
labels = torch.tensor([0]*15 + [1]*5)  # Imbalanced: 15 class-0, 5 class-1
dataset = SimpleDataset(data, labels)

# Sequential
seq_loader = DataLoader(dataset, batch_size=5, sampler=SequentialSampler(dataset))
print("SequentialSampler - first batch labels:", next(iter(seq_loader))[1].tolist())

# Random
rand_loader = DataLoader(dataset, batch_size=5, sampler=RandomSampler(dataset))
print("RandomSampler - first batch labels:", next(iter(rand_loader))[1].tolist())

# 5.2 Weighted sampler for imbalanced classes
print("\n--- 5.2 Weighted Sampler for Imbalanced Classes ---")

# Count per class
class_counts = torch.bincount(labels)
print(f"Class distribution: {class_counts.tolist()}")

# Class weights (inverse frequency)
class_weights = 1.0 / class_counts.float()
print(f"Class weights: {class_weights.tolist()}")

# Sample weights
sample_weights = class_weights[labels]

# Weighted sampler
weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True  # Can sample same item multiple times
)

weighted_loader = DataLoader(dataset, batch_size=10, sampler=weighted_sampler)

# Check class balance
all_labels = []
for _, batch_labels in weighted_loader:
    all_labels.extend(batch_labels.tolist())

print(f"After weighted sampling - class 0: {all_labels.count(0)}, class 1: {all_labels.count(1)}")
print("Classes are now more balanced!")


# 5.3 Custom sampler - Bucket sampling
print("\n--- 5.3 Custom Bucket Sampler ---")

class BucketBatchSampler(Sampler):
    """
    Group samples by length for efficient batching.
    Reduces padding waste.
    """

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Sort indices by length
        indices = np.argsort(self.lengths)

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batches.append(indices[i:i + self.batch_size].tolist())

        # Shuffle batches (not samples within batch)
        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

# Example
lengths = [3, 5, 8, 2, 10, 7, 4, 9, 1, 6]
bucket_sampler = BucketBatchSampler(lengths, batch_size=3)

print("Bucket batches (indices grouped by length):")
for batch in bucket_sampler:
    batch_lengths = [lengths[i] for i in batch]
    print(f"  Indices: {batch}, Lengths: {batch_lengths}")


# =============================================================================
# SECTION 6: EFFICIENT DATA LOADING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: EFFICIENT DATA LOADING")
print("=" * 70)

# 6.1 Best practices
print("\n--- 6.1 DataLoader Best Practices ---")
print("""
Training loader:
  DataLoader(
      train_dataset,
      batch_size=32,
      shuffle=True,           # Randomize each epoch
      num_workers=4,          # Parallel loading
      pin_memory=True,        # Faster GPU transfer
      drop_last=True,         # Consistent batch sizes
      persistent_workers=True # Keep workers between epochs
  )

Validation loader:
  DataLoader(
      val_dataset,
      batch_size=64,          # Larger (no gradients)
      shuffle=False,          # Reproducible
      num_workers=4,
      pin_memory=True,
      drop_last=False         # Evaluate ALL samples
  )
""")

# 6.2 Memory-mapped dataset
print("\n--- 6.2 Memory-Mapped Dataset for Large Files ---")

class MemmapDataset(Dataset):
    """
    Memory-mapped dataset for files too large for RAM.
    Only loads data when accessed.
    """

    def __init__(self, file_path, dtype=np.float32, shape=None):
        # Memory map the file (doesn't load into RAM)
        self.data = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load only this sample
        return torch.from_numpy(self.data[idx].copy())

print("MemmapDataset: Load data on-demand from disk")
print("Useful for datasets larger than available RAM")


# 6.3 Pre-tokenized dataset for LLM training
print("\n--- 6.3 Pre-Tokenized Dataset (LLM Training) ---")

class PreTokenizedDataset(Dataset):
    """
    Efficient dataset for pre-tokenized data.
    Common pattern in LLM training.
    """

    def __init__(self, token_ids, block_size):
        self.token_ids = np.array(token_ids, dtype=np.int32)
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size - 1

    def __getitem__(self, idx):
        # Slice numpy array (efficient)
        chunk = self.token_ids[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

# Demo
tokens = list(range(1000))
pretok_dataset = PreTokenizedDataset(tokens, block_size=128)
x, y = pretok_dataset[0]
print(f"Pre-tokenized dataset: {len(pretok_dataset)} samples")
print(f"x shape: {x.shape}, y shape: {y.shape}")


# =============================================================================
# SECTION 7: COMPLETE EXAMPLE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: COMPLETE TRAINING EXAMPLE")
print("=" * 70)

print("""
# Complete training loop with DataLoader

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        # Move to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs, labels = batch['input_ids'], batch['labels']
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs, labels = batch['input_ids'], batch['labels']
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)


# Main training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
""")


# =============================================================================
# SECTION 8: INSTRUCTION TUNING DATASET
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: INSTRUCTION TUNING DATASET (FOR FINE-TUNING LLMs)")
print("=" * 70)

class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning (like Alpaca, Vicuna).

    Format:
    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    {output}
    """

    def __init__(self, data, tokenizer_fn, max_length=512):
        self.data = data
        self.tokenize = tokenizer_fn
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def format_prompt(self, item):
        prompt = f"### Instruction:\n{item['instruction']}\n\n"
        if item.get('input'):
            prompt += f"### Input:\n{item['input']}\n\n"
        prompt += "### Response:\n"
        return prompt

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = self.format_prompt(item)
        response = item['output']

        # Tokenize
        prompt_tokens = self.tokenize(prompt)
        response_tokens = self.tokenize(response)

        # Combine
        input_ids = prompt_tokens + response_tokens
        input_ids = input_ids[:self.max_length]

        # Labels: -100 for prompt (no loss), actual tokens for response
        labels = [-100] * len(prompt_tokens) + response_tokens
        labels = labels[:self.max_length]

        # Pad
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        labels = labels + [-100] * pad_len

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels)
        }

# Example
sample_data = [
    {
        'instruction': 'Write a haiku about programming',
        'input': '',
        'output': 'Code flows like water\nBugs hide in the syntax maze\nStack overflow saves'
    }
]

# Simple tokenizer for demo
def simple_tokenize(text):
    return [ord(c) % 100 for c in text]  # Simple char-based

inst_dataset = InstructionDataset(sample_data, simple_tokenize, max_length=64)
sample = inst_dataset[0]
print("Instruction dataset sample:")
print(f"  input_ids shape: {sample['input_ids'].shape}")
print(f"  labels shape: {sample['labels'].shape}")
print(f"  Prompt tokens (labels=-100): {(sample['labels'] == -100).sum().item()}")
print(f"  Response tokens: {(sample['labels'] != -100).sum().item()}")


print("\n" + "=" * 70)
print("END OF MODULE 1.4")
print("=" * 70)
