"""
Module 5C: Text Preprocessing & Data Pipeline
=============================================

This module provides hands-on implementations of:
1. Text cleaning and normalization
2. Building PyTorch datasets for NLP
3. Collation functions for variable-length sequences
4. Efficient data loading with DataLoaders
5. Data augmentation techniques
6. Production-ready preprocessing pipelines

Run this file to see all preprocessing concepts in action!
"""

import re
import random
import unicodedata
from typing import List, Dict, Optional, Tuple, Callable
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np


# =============================================================================
# 1. TEXT CLEANING UTILITIES
# =============================================================================

class TextCleaner:
    """
    Comprehensive text cleaning utilities.

    Real-world text is messy! This class handles common issues:
    - HTML tags and entities
    - Encoding problems
    - Whitespace normalization
    - URL handling
    - Special character normalization
    """

    # Common HTML entities
    HTML_ENTITIES = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
    }

    # Unicode character normalization
    UNICODE_REPLACEMENTS = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u200b': '',   # Zero-width space
        '\ufeff': '',   # BOM
    }

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text."""
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        return text

    @staticmethod
    def decode_html_entities(text: str) -> str:
        """Decode HTML entities."""
        # Handle named entities
        for entity, replacement in TextCleaner.HTML_ENTITIES.items():
            text = text.replace(entity, replacement)

        # Handle numeric entities
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)

        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters to ASCII equivalents."""
        # Replace specific unicode characters
        for char, replacement in TextCleaner.UNICODE_REPLACEMENTS.items():
            text = text.replace(char, replacement)

        # Normalize unicode to NFKC form
        text = unicodedata.normalize('NFKC', text)

        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize various whitespace characters."""
        # Replace all whitespace types with regular space
        text = re.sub(r'[\t\r\f\v]', ' ', text)

        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def handle_urls(text: str, replacement: str = '<URL>') -> str:
        """Replace or remove URLs."""
        url_pattern = r'https?://[^\s<>"\'\)]+|www\.[^\s<>"\'\)]+'
        text = re.sub(url_pattern, replacement, text)
        return text

    @staticmethod
    def handle_emails(text: str, replacement: str = '<EMAIL>') -> str:
        """Replace or remove email addresses."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, replacement, text)
        return text

    @staticmethod
    def handle_numbers(text: str, replacement: str = '<NUM>') -> str:
        """Replace numbers with token."""
        # Replace numbers (integers and decimals)
        text = re.sub(r'\b\d+\.?\d*\b', replacement, text)
        return text

    @staticmethod
    def remove_extra_punctuation(text: str) -> str:
        """Normalize repeated punctuation."""
        # Replace multiple punctuation with single
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        return text

    @classmethod
    def clean(cls, text: str,
              remove_html: bool = True,
              normalize_unicode_chars: bool = True,
              normalize_ws: bool = True,
              handle_url: bool = True,
              handle_email: bool = False,
              handle_num: bool = False,
              lowercase: bool = False) -> str:
        """
        Apply all cleaning steps.

        Args:
            text: Input text
            remove_html: Remove HTML tags and decode entities
            normalize_unicode_chars: Normalize unicode to ASCII equivalents
            normalize_ws: Normalize whitespace
            handle_url: Replace URLs with <URL>
            handle_email: Replace emails with <EMAIL>
            handle_num: Replace numbers with <NUM>
            lowercase: Convert to lowercase

        Returns:
            Cleaned text
        """
        if remove_html:
            text = cls.remove_html_tags(text)
            text = cls.decode_html_entities(text)

        if normalize_unicode_chars:
            text = cls.normalize_unicode(text)

        if handle_url:
            text = cls.handle_urls(text)

        if handle_email:
            text = cls.handle_emails(text)

        if handle_num:
            text = cls.handle_numbers(text)

        if normalize_ws:
            text = cls.normalize_whitespace(text)

        if lowercase:
            text = text.lower()

        return text


# =============================================================================
# 2. PYTORCH DATASETS FOR NLP
# =============================================================================

class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks.

    Handles:
    - Text preprocessing
    - Tokenization
    - Padding/truncation
    - Label encoding
    """

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 tokenizer,
                 max_length: int = 512,
                 preprocessing_fn: Optional[Callable] = None):
        """
        Args:
            texts: List of input texts
            labels: List of labels (integers)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            preprocessing_fn: Optional preprocessing function
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing_fn = preprocessing_fn or (lambda x: x)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Apply preprocessing
        text = self.preprocessing_fn(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LanguageModelingDataset(Dataset):
    """
    Dataset for causal language modeling (GPT-style).

    Creates (input, target) pairs where target is input shifted by 1.
    """

    def __init__(self,
                 text: str,
                 tokenizer,
                 block_size: int = 512,
                 preprocessing_fn: Optional[Callable] = None):
        """
        Args:
            text: Training text (can be very long)
            tokenizer: Hugging Face tokenizer
            block_size: Length of each training example
            preprocessing_fn: Optional preprocessing function
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Preprocess
        if preprocessing_fn:
            text = preprocessing_fn(text)

        # Tokenize entire text at once (efficient)
        tokens = tokenizer.encode(text)

        # Create examples by chunking
        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size // 2):  # 50% overlap
            chunk = tokens[i:i + block_size + 1]
            if len(chunk) == block_size + 1:
                self.examples.append(chunk)

        print(f"Created {len(self.examples)} training examples from {len(tokens)} tokens")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]

        # Input: all tokens except last
        # Labels: all tokens except first (shifted by 1)
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }


class Seq2SeqDataset(Dataset):
    """
    Dataset for sequence-to-sequence tasks (translation, summarization).
    """

    def __init__(self,
                 sources: List[str],
                 targets: List[str],
                 tokenizer,
                 max_source_length: int = 512,
                 max_target_length: int = 128,
                 preprocessing_fn: Optional[Callable] = None):
        """
        Args:
            sources: Source texts
            targets: Target texts
            tokenizer: Hugging Face tokenizer
            max_source_length: Max length for source
            max_target_length: Max length for target
            preprocessing_fn: Optional preprocessing function
        """
        assert len(sources) == len(targets), "Sources and targets must have same length"

        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.preprocessing_fn = preprocessing_fn or (lambda x: x)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source = self.preprocessing_fn(self.sources[idx])
        target = self.preprocessing_fn(self.targets[idx])

        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors='pt'
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0)
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction-tuning (ChatGPT-style).

    Handles formatting of instruction + input + output.
    """

    PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

    PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
"""

    def __init__(self,
                 data: List[Dict],
                 tokenizer,
                 max_length: int = 2048):
        """
        Args:
            data: List of dicts with 'instruction', 'input' (optional), 'output'
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Format prompt
        if item.get('input') and item['input'].strip():
            prompt = self.PROMPT_TEMPLATE.format(
                instruction=item['instruction'],
                input=item['input']
            )
        else:
            prompt = self.PROMPT_TEMPLATE_NO_INPUT.format(
                instruction=item['instruction']
            )

        full_text = prompt + item['output']

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Create labels (mask the prompt part with -100)
        labels = encoding['input_ids'].clone()
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        labels[0, :prompt_len] = -100  # Don't compute loss on prompt

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


# =============================================================================
# 3. COLLATION FUNCTIONS
# =============================================================================

class DynamicPaddingCollator:
    """
    Collator that pads to the longest sequence in the batch (dynamic padding).

    Much more efficient than padding to a fixed max length!
    """

    def __init__(self, tokenizer, padding_side: str = 'right'):
        """
        Args:
            tokenizer: Hugging Face tokenizer (for pad_token_id)
            padding_side: 'right' for encoder, 'left' for decoder generation
        """
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic padding."""

        # Find max length in batch
        max_length = max(len(item['input_ids']) for item in batch)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item.get('attention_mask', torch.ones(len(input_ids)))
            padding_length = max_length - len(input_ids)

            if self.padding_side == 'right':
                # Right padding (for encoders)
                padded_input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id)
                ])
                padded_attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length)
                ])
            else:
                # Left padding (for decoder generation)
                padded_input_ids = torch.cat([
                    torch.full((padding_length,), self.pad_token_id),
                    input_ids
                ])
                padded_attention_mask = torch.cat([
                    torch.zeros(padding_length),
                    attention_mask
                ])

            input_ids_batch.append(padded_input_ids)
            attention_mask_batch.append(padded_attention_mask)

            # Handle labels if present
            if 'labels' in item:
                labels = item['labels']
                if self.padding_side == 'right':
                    padded_labels = torch.cat([
                        labels,
                        torch.full((padding_length,), -100)  # -100 = ignore in loss
                    ])
                else:
                    padded_labels = torch.cat([
                        torch.full((padding_length,), -100),
                        labels
                    ])
                labels_batch.append(padded_labels)

        result = {
            'input_ids': torch.stack(input_ids_batch),
            'attention_mask': torch.stack(attention_mask_batch).long()
        }

        if labels_batch:
            result['labels'] = torch.stack(labels_batch)

        return result


class LanguageModelingCollator:
    """
    Collator for language modeling with optional masking (BERT-style).
    """

    def __init__(self, tokenizer, mlm: bool = False, mlm_probability: float = 0.15):
        """
        Args:
            tokenizer: Hugging Face tokenizer
            mlm: Whether to do masked language modeling
            mlm_probability: Probability of masking tokens (if mlm=True)
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack all inputs
        input_ids = torch.stack([item['input_ids'] for item in batch])

        if 'labels' in batch[0]:
            labels = torch.stack([item['labels'] for item in batch])
        else:
            labels = input_ids.clone()

        # Apply masking for MLM
        if self.mlm:
            input_ids, labels = self._mask_tokens(input_ids, labels)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def _mask_tokens(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply BERT-style masking: 80% [MASK], 10% random, 10% unchanged.
        """
        inputs = inputs.clone()
        labels = labels.clone()

        # Create probability matrix
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for special_id in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                          self.tokenizer.sep_token_id]:
            if special_id is not None:
                special_tokens_mask |= (inputs == special_id)

        probability_matrix[special_tokens_mask] = 0.0

        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% -> random token
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 10% -> unchanged (already handled)

        return inputs, labels


# =============================================================================
# 4. EFFICIENT DATA LOADING
# =============================================================================

class BucketBatchSampler(Sampler):
    """
    Sampler that groups similar-length sequences to minimize padding.

    This can provide significant speedups by reducing wasted computation on padding!
    """

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Args:
            lengths: List of sequence lengths
            batch_size: Batch size
            shuffle: Whether to shuffle batches
            drop_last: Whether to drop the last incomplete batch
        """
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Sort indices by length
        sorted_indices = np.argsort(self.lengths)

        # Create batches
        batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch = sorted_indices[i:i + self.batch_size].tolist()
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # Shuffle batch order (not within batches)
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def create_efficient_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    use_bucketing: bool = False
) -> DataLoader:
    """
    Create an efficient DataLoader with best practices.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        collate_fn: Custom collation function
        pin_memory: Pin memory for faster GPU transfer
        use_bucketing: Use bucket batch sampler for efficiency

    Returns:
        DataLoader instance
    """
    if use_bucketing and hasattr(dataset, 'get_lengths'):
        # Use bucketing for efficiency
        lengths = dataset.get_lengths()
        batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )


# =============================================================================
# 5. DATA AUGMENTATION
# =============================================================================

class TextAugmenter:
    """
    Text augmentation techniques for NLP.

    Includes:
    - Random deletion
    - Random swap
    - Synonym replacement (requires NLTK)
    - Random insertion
    """

    def __init__(self, use_synonyms: bool = False):
        """
        Args:
            use_synonyms: Whether to use synonym-based augmentation (requires NLTK)
        """
        self.use_synonyms = use_synonyms
        if use_synonyms:
            try:
                import nltk
                from nltk.corpus import wordnet
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                self.wordnet = wordnet
            except ImportError:
                print("NLTK not available, synonym replacement disabled")
                self.use_synonyms = False

    def random_deletion(self, words: List[str], p: float = 0.1) -> List[str]:
        """Randomly delete words with probability p."""
        if len(words) <= 1:
            return words

        return [w for w in words if random.random() > p]

    def random_swap(self, words: List[str], n: int = 1) -> List[str]:
        """Randomly swap n pairs of words."""
        if len(words) < 2:
            return words

        words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return words

    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        if not self.use_synonyms:
            return []

        synonyms = []
        for syn in self.wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.append(synonym)

        return list(set(synonyms))

    def synonym_replacement(self, words: List[str], n: int = 1) -> List[str]:
        """Replace n words with synonyms."""
        if not self.use_synonyms:
            return words

        words = words.copy()
        random_word_indices = list(range(len(words)))
        random.shuffle(random_word_indices)

        num_replaced = 0
        for idx in random_word_indices:
            synonyms = self.get_synonyms(words[idx])
            if synonyms:
                words[idx] = random.choice(synonyms)
                num_replaced += 1
                if num_replaced >= n:
                    break

        return words

    def random_insertion(self, words: List[str], n: int = 1) -> List[str]:
        """Insert n random synonyms at random positions."""
        if not self.use_synonyms:
            return words

        words = words.copy()
        for _ in range(n):
            if not words:
                break

            # Pick a random word and get its synonym
            random_word = random.choice(words)
            synonyms = self.get_synonyms(random_word)

            if synonyms:
                # Insert at random position
                random_synonym = random.choice(synonyms)
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, random_synonym)

        return words

    def augment(self, text: str,
                alpha_sr: float = 0.1,
                alpha_rd: float = 0.1,
                alpha_rs: float = 0.1,
                alpha_ri: float = 0.1,
                num_aug: int = 1) -> List[str]:
        """
        Apply Easy Data Augmentation (EDA) to text.

        Args:
            text: Input text
            alpha_sr: Proportion of words for synonym replacement
            alpha_rd: Probability of random deletion
            alpha_rs: Proportion of words for random swap
            alpha_ri: Proportion of words for random insertion
            num_aug: Number of augmented texts to generate

        Returns:
            List of augmented texts
        """
        words = text.split()
        n = len(words)

        augmented = []
        for _ in range(num_aug):
            new_words = words.copy()

            # Synonym replacement
            if self.use_synonyms and alpha_sr > 0:
                n_sr = max(1, int(alpha_sr * n))
                new_words = self.synonym_replacement(new_words, n_sr)

            # Random insertion
            if self.use_synonyms and alpha_ri > 0:
                n_ri = max(1, int(alpha_ri * n))
                new_words = self.random_insertion(new_words, n_ri)

            # Random swap
            if alpha_rs > 0:
                n_rs = max(1, int(alpha_rs * n))
                new_words = self.random_swap(new_words, n_rs)

            # Random deletion
            if alpha_rd > 0:
                new_words = self.random_deletion(new_words, alpha_rd)

            if new_words:
                augmented.append(' '.join(new_words))

        return augmented


class CharacterAugmenter:
    """
    Character-level augmentation for robustness to typos.
    """

    # Common keyboard adjacency for typo simulation
    KEYBOARD_NEIGHBORS = {
        'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc',
        'e': 'wrsd', 'f': 'rtdgvc', 'g': 'tyfhbv', 'h': 'yugjbn',
        'i': 'uojk', 'j': 'uihknm', 'k': 'iojlm', 'l': 'opk',
        'm': 'njk', 'n': 'bhjm', 'o': 'iplk', 'p': 'ol',
        'q': 'wa', 'r': 'etdf', 's': 'weadzx', 't': 'ryfg',
        'u': 'yihj', 'v': 'cfgb', 'w': 'qeas', 'x': 'zsdc',
        'y': 'tugh', 'z': 'asx'
    }

    def random_typo(self, text: str, p: float = 0.05) -> str:
        """Introduce random typos."""
        chars = list(text)

        for i in range(len(chars)):
            if random.random() < p and chars[i].lower() in self.KEYBOARD_NEIGHBORS:
                neighbors = self.KEYBOARD_NEIGHBORS[chars[i].lower()]
                new_char = random.choice(neighbors)
                if chars[i].isupper():
                    new_char = new_char.upper()
                chars[i] = new_char

        return ''.join(chars)

    def random_delete_char(self, text: str, p: float = 0.02) -> str:
        """Randomly delete characters."""
        return ''.join(c for c in text if random.random() > p)

    def random_duplicate_char(self, text: str, p: float = 0.02) -> str:
        """Randomly duplicate characters (common typo)."""
        result = []
        for c in text:
            result.append(c)
            if random.random() < p and c.isalpha():
                result.append(c)
        return ''.join(result)


# =============================================================================
# 6. PRODUCTION PIPELINE
# =============================================================================

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for production use.

    Ensures consistency between training and inference.
    """

    def __init__(self,
                 tokenizer,
                 max_length: int = 512,
                 clean_html: bool = True,
                 normalize_unicode: bool = True,
                 lowercase: bool = False,
                 handle_urls: bool = True):
        """
        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            clean_html: Whether to remove HTML
            normalize_unicode: Whether to normalize unicode
            lowercase: Whether to lowercase
            handle_urls: Whether to replace URLs
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clean_html = clean_html
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
        self.handle_urls = handle_urls

        # Save config for reproducibility
        self.config = {
            'max_length': max_length,
            'clean_html': clean_html,
            'normalize_unicode': normalize_unicode,
            'lowercase': lowercase,
            'handle_urls': handle_urls,
            'tokenizer_name': getattr(tokenizer, 'name_or_path', 'unknown')
        }

    def preprocess(self, text: str) -> str:
        """Apply preprocessing steps."""
        return TextCleaner.clean(
            text,
            remove_html=self.clean_html,
            normalize_unicode_chars=self.normalize_unicode,
            lowercase=self.lowercase,
            handle_url=self.handle_urls
        )

    def encode(self, text: str, return_tensors: str = 'pt') -> Dict:
        """Preprocess and tokenize."""
        text = self.preprocess(text)
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )

    def encode_batch(self, texts: List[str], return_tensors: str = 'pt') -> Dict:
        """Preprocess and tokenize a batch."""
        texts = [self.preprocess(t) for t in texts]
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=return_tensors
        )

    def save_config(self, path: str):
        """Save preprocessing config for reproducibility."""
        import json
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load_config(cls, path: str, tokenizer) -> 'PreprocessingPipeline':
        """Load pipeline from saved config."""
        import json
        with open(path, 'r') as f:
            config = json.load(f)

        return cls(
            tokenizer=tokenizer,
            max_length=config['max_length'],
            clean_html=config['clean_html'],
            normalize_unicode=config['normalize_unicode'],
            lowercase=config['lowercase'],
            handle_urls=config['handle_urls']
        )


# =============================================================================
# MAIN: RUN ALL DEMOS
# =============================================================================

def main():
    """Run all preprocessing demonstrations."""

    print("=" * 70)
    print("MODULE 5C: TEXT PREPROCESSING & DATA PIPELINE - DEMONSTRATIONS")
    print("=" * 70)

    # ===================
    # 1. Text Cleaning Demo
    # ===================
    print("\n\n" + "=" * 70)
    print("1. TEXT CLEANING")
    print("=" * 70)

    messy_texts = [
        "<p>Hello &amp; welcome to the    website!</p>",
        "Check out https://example.com for more info!!!",
        "This has "fancy quotes" and an em—dash",
        "Multiple   spaces   and\n\n\nmany newlines",
        "Email me at test@example.com for questions",
        "The price is $19.99 for 100 items",
    ]

    print("\nCleaning examples:")
    for text in messy_texts:
        cleaned = TextCleaner.clean(text)
        print(f"\n  Original: {repr(text)}")
        print(f"  Cleaned:  {repr(cleaned)}")

    # ===================
    # 2. Datasets Demo
    # ===================
    print("\n\n" + "=" * 70)
    print("2. PYTORCH DATASETS FOR NLP")
    print("=" * 70)

    # Create mock tokenizer for demo (simple word-based)
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            self.pad_token_id = 0
            self.mask_token_id = 2
            self.cls_token_id = 3
            self.sep_token_id = 4

        def __call__(self, text, truncation=True, max_length=512, return_tensors=None, padding=False):
            if isinstance(text, list):
                return self._batch_encode(text, truncation, max_length, return_tensors, padding)
            return self._encode(text, truncation, max_length, return_tensors)

        def _encode(self, text, truncation, max_length, return_tensors):
            words = text.lower().split()
            ids = []
            for word in words[:max_length]:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                ids.append(self.vocab[word])

            result = {
                'input_ids': ids,
                'attention_mask': [1] * len(ids)
            }

            if return_tensors == 'pt':
                result['input_ids'] = torch.tensor([result['input_ids']])
                result['attention_mask'] = torch.tensor([result['attention_mask']])

            return result

        def _batch_encode(self, texts, truncation, max_length, return_tensors, padding):
            encodings = [self._encode(t, truncation, max_length, None) for t in texts]

            if padding:
                max_len = max(len(e['input_ids']) for e in encodings)
                for e in encodings:
                    pad_len = max_len - len(e['input_ids'])
                    e['input_ids'] += [0] * pad_len
                    e['attention_mask'] += [0] * pad_len

            if return_tensors == 'pt':
                return {
                    'input_ids': torch.tensor([e['input_ids'] for e in encodings]),
                    'attention_mask': torch.tensor([e['attention_mask'] for e in encodings])
                }
            return encodings

        def encode(self, text):
            words = text.lower().split()
            return [self.vocab.get(w, 1) for w in words]

    tokenizer = SimpleTokenizer()

    # Classification dataset
    print("\nClassification Dataset:")
    texts = ["This is great!", "This is terrible.", "Okay product"]
    labels = [1, 0, 1]

    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=10)
    print(f"  Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")

    # Language modeling dataset
    print("\nLanguage Modeling Dataset:")
    text = "The quick brown fox jumps over the lazy dog. " * 10
    lm_dataset = LanguageModelingDataset(text, tokenizer, block_size=20)
    print(f"  Dataset size: {len(lm_dataset)}")
    if len(lm_dataset) > 0:
        sample = lm_dataset[0]
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")

    # ===================
    # 3. Collation Demo
    # ===================
    print("\n\n" + "=" * 70)
    print("3. COLLATION FUNCTIONS")
    print("=" * 70)

    # Create variable-length batch
    batch = [
        {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor(0)},
        {'input_ids': torch.tensor([4, 5, 6, 7, 8]), 'labels': torch.tensor(1)},
        {'input_ids': torch.tensor([9, 10]), 'labels': torch.tensor(0)},
    ]

    print("\nOriginal batch (variable lengths):")
    for i, item in enumerate(batch):
        print(f"  Item {i}: length = {len(item['input_ids'])}")

    # Right padding (for encoders)
    collator_right = DynamicPaddingCollator(tokenizer, padding_side='right')
    padded_right = collator_right(batch)
    print("\nRight-padded (for encoders like BERT):")
    print(f"  Input IDs:\n{padded_right['input_ids']}")
    print(f"  Attention mask:\n{padded_right['attention_mask']}")

    # Left padding (for decoders)
    collator_left = DynamicPaddingCollator(tokenizer, padding_side='left')
    padded_left = collator_left(batch)
    print("\nLeft-padded (for decoders like GPT):")
    print(f"  Input IDs:\n{padded_left['input_ids']}")
    print(f"  Attention mask:\n{padded_left['attention_mask']}")

    # ===================
    # 4. Efficient Loading Demo
    # ===================
    print("\n\n" + "=" * 70)
    print("4. EFFICIENT DATA LOADING")
    print("=" * 70)

    # Create dataset with varying lengths
    class VarLengthDataset(Dataset):
        def __init__(self, n=100):
            self.lengths = [random.randint(5, 50) for _ in range(n)]

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, idx):
            length = self.lengths[idx]
            return {
                'input_ids': torch.randint(1, 1000, (length,)),
                'attention_mask': torch.ones(length),
                'labels': torch.tensor(idx % 2)
            }

        def get_lengths(self):
            return self.lengths

    var_dataset = VarLengthDataset(100)

    print("\nBucket Batch Sampler (groups similar lengths):")
    lengths = var_dataset.get_lengths()
    sampler = BucketBatchSampler(lengths, batch_size=10)

    batches = list(sampler)
    for i, batch in enumerate(batches[:3]):
        batch_lengths = [lengths[idx] for idx in batch]
        print(f"  Batch {i}: lengths = {batch_lengths}")
        print(f"           range = {max(batch_lengths) - min(batch_lengths)}")

    # ===================
    # 5. Data Augmentation Demo
    # ===================
    print("\n\n" + "=" * 70)
    print("5. DATA AUGMENTATION")
    print("=" * 70)

    augmenter = TextAugmenter(use_synonyms=False)  # Without NLTK dependency

    text = "The quick brown fox jumps over the lazy dog"
    print(f"\nOriginal: {text}")

    # Random deletion
    words = text.split()
    deleted = augmenter.random_deletion(words, p=0.2)
    print(f"Random deletion: {' '.join(deleted)}")

    # Random swap
    swapped = augmenter.random_swap(words, n=2)
    print(f"Random swap: {' '.join(swapped)}")

    # Character augmentation
    char_augmenter = CharacterAugmenter()
    typo_text = char_augmenter.random_typo(text, p=0.1)
    print(f"With typos: {typo_text}")

    dup_text = char_augmenter.random_duplicate_char(text, p=0.1)
    print(f"With duplicates: {dup_text}")

    # ===================
    # 6. Production Pipeline Demo
    # ===================
    print("\n\n" + "=" * 70)
    print("6. PRODUCTION PIPELINE")
    print("=" * 70)

    pipeline = PreprocessingPipeline(
        tokenizer=tokenizer,
        max_length=512,
        clean_html=True,
        normalize_unicode=True,
        lowercase=True,
        handle_urls=True
    )

    test_input = "<p>Check out https://example.com for more INFO!</p>"
    print(f"\nInput: {test_input}")
    preprocessed = pipeline.preprocess(test_input)
    print(f"Preprocessed: {preprocessed}")

    encoded = pipeline.encode(test_input)
    print(f"Encoded keys: {list(encoded.keys())}")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")

    print(f"\nPipeline config: {pipeline.config}")

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Takeaways:

1. TEXT CLEANING is essential:
   - Handle encoding issues, HTML, whitespace
   - Normalize unicode characters
   - Be consistent between train/inference!

2. DATASET CLASSES for different tasks:
   - TextClassificationDataset: (text, label) pairs
   - LanguageModelingDataset: Next token prediction
   - Seq2SeqDataset: Source-target pairs
   - InstructionDataset: Instruction tuning

3. COLLATION handles variable lengths:
   - Dynamic padding (to longest in batch) is efficient
   - Right-pad for encoders (BERT)
   - Left-pad for decoders (GPT generation)
   - Use attention masks to ignore padding

4. EFFICIENT LOADING:
   - Use multiple workers (num_workers=4)
   - Pin memory for GPU (pin_memory=True)
   - Bucket similar lengths together
   - Prefetch batches

5. AUGMENTATION for small datasets:
   - Random deletion, swap, insertion
   - Synonym replacement (with NLTK)
   - Character-level typos for robustness

6. PRODUCTION PIPELINES:
   - Save preprocessing config
   - Use same preprocessing for train/inference
   - Batch efficiently for inference
""")


if __name__ == "__main__":
    main()
