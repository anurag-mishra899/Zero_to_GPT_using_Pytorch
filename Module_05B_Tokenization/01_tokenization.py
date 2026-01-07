"""
Module 5B: Tokenization - From Text to Tokens
============================================

This module provides hands-on implementations of:
1. Character-level tokenization
2. Word-level tokenization
3. BPE (Byte Pair Encoding) from scratch
4. Using Hugging Face tokenizers
5. Using tiktoken (OpenAI)
6. Training custom tokenizers
7. Practical tokenization utilities

Run this file to see all tokenization concepts in action!
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import json


# =============================================================================
# 1. BASIC TOKENIZATION APPROACHES
# =============================================================================

class CharacterTokenizer:
    """
    Simple character-level tokenizer.

    Every character is a token. Simple but creates very long sequences.

    Example:
        "Hello" -> ['H', 'e', 'l', 'l', 'o'] (5 tokens)
    """

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        chars = set()
        for text in texts:
            chars.update(text)

        # Sort for reproducibility
        chars = sorted(chars)

        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for idx, token in enumerate(special_tokens):
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

        # Add characters
        for idx, char in enumerate(chars, start=len(special_tokens)):
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.char_to_id.get(c, self.char_to_id['<UNK>']) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ''.join(self.id_to_char.get(id, '<UNK>') for id in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)


class WordTokenizer:
    """
    Simple word-level tokenizer.

    Splits on whitespace and punctuation. Creates smaller sequences but
    large vocabulary and OOV issues.

    Example:
        "Hello, world!" -> ['Hello', ',', 'world', '!'] (4 tokens)
    """

    def __init__(self, min_freq: int = 1):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.min_freq = min_freq

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words (simple regex-based)."""
        # Split on whitespace and keep punctuation as separate tokens
        pattern = r"(\w+|[^\w\s])"
        return re.findall(pattern, text)

    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        # Filter by frequency
        vocab = [w for w, c in word_counts.items() if c >= self.min_freq]
        vocab = sorted(vocab)

        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        # Add words
        for idx, word in enumerate(vocab, start=len(special_tokens)):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = self._tokenize(text)
        return [self.word_to_id.get(w, self.word_to_id['<UNK>']) for w in words]

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        words = [self.id_to_word.get(id, '<UNK>') for id in ids]
        # Simple joining (not perfect for punctuation)
        return ' '.join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)


# =============================================================================
# 2. BPE FROM SCRATCH
# =============================================================================

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer implemented from scratch.

    This is a simplified version that demonstrates the core BPE algorithm:
    1. Start with character-level vocabulary
    2. Iteratively merge most frequent adjacent pairs
    3. Stop when desired vocabulary size is reached

    Example:
        Training: ["low", "lower", "lowest"]
        Result: learns merges like 'l'+'o'->'lo', 'lo'+'w'->'low'
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []  # Ordered merge operations
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """Merge a pair in all words."""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def fit(self, texts: List[str], verbose: bool = False):
        """
        Train BPE on a corpus.

        Args:
            texts: List of training texts
            verbose: Print merge operations
        """
        # Step 1: Build word frequency dictionary
        # Add end-of-word marker </w> and split into characters
        word_freqs = defaultdict(int)
        for text in texts:
            words = text.strip().split()
            for word in words:
                # Add space between characters, add end-of-word marker
                word_with_eow = ' '.join(list(word)) + ' </w>'
                word_freqs[word_with_eow] += 1

        # Step 2: Build initial vocabulary (characters)
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        idx = len(self.vocab)

        for word in word_freqs:
            for char in word.split():
                if char not in self.vocab:
                    self.vocab[char] = idx
                    idx += 1

        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge this pair
            word_freqs = self._merge_vocab(best_pair, word_freqs)

            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = idx
                idx += 1

            # Save merge operation
            self.merges.append(best_pair)

            if verbose and i < 10:
                print(f"Merge {i+1}: {best_pair} -> '{merged_token}' (freq: {pairs[best_pair]})")

        # Build inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"\nFinal vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges."""
        # Start with characters
        word = ' '.join(list(word)) + ' </w>'

        # Apply merges in order
        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            word = word.replace(bigram, replacement)

        return word.split()

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = []
        words = text.strip().split()

        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.vocab['<UNK>']))

        return tokens

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.inverse_vocab.get(id, '<UNK>') for id in ids]

        # Join tokens and remove end-of-word markers
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')

        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Get tokens (not IDs) for text."""
        tokens = []
        words = text.strip().split()

        for word in words:
            tokens.extend(self._tokenize_word(word))

        return tokens


# =============================================================================
# 3. BYTE-LEVEL BPE (GPT-2 STYLE)
# =============================================================================

class ByteLevelBPE:
    """
    Simplified Byte-Level BPE similar to GPT-2.

    Key insight: Work at byte level (256 possible values) instead of
    character level (150K+ Unicode characters).

    This ensures:
    - No OOV tokens ever (any byte sequence is valid)
    - Small initial vocabulary (256)
    - Language-agnostic
    """

    # Mapping bytes to printable characters (GPT-2 style)
    # This makes the vocabulary more readable
    @staticmethod
    def bytes_to_unicode() -> Dict[int, str]:
        """
        Create a mapping from bytes to unicode characters.

        GPT-2 maps bytes 0-255 to printable unicode characters
        so the vocabulary is human-readable.
        """
        # Printable ASCII characters
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))

        cs = bs[:]
        n = 0

        # Map remaining bytes (0-32, etc.) to higher unicode
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        return {b: chr(c) for b, c in zip(bs, cs)}

    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []

    def _encode_text_to_bytes(self, text: str) -> str:
        """Convert text to byte-level representation."""
        text_bytes = text.encode('utf-8')
        return ''.join(self.byte_encoder[b] for b in text_bytes)

    def _decode_bytes_to_text(self, byte_text: str) -> str:
        """Convert byte-level representation back to text."""
        byte_values = bytes([self.byte_decoder[c] for c in byte_text])
        return byte_values.decode('utf-8', errors='replace')

    def fit(self, texts: List[str], verbose: bool = False):
        """Train byte-level BPE."""
        # Initialize vocabulary with all bytes
        self.vocab = {self.byte_encoder[i]: i for i in range(256)}
        idx = 256

        # Count word frequencies in byte representation
        word_freqs = defaultdict(int)
        for text in texts:
            # Simple word splitting (GPT-2 uses regex)
            words = text.split()
            for word in words:
                byte_word = ' '.join(self._encode_text_to_bytes(word))
                word_freqs[byte_word] += 1

        # BPE merges (same algorithm as before)
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            # Count pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq

            if not pairs:
                break

            # Merge most frequent
            best_pair = max(pairs, key=pairs.get)

            # Update word_freqs
            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)

            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

            # Add to vocabulary
            merged = ''.join(best_pair)
            if merged not in self.vocab:
                self.vocab[merged] = idx
                idx += 1
                self.merges.append(best_pair)

            if verbose and i < 5:
                print(f"Merge {i+1}: {best_pair} -> '{merged}'")

        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Convert to bytes
        byte_text = self._encode_text_to_bytes(text)

        # Start with individual bytes
        tokens = list(byte_text)

        # Apply merges
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i:i+2] = [''.join(pair)]
                else:
                    i += 1

        # Convert to IDs
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        byte_text = ''.join(inv_vocab.get(id, '') for id in ids)
        return self._decode_bytes_to_text(byte_text)


# =============================================================================
# 4. HUGGING FACE TOKENIZERS (PRODUCTION USE)
# =============================================================================

def demo_huggingface_tokenizers():
    """
    Demonstrate using Hugging Face tokenizers.

    This is what you'll use in practice - pre-trained tokenizers
    from popular models.
    """
    print("\n" + "=" * 60)
    print("HUGGING FACE TOKENIZERS DEMO")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install transformers: pip install transformers")
        return

    # Test text
    text = "Hello, I'm learning about tokenization! It's fascinating."

    # Different model tokenizers
    models = [
        ("bert-base-uncased", "BERT (WordPiece)"),
        ("gpt2", "GPT-2 (BPE)"),
        ("t5-small", "T5 (SentencePiece)"),
    ]

    for model_name, description in models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)

            print(f"\n{description}:")
            print(f"  Vocab size: {tokenizer.vocab_size:,}")
            print(f"  Tokens: {tokens}")
            print(f"  IDs: {ids}")
            print(f"  Decoded: {decoded}")
            print(f"  Num tokens: {len(tokens)}")
        except Exception as e:
            print(f"\n{description}: Could not load - {e}")

    # Demonstrate special tokens with BERT
    print("\n" + "-" * 40)
    print("SPECIAL TOKENS (BERT)")
    print("-" * 40)

    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Single sentence
        single = tokenizer("Hello world", return_tensors=None)
        print(f"Single sentence: {tokenizer.convert_ids_to_tokens(single['input_ids'])}")

        # Sentence pair
        pair = tokenizer("Hello", "World", return_tensors=None)
        print(f"Sentence pair: {tokenizer.convert_ids_to_tokens(pair['input_ids'])}")

        print(f"\nSpecial tokens:")
        print(f"  [CLS] = {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
        print(f"  [SEP] = {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
        print(f"  [PAD] = {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  [MASK] = {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    except Exception as e:
        print(f"Could not demo special tokens: {e}")


# =============================================================================
# 5. TIKTOKEN (OPENAI)
# =============================================================================

def demo_tiktoken():
    """
    Demonstrate using tiktoken (OpenAI's tokenizer).

    This is the tokenizer used by GPT-3.5, GPT-4, etc.
    Useful for counting tokens before API calls.
    """
    print("\n" + "=" * 60)
    print("TIKTOKEN (OPENAI) DEMO")
    print("=" * 60)

    try:
        import tiktoken
    except ImportError:
        print("Install tiktoken: pip install tiktoken")
        return

    # Different encodings
    encodings = [
        ("gpt2", "GPT-2"),
        ("cl100k_base", "GPT-3.5/GPT-4"),
    ]

    text = "Hello, I'm learning about tokenization! It's fascinating."

    for enc_name, description in encodings:
        try:
            enc = tiktoken.get_encoding(enc_name)
            tokens = enc.encode(text)
            decoded = enc.decode(tokens)

            print(f"\n{description} ({enc_name}):")
            print(f"  Tokens: {tokens}")
            print(f"  Num tokens: {len(tokens)}")
            print(f"  Decoded: {decoded}")

            # Show individual tokens
            print("  Token breakdown:")
            for token_id in tokens[:10]:  # First 10
                token_bytes = enc.decode_single_token_bytes(token_id)
                try:
                    token_str = token_bytes.decode('utf-8')
                except:
                    token_str = str(token_bytes)
                print(f"    {token_id} -> '{token_str}'")
        except Exception as e:
            print(f"\n{description}: Could not load - {e}")

    # Token counting utility
    print("\n" + "-" * 40)
    print("TOKEN COUNTING UTILITY")
    print("-" * 40)

    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """Count tokens for OpenAI API calls."""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    test_texts = [
        "Hello!",
        "This is a longer sentence with more words in it.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]

    for t in test_texts:
        count = count_tokens(t)
        print(f"'{t[:40]}...' -> {count} tokens" if len(t) > 40 else f"'{t}' -> {count} tokens")


# =============================================================================
# 6. TRAINING A CUSTOM TOKENIZER
# =============================================================================

def demo_custom_tokenizer_training():
    """
    Demonstrate training a custom tokenizer using Hugging Face tokenizers library.

    Useful when you have domain-specific text (medical, legal, code, etc.)
    """
    print("\n" + "=" * 60)
    print("TRAINING CUSTOM TOKENIZER")
    print("=" * 60)

    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    except ImportError:
        print("Install tokenizers: pip install tokenizers")
        return

    # Sample training corpus (in practice, use large corpus)
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts.",
        "BERT and GPT are popular transformer-based models.",
        "Tokenization is the first step in text processing.",
        "Subword tokenization helps handle rare words effectively.",
        "The vocabulary size affects model size and performance.",
    ] * 10  # Repeat for more training data

    # Method 1: BPE Tokenizer
    print("\n1. Training BPE Tokenizer:")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=500,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(corpus, trainer)

    test_text = "Machine learning models process natural language."
    output = tokenizer.encode(test_text)

    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Test: '{test_text}'")
    print(f"  Tokens: {output.tokens}")
    print(f"  IDs: {output.ids}")

    # Method 2: WordPiece Tokenizer
    print("\n2. Training WordPiece Tokenizer:")

    tokenizer_wp = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer_wp.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer_wp = trainers.WordPieceTrainer(
        vocab_size=500,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )

    tokenizer_wp.train_from_iterator(corpus, trainer_wp)

    output_wp = tokenizer_wp.encode(test_text)

    print(f"  Vocab size: {tokenizer_wp.get_vocab_size()}")
    print(f"  Test: '{test_text}'")
    print(f"  Tokens: {output_wp.tokens}")
    print(f"  IDs: {output_wp.ids}")

    # Save and load
    print("\n3. Saving and Loading:")
    tokenizer.save("custom_bpe_tokenizer.json")
    loaded_tokenizer = Tokenizer.from_file("custom_bpe_tokenizer.json")
    print(f"  Loaded tokenizer vocab size: {loaded_tokenizer.get_vocab_size()}")

    # Clean up
    import os
    os.remove("custom_bpe_tokenizer.json")


# =============================================================================
# 7. PRACTICAL UTILITIES
# =============================================================================

def analyze_tokenization(text: str, tokenizer_name: str = "gpt2"):
    """
    Analyze how a text is tokenized - useful for debugging.
    """
    print("\n" + "=" * 60)
    print(f"TOKENIZATION ANALYSIS: {tokenizer_name}")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
        print("Install transformers: pip install transformers")
        return
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return

    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)

    print(f"\nInput: '{text}'")
    print(f"Number of tokens: {len(tokens)}")
    print(f"\nToken-by-token breakdown:")
    print("-" * 50)

    for i, (token, id) in enumerate(zip(tokens, ids[1:-1] if 'bert' in tokenizer_name.lower() else ids)):
        # Decode individual token to see what it represents
        decoded = tokenizer.decode([id])
        print(f"  {i+1:3}. Token: {token:20} ID: {id:6}  Decoded: '{decoded}'")

    print("-" * 50)
    print(f"Full decode: '{tokenizer.decode(ids)}'")


def compare_tokenizers(text: str):
    """
    Compare how different tokenizers handle the same text.
    """
    print("\n" + "=" * 60)
    print("TOKENIZER COMPARISON")
    print("=" * 60)
    print(f"\nInput: '{text}'")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install transformers: pip install transformers")
        return

    tokenizers = [
        "bert-base-uncased",
        "bert-base-cased",
        "gpt2",
        "roberta-base",
    ]

    print("\n" + "-" * 70)
    print(f"{'Tokenizer':<25} {'Vocab Size':>12} {'Num Tokens':>12} {'Tokens'}")
    print("-" * 70)

    for name in tokenizers:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            tokens = tok.tokenize(text)
            print(f"{name:<25} {tok.vocab_size:>12,} {len(tokens):>12} {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
        except Exception as e:
            print(f"{name:<25} Error: {e}")


def demo_padding_truncation():
    """
    Demonstrate padding and truncation - critical for batching.
    """
    print("\n" + "=" * 60)
    print("PADDING AND TRUNCATION")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except ImportError:
        print("Install transformers: pip install transformers")
        return

    texts = [
        "Short text.",
        "This is a medium length sentence for testing.",
        "This is a much longer sentence that contains many more words and should demonstrate how truncation works in practice.",
    ]

    print("\n1. Without padding (variable lengths):")
    for text in texts:
        tokens = tokenizer(text)
        print(f"  Length {len(tokens['input_ids']):3}: {text[:50]}...")

    print("\n2. With padding (max_length=20):")
    batch = tokenizer(texts, padding='max_length', max_length=20, truncation=True)
    for i, text in enumerate(texts):
        print(f"  {batch['input_ids'][i][:20]}...")
        print(f"  Attention mask: {batch['attention_mask'][i][:20]}")

    print("\n3. With padding (to longest in batch):")
    batch = tokenizer(texts, padding='longest', truncation=True, max_length=50)
    for i, text in enumerate(texts):
        print(f"  Length {len(batch['input_ids'][i])}: {batch['input_ids'][i][:15]}...")

    print("\n4. Truncation strategies:")
    long_text = "word " * 100

    # Left truncation (for decoder models)
    tokenizer.truncation_side = 'left'
    left_trunc = tokenizer(long_text, truncation=True, max_length=10)

    # Right truncation (default, for encoder models)
    tokenizer.truncation_side = 'right'
    right_trunc = tokenizer(long_text, truncation=True, max_length=10)

    print(f"  Left truncation:  {tokenizer.decode(left_trunc['input_ids'])}")
    print(f"  Right truncation: {tokenizer.decode(right_trunc['input_ids'])}")


# =============================================================================
# MAIN: RUN ALL DEMOS
# =============================================================================

def main():
    """Run all tokenization demonstrations."""

    print("=" * 60)
    print("MODULE 5B: TOKENIZATION - HANDS-ON DEMONSTRATIONS")
    print("=" * 60)

    # 1. Basic tokenizers (from scratch)
    print("\n\n" + "=" * 60)
    print("1. BASIC TOKENIZERS (FROM SCRATCH)")
    print("=" * 60)

    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Natural language processing with transformers",
    ]
    test = "Machine learning is quick"

    # Character tokenizer
    print("\nCharacter Tokenizer:")
    char_tok = CharacterTokenizer()
    char_tok.fit(corpus)
    encoded = char_tok.encode(test)
    decoded = char_tok.decode(encoded)
    print(f"  Vocab size: {char_tok.vocab_size}")
    print(f"  Input: '{test}'")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Num tokens: {len(encoded)}")

    # Word tokenizer
    print("\nWord Tokenizer:")
    word_tok = WordTokenizer()
    word_tok.fit(corpus)
    encoded = word_tok.encode(test)
    decoded = word_tok.decode(encoded)
    print(f"  Vocab size: {word_tok.vocab_size}")
    print(f"  Input: '{test}'")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Num tokens: {len(encoded)}")

    # 2. BPE from scratch
    print("\n\n" + "=" * 60)
    print("2. BPE TOKENIZER (FROM SCRATCH)")
    print("=" * 60)

    bpe_corpus = [
        "low lower lowest",
        "newer newest new",
        "wider widest wide",
    ] * 20  # Repeat for more data

    bpe_tok = BPETokenizer(vocab_size=100)
    bpe_tok.fit(bpe_corpus, verbose=True)

    test_words = ["lowest", "newer", "unknown"]
    print("\nBPE Tokenization:")
    for word in test_words:
        tokens = bpe_tok.tokenize(word)
        ids = bpe_tok.encode(word)
        decoded = bpe_tok.decode(ids)
        print(f"  '{word}' -> {tokens} -> {ids} -> '{decoded}'")

    # 3. Byte-Level BPE
    print("\n\n" + "=" * 60)
    print("3. BYTE-LEVEL BPE (GPT-2 STYLE)")
    print("=" * 60)

    byte_bpe = ByteLevelBPE(vocab_size=300)
    byte_bpe.fit(corpus, verbose=True)

    test_texts = ["Hello world!", "Machine learning"]
    print("\nByte-Level BPE Tokenization:")
    for text in test_texts:
        ids = byte_bpe.encode(text)
        decoded = byte_bpe.decode(ids)
        print(f"  '{text}' -> {ids} -> '{decoded}'")

    # 4. Hugging Face tokenizers
    demo_huggingface_tokenizers()

    # 5. tiktoken
    demo_tiktoken()

    # 6. Custom tokenizer training
    demo_custom_tokenizer_training()

    # 7. Practical utilities
    print("\n\n" + "=" * 60)
    print("7. PRACTICAL UTILITIES")
    print("=" * 60)

    analyze_tokenization("Hello, I'm learning about tokenization!", "gpt2")

    compare_tokenizers("The quick brown fox jumps over the lazy dog.")

    demo_padding_truncation()

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:
1. Character tokenization: Simple but very long sequences
2. Word tokenization: Short sequences but OOV problems
3. Subword (BPE/WordPiece): Best of both worlds
4. Byte-level BPE: No OOV ever (used by GPT-2+)

For production use:
- Use pre-trained tokenizers from Hugging Face
- Use tiktoken for OpenAI models
- Train custom tokenizers only for specialized domains

Critical rules:
- ALWAYS use the same tokenizer for training and inference
- Be aware of special tokens ([CLS], [SEP], etc.)
- Handle padding correctly (left for decoder, right for encoder)
- Check token counts before API calls
""")


if __name__ == "__main__":
    main()
