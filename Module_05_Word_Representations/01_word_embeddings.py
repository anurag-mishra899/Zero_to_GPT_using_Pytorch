"""
Module 5.1: Word Embeddings - From One-Hot to Word2Vec
Complete implementation of word embedding techniques.

This module covers:
1. One-hot encoding
2. Word2Vec (Skip-gram and CBOW)
3. Negative Sampling
4. Embedding operations and analogies
5. Practical embedding usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import math
import random
from typing import List, Tuple, Dict, Optional

print("=" * 70)
print("MODULE 5.1: WORD EMBEDDINGS")
print("=" * 70)

# =============================================================================
# SECTION 1: ONE-HOT ENCODING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: ONE-HOT ENCODING")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 One-Hot Encoding Implementation
# -----------------------------------------------------------------------------
print("\n1.1 One-Hot Encoding")
print("-" * 40)


class OneHotEncoder:
    """
    One-hot encoding for words.

    Each word becomes a vector of vocabulary size with single 1.
    """

    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        self.word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocabulary)

    def encode(self, word: str) -> torch.Tensor:
        """Encode single word as one-hot vector."""
        if word not in self.word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary")

        idx = self.word_to_idx[word]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[idx] = 1.0
        return one_hot

    def encode_batch(self, words: List[str]) -> torch.Tensor:
        """Encode multiple words."""
        return torch.stack([self.encode(word) for word in words])

    def decode(self, one_hot: torch.Tensor) -> str:
        """Decode one-hot vector to word."""
        idx = one_hot.argmax().item()
        return self.idx_to_word[idx]


# Demonstrate one-hot encoding
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
encoder = OneHotEncoder(vocab)

print(f"Vocabulary: {vocab}")
print(f"Vocabulary size: {encoder.vocab_size}")

for word in ["cat", "dog", "mat"]:
    one_hot = encoder.encode(word)
    print(f"  '{word}' → {one_hot.tolist()}")

# Show problem: orthogonality
cat_vec = encoder.encode("cat")
dog_vec = encoder.encode("dog")
mat_vec = encoder.encode("mat")

sim_cat_dog = F.cosine_similarity(cat_vec.unsqueeze(0), dog_vec.unsqueeze(0))
sim_cat_mat = F.cosine_similarity(cat_vec.unsqueeze(0), mat_vec.unsqueeze(0))

print(f"\nProblem - All vectors orthogonal:")
print(f"  cos(cat, dog) = {sim_cat_dog.item():.4f}")
print(f"  cos(cat, mat) = {sim_cat_mat.item():.4f}")
print("  Cat is no more similar to dog than to mat!")


# =============================================================================
# SECTION 2: EMBEDDING LAYER
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: EMBEDDING LAYER")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 Embedding Layer Implementation
# -----------------------------------------------------------------------------
print("\n2.1 Embedding Layer (Lookup Table)")
print("-" * 40)


class EmbeddingLayer(nn.Module):
    """
    Embedding layer = lookup table for dense word vectors.

    Equivalent to: one_hot @ weight_matrix
    But more efficient: direct indexing
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Weight matrix: (vocab_size, embed_dim)
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))

        # Initialize
        nn.init.uniform_(self.weight, -0.5 / embed_dim, 0.5 / embed_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for given indices.

        Args:
            indices: (batch,) or (batch, seq_len) of word indices

        Returns:
            embeddings: (..., embed_dim)
        """
        return F.embedding(indices, self.weight)


# Demonstrate embedding lookup
vocab_size = 1000
embed_dim = 64

embed = EmbeddingLayer(vocab_size, embed_dim)
word_indices = torch.tensor([42, 100, 500])

embeddings = embed(word_indices)
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embed_dim}")
print(f"Input indices: {word_indices.tolist()}")
print(f"Output shape: {embeddings.shape}")

# Show that embeddings capture more than one-hot
# (Similar indices don't mean similar embeddings - they're learned!)


# -----------------------------------------------------------------------------
# 2.2 Embedding vs One-Hot Matrix Multiply
# -----------------------------------------------------------------------------
print("\n\n2.2 Embedding = Efficient One-Hot Lookup")
print("-" * 40)


def one_hot_to_embedding(one_hot: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot through matrix multiply.

    This is what embedding layer does, but less efficiently.
    """
    return one_hot @ weight


# Show equivalence
vocab_size = 10
embed_dim = 4

# Create embedding layer
embed = nn.Embedding(vocab_size, embed_dim)

# Get embedding via indexing
idx = torch.tensor([3])
embed_indexed = embed(idx)

# Get embedding via one-hot multiply
one_hot = torch.zeros(vocab_size)
one_hot[3] = 1.0
embed_matmul = one_hot @ embed.weight

print("Embedding via indexing vs one-hot multiply:")
print(f"  Indexed: {embed_indexed.squeeze().tolist()}")
print(f"  Matmul:  {embed_matmul.tolist()}")
print("  These are identical!")


# =============================================================================
# SECTION 3: WORD2VEC SKIP-GRAM
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: WORD2VEC SKIP-GRAM")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Skip-gram Model
# -----------------------------------------------------------------------------
print("\n3.1 Skip-gram Model")
print("-" * 40)


class SkipGramModel(nn.Module):
    """
    Word2Vec Skip-gram model.

    Architecture:
    - Input: Center word (one-hot or index)
    - Hidden: Embedding layer (this becomes our word vectors)
    - Output: Probability distribution over vocabulary

    Objective: Predict context words from center word.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()

        # Input word embeddings (these are the final embeddings)
        self.center_embed = nn.Embedding(vocab_size, embed_dim)

        # Output word embeddings (context embeddings)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self.center_embed.embedding_dim
        self.center_embed.weight.data.uniform_(-initrange, initrange)
        self.context_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, center_words: torch.Tensor, context_words: torch.Tensor) -> torch.Tensor:
        """
        Compute score for (center, context) pairs.

        Args:
            center_words: (batch,) center word indices
            context_words: (batch,) context word indices

        Returns:
            scores: (batch,) dot products (higher = more likely pair)
        """
        center_embeds = self.center_embed(center_words)    # (batch, embed_dim)
        context_embeds = self.context_embed(context_words)  # (batch, embed_dim)

        # Dot product scores
        scores = (center_embeds * context_embeds).sum(dim=-1)
        return scores

    def get_embeddings(self) -> torch.Tensor:
        """Return learned word embeddings."""
        return self.center_embed.weight.data


# Create model
model = SkipGramModel(vocab_size=1000, embed_dim=100)
print(f"Skip-gram model:")
print(f"  Vocabulary: 1000")
print(f"  Embedding dim: 100")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


# -----------------------------------------------------------------------------
# 3.2 Creating Training Data
# -----------------------------------------------------------------------------
print("\n\n3.2 Creating Skip-gram Training Data")
print("-" * 40)


class SkipGramDataset:
    """
    Create (center, context) pairs for Skip-gram training.
    """

    def __init__(self, corpus: List[str], window_size: int = 2, min_count: int = 1):
        self.window_size = window_size

        # Build vocabulary
        self.word_counts = Counter(corpus)
        self.vocab = [word for word, count in self.word_counts.items() if count >= min_count]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)

        # Convert corpus to indices
        self.corpus_indices = [self.word_to_idx[w] for w in corpus if w in self.word_to_idx]

    def generate_pairs(self) -> List[Tuple[int, int]]:
        """Generate all (center, context) pairs."""
        pairs = []

        for i, center_idx in enumerate(self.corpus_indices):
            # Define context window
            start = max(0, i - self.window_size)
            end = min(len(self.corpus_indices), i + self.window_size + 1)

            for j in range(start, end):
                if i != j:  # Skip center word itself
                    context_idx = self.corpus_indices[j]
                    pairs.append((center_idx, context_idx))

        return pairs


# Demonstrate
corpus = "the quick brown fox jumps over the lazy dog the cat sat on the mat".split()
dataset = SkipGramDataset(corpus, window_size=2)

print(f"Corpus: {' '.join(corpus[:10])}...")
print(f"Vocabulary size: {dataset.vocab_size}")

pairs = dataset.generate_pairs()
print(f"Training pairs: {len(pairs)}")
print("\nExample pairs:")
for i in range(min(5, len(pairs))):
    center, context = pairs[i]
    print(f"  ({dataset.idx_to_word[center]}, {dataset.idx_to_word[context]})")


# =============================================================================
# SECTION 4: NEGATIVE SAMPLING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: NEGATIVE SAMPLING")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 Negative Sampling Implementation
# -----------------------------------------------------------------------------
print("\n4.1 Negative Sampling")
print("-" * 40)


class NegativeSampler:
    """
    Sample negative words for Word2Vec training.

    Uses unigram distribution raised to 0.75 power.
    """

    def __init__(self, word_counts: Dict[str, int], word_to_idx: Dict[str, int], power: float = 0.75):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        self.vocab_size = len(word_to_idx)

        # Build sampling distribution: count^0.75
        counts = []
        for word in word_to_idx:
            counts.append(word_counts.get(word, 1) ** power)

        total = sum(counts)
        self.sampling_probs = [c / total for c in counts]

        # Pre-compute cumulative distribution for efficient sampling
        self.cumulative = []
        cumsum = 0
        for p in self.sampling_probs:
            cumsum += p
            self.cumulative.append(cumsum)

    def sample(self, n_samples: int, exclude: Optional[List[int]] = None) -> List[int]:
        """
        Sample negative word indices.

        Args:
            n_samples: Number of negative samples
            exclude: Indices to exclude (positive examples)
        """
        exclude = set(exclude) if exclude else set()
        samples = []

        while len(samples) < n_samples:
            # Sample from distribution
            r = random.random()
            idx = 0
            for i, cum in enumerate(self.cumulative):
                if r <= cum:
                    idx = i
                    break

            if idx not in exclude:
                samples.append(idx)

        return samples


# Demonstrate
word_counts = Counter(corpus)
sampler = NegativeSampler(word_counts, dataset.word_to_idx)

center_word = dataset.word_to_idx["the"]
negative_samples = sampler.sample(5, exclude=[center_word])

print(f"Center word: 'the' (idx={center_word})")
print(f"Negative samples: {[dataset.idx_to_word[idx] for idx in negative_samples]}")


# -----------------------------------------------------------------------------
# 4.2 Skip-gram with Negative Sampling Loss
# -----------------------------------------------------------------------------
print("\n\n4.2 Skip-gram Negative Sampling Loss")
print("-" * 40)


class SkipGramNegativeSampling(nn.Module):
    """
    Skip-gram with negative sampling loss.

    Loss = -log σ(v_c · v_o) - Σ_k log σ(-v_c · v_k)

    where:
        v_c = center word embedding
        v_o = context (positive) word embedding
        v_k = negative sample embeddings
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

        # Initialize
        nn.init.uniform_(self.center_embed.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.uniform_(self.context_embed.weight, -0.5/embed_dim, 0.5/embed_dim)

    def forward(
        self,
        center: torch.Tensor,     # (batch,)
        positive: torch.Tensor,   # (batch,)
        negatives: torch.Tensor   # (batch, n_neg)
    ) -> torch.Tensor:
        """
        Compute negative sampling loss.
        """
        batch_size = center.size(0)
        n_neg = negatives.size(1)

        # Embeddings
        center_emb = self.center_embed(center)      # (batch, dim)
        pos_emb = self.context_embed(positive)      # (batch, dim)
        neg_emb = self.context_embed(negatives)     # (batch, n_neg, dim)

        # Positive score: v_c · v_o
        pos_score = (center_emb * pos_emb).sum(dim=-1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score)

        # Negative scores: v_c · v_k
        # center_emb: (batch, dim) -> (batch, 1, dim)
        neg_scores = torch.bmm(neg_emb, center_emb.unsqueeze(-1)).squeeze(-1)  # (batch, n_neg)
        neg_loss = -F.logsigmoid(-neg_scores).sum(dim=-1)  # (batch,)

        return (pos_loss + neg_loss).mean()

    def get_embeddings(self) -> torch.Tensor:
        return self.center_embed.weight.data


# Test
model = SkipGramNegativeSampling(vocab_size=1000, embed_dim=100)
center = torch.randint(0, 1000, (32,))
positive = torch.randint(0, 1000, (32,))
negatives = torch.randint(0, 1000, (32, 5))

loss = model(center, positive, negatives)
print(f"Batch size: 32, Negative samples: 5")
print(f"Loss: {loss.item():.4f}")


# =============================================================================
# SECTION 5: COMPLETE WORD2VEC TRAINING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: COMPLETE WORD2VEC TRAINING")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Word2Vec Trainer
# -----------------------------------------------------------------------------
print("\n5.1 Complete Word2Vec Training Loop")
print("-" * 40)


class Word2VecTrainer:
    """Complete Word2Vec training with negative sampling."""

    def __init__(
        self,
        corpus: List[str],
        embed_dim: int = 100,
        window_size: int = 5,
        n_negatives: int = 5,
        min_count: int = 5,
        learning_rate: float = 0.025,
        subsample_threshold: float = 1e-5
    ):
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_negatives = n_negatives

        # Build vocabulary
        word_counts = Counter(corpus)
        self.vocab = [w for w, c in word_counts.items() if c >= min_count]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)

        # Convert corpus
        self.corpus = [self.word_to_idx[w] for w in corpus if w in self.word_to_idx]

        # Build subsampling probabilities
        total_count = sum(word_counts[w] for w in self.vocab)
        self.subsample_probs = {}
        for word in self.vocab:
            freq = word_counts[word] / total_count
            prob_keep = min(1.0, math.sqrt(subsample_threshold / freq))
            self.subsample_probs[self.word_to_idx[word]] = prob_keep

        # Build negative sampling distribution
        counts_powered = [word_counts[w] ** 0.75 for w in self.vocab]
        total_powered = sum(counts_powered)
        self.neg_probs = torch.tensor([c / total_powered for c in counts_powered])

        # Model
        self.model = SkipGramNegativeSampling(self.vocab_size, embed_dim)
        self.optimizer = optim.SparseAdam(self.model.parameters(), lr=learning_rate)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Corpus length: {len(self.corpus)}")

    def _subsample(self, indices: List[int]) -> List[int]:
        """Apply subsampling to frequent words."""
        return [idx for idx in indices if random.random() < self.subsample_probs[idx]]

    def _sample_negatives(self, batch_size: int) -> torch.Tensor:
        """Sample negative indices."""
        return torch.multinomial(self.neg_probs, batch_size * self.n_negatives, replacement=True).view(batch_size, self.n_negatives)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        # Subsample corpus
        subsampled = self._subsample(self.corpus)

        # Generate training pairs
        pairs = []
        for i, center_idx in enumerate(subsampled):
            start = max(0, i - self.window_size)
            end = min(len(subsampled), i + self.window_size + 1)

            for j in range(start, end):
                if i != j:
                    pairs.append((center_idx, subsampled[j]))

        random.shuffle(pairs)

        # Training
        total_loss = 0
        batch_size = 512

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            centers = torch.tensor([p[0] for p in batch_pairs])
            positives = torch.tensor([p[1] for p in batch_pairs])
            negatives = self._sample_negatives(len(batch_pairs))

            self.optimizer.zero_grad()
            loss = self.model(centers, positives, negatives)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / (len(pairs) // batch_size + 1)

    def get_embedding(self, word: str) -> torch.Tensor:
        """Get embedding for a word."""
        if word not in self.word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        idx = self.word_to_idx[word]
        return self.model.get_embeddings()[idx]

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words."""
        word_embed = self.get_embedding(word)
        all_embeds = self.model.get_embeddings()

        # Cosine similarity
        similarities = F.cosine_similarity(word_embed.unsqueeze(0), all_embeds)

        # Get top-k (excluding the word itself)
        word_idx = self.word_to_idx[word]
        similarities[word_idx] = -float('inf')

        top_indices = similarities.topk(top_k).indices.tolist()
        top_sims = similarities[top_indices].tolist()

        return [(self.idx_to_word[idx], sim) for idx, sim in zip(top_indices, top_sims)]


# Train on small corpus (in practice, use much larger data)
print("\nTraining Word2Vec on small corpus...")
large_corpus = corpus * 1000  # Repeat for more data
trainer = Word2VecTrainer(large_corpus, embed_dim=50, window_size=2, n_negatives=5, min_count=1)

for epoch in range(3):
    loss = trainer.train_epoch()
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")


# =============================================================================
# SECTION 6: EMBEDDING OPERATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: EMBEDDING OPERATIONS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 Similarity and Analogies
# -----------------------------------------------------------------------------
print("\n6.1 Embedding Operations")
print("-" * 40)


class EmbeddingOperations:
    """Operations on word embeddings."""

    def __init__(self, embeddings: torch.Tensor, idx_to_word: Dict[int, str], word_to_idx: Dict[str, int]):
        self.embeddings = embeddings
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

        # Normalize embeddings for cosine similarity
        self.normalized = F.normalize(embeddings, dim=-1)

    def similarity(self, word1: str, word2: str) -> float:
        """Cosine similarity between two words."""
        idx1 = self.word_to_idx[word1]
        idx2 = self.word_to_idx[word2]
        return (self.normalized[idx1] @ self.normalized[idx2]).item()

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words."""
        idx = self.word_to_idx[word]
        word_vec = self.normalized[idx]

        similarities = self.normalized @ word_vec
        similarities[idx] = -float('inf')  # Exclude itself

        top_indices = similarities.topk(top_k).indices.tolist()
        return [(self.idx_to_word[i], similarities[i].item()) for i in top_indices]

    def analogy(self, a: str, b: str, c: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Solve analogy: A is to B as C is to ?

        Formula: result = B - A + C
        """
        idx_a = self.word_to_idx[a]
        idx_b = self.word_to_idx[b]
        idx_c = self.word_to_idx[c]

        # Compute target vector
        target = self.normalized[idx_b] - self.normalized[idx_a] + self.normalized[idx_c]
        target = F.normalize(target, dim=0)

        # Find closest
        similarities = self.normalized @ target
        for idx in [idx_a, idx_b, idx_c]:
            similarities[idx] = -float('inf')  # Exclude inputs

        top_indices = similarities.topk(top_k).indices.tolist()
        return [(self.idx_to_word[i], similarities[i].item()) for i in top_indices]


# Demonstrate with pre-trained-like embeddings (random for demo)
print("Embedding operations demo (random embeddings for illustration):")

# Create random embeddings
demo_vocab = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
demo_embeddings = torch.randn(len(demo_vocab), 50)
demo_word_to_idx = {w: i for i, w in enumerate(demo_vocab)}
demo_idx_to_word = {i: w for w, i in demo_word_to_idx.items()}

ops = EmbeddingOperations(demo_embeddings, demo_idx_to_word, demo_word_to_idx)

print(f"\nSimilarity examples:")
print(f"  sim(king, queen) = {ops.similarity('king', 'queen'):.4f}")
print(f"  sim(king, man) = {ops.similarity('king', 'man'):.4f}")

print(f"\nMost similar to 'king':")
for word, sim in ops.most_similar('king', top_k=3):
    print(f"  {word}: {sim:.4f}")


# =============================================================================
# SECTION 7: PRACTICAL EMBEDDING USAGE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: PRACTICAL EMBEDDING USAGE")
print("=" * 70)

# -----------------------------------------------------------------------------
# 7.1 Embedding Layer in Neural Networks
# -----------------------------------------------------------------------------
print("\n7.1 Using Embeddings in Neural Networks")
print("-" * 40)


class TextClassifier(nn.Module):
    """
    Simple text classifier using embeddings.

    Architecture:
    - Embedding layer (can be pre-trained or learned)
    - Average pooling
    - Linear classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        """
        # Get embeddings
        embeds = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Average pooling (masked if provided)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = embeds.mean(dim=1)

        # Classify
        return self.classifier(pooled)


# Demonstrate
classifier = TextClassifier(vocab_size=10000, embed_dim=100, num_classes=3)
batch = torch.randint(0, 10000, (8, 20))  # 8 sequences of length 20
logits = classifier(batch)

print(f"Text Classifier:")
print(f"  Input: {batch.shape}")
print(f"  Output: {logits.shape}")
print(f"  Parameters: {sum(p.numel() for p in classifier.parameters()):,}")


# -----------------------------------------------------------------------------
# 7.2 Loading Pre-trained Embeddings
# -----------------------------------------------------------------------------
print("\n\n7.2 Loading Pre-trained Embeddings (GloVe/Word2Vec)")
print("-" * 40)


def load_pretrained_embeddings(
    embeddings_file: str,
    word_to_idx: Dict[str, int],
    embed_dim: int
) -> torch.Tensor:
    """
    Load pre-trained embeddings (GloVe, Word2Vec format).

    Format: word dim1 dim2 dim3 ...
    """
    vocab_size = len(word_to_idx)
    embeddings = torch.zeros(vocab_size, embed_dim)

    # Initialize with random for OOV words
    nn.init.normal_(embeddings, mean=0, std=0.1)

    found = 0
    # In practice, read from file:
    # with open(embeddings_file, 'r') as f:
    #     for line in f:
    #         parts = line.strip().split()
    #         word = parts[0]
    #         if word in word_to_idx:
    #             vector = torch.tensor([float(x) for x in parts[1:]])
    #             embeddings[word_to_idx[word]] = vector
    #             found += 1

    print(f"Pretrained embedding loading pattern:")
    print(f"  1. Read embeddings file (e.g., glove.6B.100d.txt)")
    print(f"  2. Match words with your vocabulary")
    print(f"  3. Copy vectors for matched words")
    print(f"  4. Random init for unmatched (OOV) words")

    return embeddings


# =============================================================================
# SECTION 8: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. Word2Vec Skip-gram:
   - Predict context from center word
   - Training pairs: (center, context)
   - Embedding layer weights = word vectors

2. Negative Sampling:
   - Binary classification instead of softmax
   - Positive: real (center, context) pairs
   - Negative: (center, random_word) pairs
   - Distribution: unigram^0.75

3. Embedding Properties:
   - king - man + woman ≈ queen
   - Cosine similarity for semantic closeness
   - Relationships as directions

4. One-hot vs Dense:
   - One-hot: sparse, no semantics
   - Dense: learned, semantic similarity

5. Limitations:
   - One embedding per word
   - No context sensitivity
   - Out-of-vocabulary problem

6. Modern Extensions:
   - FastText: subword embeddings
   - BERT/GPT: contextual embeddings
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 5.1 SUMMARY: WORD EMBEDDINGS")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬────────────────────────────────┬─────────────────────────┐
│ Method          │ Description                    │ Properties              │
├─────────────────┼────────────────────────────────┼─────────────────────────┤
│ One-hot         │ Sparse, vocab-sized            │ No semantics            │
│ Word2Vec        │ Dense, learned from context    │ Semantic similarity     │
│ Skip-gram       │ Predict context from center    │ Better for rare words   │
│ CBOW            │ Predict center from context    │ Faster training         │
│ Neg. Sampling   │ Binary classifier              │ Efficient training      │
└─────────────────┴────────────────────────────────┴─────────────────────────┘

KEY EQUATIONS:

Skip-gram Objective:
  max Σ log P(context | center)

Negative Sampling Loss:
  L = -log σ(v_c · v_o) - Σ_k log σ(-v_c · v_k)

Analogy:
  A : B :: C : ?
  answer = argmax_d cos(d, B - A + C)

HYPERPARAMETERS:
  Embedding dim: 100-300
  Window size: 5-10
  Negative samples: 5-20
  Min count: 5
  Subsampling: 1e-5

KEY TAKEAWAYS:
1. Dense embeddings capture semantics
2. Context predicts meaning (distributional hypothesis)
3. Negative sampling makes training efficient
4. Relationships encoded as directions
5. Contextual embeddings (BERT/GPT) solve polysemy
""")

print("\nModule 5.1 Complete!")
