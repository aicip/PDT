"""
AG News dataset loader using HuggingFace datasets library.

This implementation uses the industry-standard HuggingFace datasets library
for better reproducibility and maintainability compared to deprecated torchtext.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter


class AGNewsDataset(Dataset):
    """
    AG News dataset wrapper for PyTorch DataLoader.

    Uses simple word-level tokenization and vocabulary built from training data.
    """

    def __init__(self, hf_dataset, vocab, max_len=200):
        """
        Args:
            hf_dataset: HuggingFace dataset split
            vocab: Dictionary mapping words to indices
            max_len: Maximum sequence length (pad/truncate)
        """
        self.data = hf_dataset
        self.vocab = vocab
        self.max_len = max_len
        self.unk_idx = vocab.get('<unk>', 1)
        self.pad_idx = vocab.get('<pad>', 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']  # Already 0-3 in HuggingFace version

        # Simple tokenization: lowercase + split
        tokens = text.lower().split()

        # Convert to indices
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]

        # Truncate or pad
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.pad_idx] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), label


def build_vocab_from_dataset(dataset, vocab_size=50000, min_freq=2):
    """
    Build vocabulary from dataset.

    Args:
        dataset: HuggingFace dataset
        vocab_size: Maximum vocabulary size
        min_freq: Minimum frequency for a word to be included

    Returns:
        vocab: Dictionary mapping words to indices
        actual_size: Actual vocabulary size
    """
    print(f"Building vocabulary (max_size={vocab_size}, min_freq={min_freq})...")

    # Count word frequencies
    word_counts = Counter()
    for item in dataset:
        tokens = item['text'].lower().split()
        word_counts.update(tokens)

    # Select top words
    most_common = word_counts.most_common(vocab_size - 2)  # Reserve 2 for special tokens

    # Filter by minimum frequency
    filtered_words = [word for word, count in most_common if count >= min_freq]

    # Build vocabulary
    vocab = {'<pad>': 0, '<unk>': 1}
    for idx, word in enumerate(filtered_words, start=2):
        vocab[word] = idx

    print(f"Vocabulary built. Size: {len(vocab)} (from {len(word_counts)} unique words)")

    return vocab, len(vocab)


def create_ag_news(
    dataset_dir='data',  # Not used for HF datasets, kept for API compatibility
    train_batch_size=64,
    test_batch_size=128,
    max_len=50,  # Reduced from 200 - long sequences cause LSTM gradient vanishing
    vocab_size=5000,  # Reduced from 50000 - large vocab causes LSTM training issues
    dist=False,
    ws=1,
    rank=0,
    num_workers=4
):
    """
    Create AG News DataLoaders using HuggingFace datasets.

    Args:
        dataset_dir: Not used (kept for compatibility), HF datasets auto-caches
        train_batch_size: Training batch size
        test_batch_size: Test batch size
        max_len: Maximum sequence length
        vocab_size: Maximum vocabulary size
        dist: Whether using distributed training
        ws: World size (number of GPUs)
        rank: Current process rank
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        actual_vocab_size: Actual vocabulary size (for model initialization)
    """
    print("Loading AG News dataset from HuggingFace...")

    # Load dataset from HuggingFace
    # This will auto-cache to ~/.cache/huggingface/datasets
    dataset = load_dataset('ag_news')

    train_data = dataset['train']
    test_data = dataset['test']

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Build vocabulary from training set
    vocab, actual_vocab_size = build_vocab_from_dataset(
        train_data,
        vocab_size=vocab_size,
        min_freq=2
    )

    # Create PyTorch Dataset objects
    train_dataset = AGNewsDataset(train_data, vocab, max_len)
    test_dataset = AGNewsDataset(test_data, vocab, max_len)

    # Create samplers for distributed training
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=ws, rank=rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=ws, rank=rank
        )
    else:
        train_sampler, test_sampler = None, None

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers // 2,
        pin_memory=True
    )

    print(f"DataLoaders created. Vocabulary size: {actual_vocab_size}")

    return train_loader, test_loader, actual_vocab_size


# For testing
if __name__ == "__main__":
    print("Testing AG News dataset loader...")
    train_loader, test_loader, vocab_size = create_ag_news(
        train_batch_size=4,
        test_batch_size=4,
        max_len=50,
        num_workers=0
    )

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    for batch_idx, (texts, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Text shape: {texts.shape}")
        print(f"  Labels: {labels}")
        print(f"  Label range: {labels.min()}-{labels.max()}")
        break

    print("\nDataset loader test passed!")
