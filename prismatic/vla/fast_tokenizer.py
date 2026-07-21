"""
fast_tokenizer.py

FAST (Frequency-space Action Sequence Tokenization) for VLA-Adapter.
Implements DCT-based action tokenization following the pi0-FAST paper (arXiv:2501.09747).

Pipeline: normalize → DCT → quantize → column-first flatten → BPE encode
Inverse:  BPE decode → unflatten → dequantize → IDCT → denormalize

This is a self-contained implementation that does NOT require scipy.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ============================================================================
# DCT / IDCT via FFT (no scipy dependency)
# ============================================================================

def dct_1d(x: np.ndarray) -> np.ndarray:
    """Type-II Discrete Cosine Transform along the last axis, using FFT.

    Args:
        x: (..., N) input array.

    Returns:
        (..., N) DCT coefficients.
    """
    N = x.shape[-1]
    # Reorder: even indices first, then odd in reverse
    idx = np.empty(N, dtype=int)
    half = (N + 1) // 2
    idx[:half] = np.arange(0, N, 2)
    idx[half:] = np.arange(2 * (N // 2) - 1, 0, -2)
    y = x[..., idx]
    Y = np.fft.fft(y, axis=-1)
    k = np.arange(N)
    phase = np.exp(-1j * np.pi * k / (2 * N)) * 2
    return (Y * phase).real


def idct_1d(X: np.ndarray) -> np.ndarray:
    """Type-II Inverse DCT along the last axis, using FFT.

    Args:
        X: (..., N) DCT coefficients.

    Returns:
        (..., N) reconstructed signal.
    """
    N = X.shape[-1]
    # Undo the phase multiplication
    k = np.arange(N)
    phase = np.exp(1j * np.pi * k / (2 * N)) / 2
    Y = X * phase
    # Symmetry: construct full FFT from half-spectrum
    Y[..., 0] /= 2
    y = np.fft.ifft(Y, axis=-1).real * N
    # Undo the reorder
    out = np.empty_like(y)
    half = (N + 1) // 2
    out[..., 0::2] = y[..., :half]
    out[..., 1::2] = y[..., half:][::-1] if N % 2 == 0 else y[..., half:][::-1]
    # Proper inverse via even/odd interleaving
    # Actually, let's use the standard IDCT-II formula directly for correctness
    return _idct_direct(X)


def _idct_direct(X: np.ndarray) -> np.ndarray:
    """Direct IDCT-II computation (slower but correct reference).

    Inverse of DCT-II: x[n] = X[0]/(2N) + (1/N) * sum_{k=1}^{N-1} X[k] * cos(pi*k*(2n+1)/(2N))
    """
    N = X.shape[-1]
    n = np.arange(N)
    k = np.arange(N)
    # basis[n_idx, k_idx] = cos(pi * k * (2n+1) / (2N))
    basis = np.cos(np.pi * k[None, :] * (2 * n[:, None] + 1) / (2 * N))
    # Weights: k=0 -> 1/(2N), k>0 -> 1/N
    weights = np.full(N, 1.0 / N)
    weights[0] = 1.0 / (2 * N)
    # Apply: x[n] = sum_k X[k] * w[k] * basis[n, k]
    if X.ndim == 1:
        return (X * weights) @ basis.T
    else:
        return np.einsum("...k,nk->...n", X * weights, basis)


# ============================================================================
# FAST Tokenizer
# ============================================================================

class FASTTokenizer:
    """FAST action tokenizer: DCT + quantize + BPE.

    This tokenizer converts continuous action chunks of shape (H, D) into
    a sequence of integer tokens from a fixed vocabulary, and can decode
    them back to continuous actions.

    Attributes:
        scale: Quantization scaling factor (default=10).
        vocab_size: BPE vocabulary size (default=1024).
        horizon: Action chunk length (number of timesteps).
        action_dim: Number of action dimensions.
        norm_stats: Normalization statistics {dim: (q01, q99)}.
        bpe_merges: List of BPE merge rules.
        bpe_vocab: Token vocabulary mapping.
    """

    def __init__(
        self,
        scale: float = 10.0,
        vocab_size: int = 1024,
        horizon: int = 8,
        action_dim: int = 7,
    ):
        self.scale = scale
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.action_dim = action_dim

        # Normalization stats: per-dim q01 and q99
        self.q01 = np.zeros(action_dim)
        self.q99 = np.ones(action_dim)

        # BPE state
        self.bpe_merges: List[Tuple[int, int]] = []
        self.bpe_vocab: Dict[int, List[int]] = {}  # token_id -> byte sequence
        self.bpe_vocab_inv: Dict[tuple, int] = {}  # byte sequence -> token_id
        self._base_vocab_size = 0  # number of unique integers before BPE

    # ========================================================================
    # Normalization
    # ========================================================================

    def _normalize(self, actions: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] using quantile statistics."""
        range_ = self.q99 - self.q01
        range_ = np.where(range_ < 1e-8, 1.0, range_)  # avoid division by zero
        return 2 * (actions - self.q01) / range_ - 1

    def _denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Inverse normalization from [-1, 1] back to original scale."""
        range_ = self.q99 - self.q01
        return (actions + 1) / 2 * range_ + self.q01

    # ========================================================================
    # DCT encode/decode
    # ========================================================================

    def _dct_encode(self, actions: np.ndarray) -> np.ndarray:
        """Apply DCT to each action dimension.

        Args:
            actions: (H, D) normalized actions.

        Returns:
            (H, D) DCT coefficient matrix.
        """
        # DCT along time axis (axis=0) for each dimension
        return dct_1d(actions.T).T  # transpose to apply along H, then back

    def _dct_decode(self, coeffs: np.ndarray) -> np.ndarray:
        """Apply inverse DCT to reconstruct actions.

        Args:
            coeffs: (H, D) DCT coefficient matrix.

        Returns:
            (H, D) reconstructed actions.
        """
        return _idct_direct(coeffs.T).T

    # ========================================================================
    # Quantization
    # ========================================================================

    def _quantize(self, coeffs: np.ndarray) -> np.ndarray:
        """Scale and round DCT coefficients to integers."""
        return np.round(self.scale * coeffs).astype(int)

    def _dequantize(self, qcoeffs: np.ndarray) -> np.ndarray:
        """Inverse quantization."""
        return qcoeffs.astype(float) / self.scale

    # ========================================================================
    # Column-first flattening
    # ========================================================================

    def _flatten(self, qcoeffs: np.ndarray) -> List[int]:
        """Column-first flattening: lowest frequency across all dims first.

        Args:
            qcoeffs: (H, D) integer DCT coefficients.

        Returns:
            List of integers of length H*D.
        """
        # Column-first = Fortran order
        return qcoeffs.flatten(order='F').tolist()

    def _unflatten(self, flat: List[int]) -> np.ndarray:
        """Reconstruct (H, D) matrix from column-first flattened sequence."""
        arr = np.array(flat, dtype=int)
        return arr.reshape((self.horizon, self.action_dim), order='F')

    # ========================================================================
    # BPE encoding/decoding
    # ========================================================================

    def _bpe_encode(self, integers: List[int]) -> List[int]:
        """Apply BPE merges to compress integer sequence into tokens."""
        # Shift integers to positive range for BPE (base vocabulary)
        # Map: original int -> base_token = int + offset
        tokens = [self._int_to_base(i) for i in integers]

        # Apply merges greedily
        for merge_a, merge_b in self.bpe_merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge_a and tokens[i + 1] == merge_b:
                    # Merge found: replace pair with merged token
                    merged_id = self.bpe_vocab_inv.get((merge_a, merge_b))
                    if merged_id is not None:
                        new_tokens.append(merged_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def _bpe_decode(self, tokens: List[int]) -> List[int]:
        """Decode BPE tokens back to integer sequence."""
        # Expand each token to its base integers
        expanded = []
        for t in tokens:
            if t in self.bpe_vocab:
                expanded.extend(self.bpe_vocab[t])
            else:
                expanded.append(t)
        # Convert base tokens back to original integers
        return [self._base_to_int(b) for b in expanded]

    def _int_to_base(self, val: int) -> int:
        """Map an arbitrary integer to a base vocabulary token."""
        # Shift by offset so all values are non-negative
        return val + self._int_offset

    def _base_to_int(self, base: int) -> int:
        """Map base vocabulary token back to integer."""
        return base - self._int_offset

    # ========================================================================
    # Full encode/decode pipeline
    # ========================================================================

    def encode(self, actions: np.ndarray) -> List[int]:
        """Encode a single action chunk to FAST tokens.

        Args:
            actions: (H, D) continuous action chunk (unnormalized).

        Returns:
            List of integer token IDs.
        """
        assert actions.shape == (self.horizon, self.action_dim), \
            f"Expected ({self.horizon}, {self.action_dim}), got {actions.shape}"

        normed = self._normalize(actions)
        coeffs = self._dct_encode(normed)
        qcoeffs = self._quantize(coeffs)
        flat = self._flatten(qcoeffs)
        tokens = self._bpe_encode(flat)
        return tokens

    def decode(self, tokens: List[int]) -> np.ndarray:
        """Decode FAST tokens back to continuous actions.

        Args:
            tokens: List of integer token IDs.

        Returns:
            (H, D) continuous action chunk (unnormalized).
        """
        flat = self._bpe_decode(tokens)
        # Handle length mismatch (BPE decode may produce slightly different length)
        expected_len = self.horizon * self.action_dim
        if len(flat) < expected_len:
            flat.extend([0] * (expected_len - len(flat)))
        elif len(flat) > expected_len:
            flat = flat[:expected_len]

        qcoeffs = self._unflatten(flat)
        coeffs = self._dequantize(qcoeffs)
        normed = self._dct_decode(coeffs)
        actions = self._denormalize(normed)
        return actions

    def encode_batch(self, actions_batch: np.ndarray) -> List[List[int]]:
        """Encode a batch of action chunks.

        Args:
            actions_batch: (B, H, D) batch of action chunks.

        Returns:
            List of B token sequences.
        """
        return [self.encode(actions_batch[i]) for i in range(len(actions_batch))]

    def decode_batch(self, tokens_batch: List[List[int]]) -> np.ndarray:
        """Decode a batch of token sequences.

        Args:
            tokens_batch: List of B token sequences.

        Returns:
            (B, H, D) reconstructed action chunks.
        """
        decoded = [self.decode(t) for t in tokens_batch]
        return np.stack(decoded)

    # ========================================================================
    # Training (fit BPE from data)
    # ========================================================================

    def fit(self, action_chunks: np.ndarray, num_bpe_merges: Optional[int] = None) -> "FASTTokenizer":
        """Train the tokenizer on action data.

        Args:
            action_chunks: (N, H, D) training action chunks.
            num_bpe_merges: Number of BPE merges (default: vocab_size - base_vocab_size).

        Returns:
            self (for chaining).
        """
        N = action_chunks.shape[0]
        assert action_chunks.shape[1:] == (self.horizon, self.action_dim), \
            f"Expected (N, {self.horizon}, {self.action_dim}), got {action_chunks.shape}"

        print(f"[FAST] Fitting tokenizer on {N} action chunks...")

        # Step 1: Compute normalization statistics
        flat_actions = action_chunks.reshape(-1, self.action_dim)
        self.q01 = np.percentile(flat_actions, 1, axis=0)
        self.q99 = np.percentile(flat_actions, 99, axis=0)
        print(f"[FAST] Normalization: q01={self.q01}, q99={self.q99}")

        # Step 2: Encode all chunks to quantized integers
        all_flat_sequences = []
        for i in range(N):
            normed = self._normalize(action_chunks[i])
            coeffs = self._dct_encode(normed)
            qcoeffs = self._quantize(coeffs)
            flat = self._flatten(qcoeffs)
            all_flat_sequences.append(flat)

        # Step 3: Build base vocabulary (all unique integers)
        all_ints = set()
        for seq in all_flat_sequences:
            all_ints.update(seq)

        min_int = min(all_ints)
        max_int = max(all_ints)
        self._int_offset = -min_int  # shift so min -> 0
        self._base_vocab_size = max_int - min_int + 1

        print(f"[FAST] Integer range: [{min_int}, {max_int}], "
              f"base vocab: {self._base_vocab_size}, offset: {self._int_offset}")

        # Step 4: Convert to base tokens
        base_sequences = []
        for seq in all_flat_sequences:
            base_sequences.append([self._int_to_base(i) for i in seq])

        # Step 5: Train BPE
        if num_bpe_merges is None:
            # Target: vocab_size total tokens
            num_bpe_merges = max(0, self.vocab_size - self._base_vocab_size)

        self.bpe_merges = []
        self.bpe_vocab = {}
        self.bpe_vocab_inv = {}

        # Initialize: base vocab entries are their own expansion
        for base_tok in range(self._base_vocab_size):
            self.bpe_vocab[base_tok] = [base_tok]

        next_token_id = self._base_vocab_size

        print(f"[FAST] Training BPE with {num_bpe_merges} merges...")
        for merge_idx in range(num_bpe_merges):
            # Count bigram frequencies across all sequences
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq in base_sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                print(f"[FAST] No more pairs to merge at step {merge_idx}")
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                print(f"[FAST] Best pair count < 2 at step {merge_idx}, stopping")
                break

            # Record merge
            self.bpe_merges.append(best_pair)

            # Create new token
            a, b = best_pair
            new_expansion = []
            if a in self.bpe_vocab:
                new_expansion.extend(self.bpe_vocab[a])
            else:
                new_expansion.append(a)
            if b in self.bpe_vocab:
                new_expansion.extend(self.bpe_vocab[b])
            else:
                new_expansion.append(b)

            self.bpe_vocab[next_token_id] = new_expansion
            self.bpe_vocab_inv[best_pair] = next_token_id

            # Apply merge to all sequences
            for seq_idx in range(len(base_sequences)):
                new_seq = []
                i = 0
                seq = base_sequences[seq_idx]
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                        new_seq.append(next_token_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                base_sequences[seq_idx] = new_seq

            next_token_id += 1

            if (merge_idx + 1) % 100 == 0:
                avg_len = np.mean([len(s) for s in base_sequences])
                print(f"[FAST] Merge {merge_idx + 1}/{num_bpe_merges}: "
                      f"pair={best_pair} (count={best_count}), "
                      f"avg seq len={avg_len:.1f}")

        total_vocab = next_token_id
        avg_tokens = np.mean([len(s) for s in base_sequences])
        orig_len = self.horizon * self.action_dim
        print(f"[FAST] Done: {len(self.bpe_merges)} merges, "
              f"total vocab={total_vocab}, "
              f"avg tokens/chunk={avg_tokens:.1f} (from {orig_len}), "
              f"compression={orig_len / avg_tokens:.1f}x")

        return self

    # ========================================================================
    # Save / Load
    # ========================================================================

    def save(self, path: str) -> None:
        """Save tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        state = {
            "scale": self.scale,
            "vocab_size": self.vocab_size,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "int_offset": self._int_offset,
            "base_vocab_size": self._base_vocab_size,
            "bpe_merges": self.bpe_merges,
            "bpe_vocab": {str(k): v for k, v in self.bpe_vocab.items()},
            "bpe_vocab_inv": {str(k): v for k, v in self.bpe_vocab_inv.items()},
        }
        with open(os.path.join(path, "fast_tokenizer.json"), "w") as f:
            json.dump(state, f, indent=2)
        print(f"[FAST] Saved tokenizer to {path}")

    @classmethod
    def load(cls, path: str) -> "FASTTokenizer":
        """Load tokenizer from directory."""
        with open(os.path.join(path, "fast_tokenizer.json")) as f:
            state = json.load(f)

        tok = cls(
            scale=state["scale"],
            vocab_size=state["vocab_size"],
            horizon=state["horizon"],
            action_dim=state["action_dim"],
        )
        tok.q01 = np.array(state["q01"])
        tok.q99 = np.array(state["q99"])
        tok._int_offset = state["int_offset"]
        tok._base_vocab_size = state["base_vocab_size"]
        tok.bpe_merges = [tuple(m) for m in state["bpe_merges"]]
        tok.bpe_vocab = {int(k): v for k, v in state["bpe_vocab"].items()}
        tok.bpe_vocab_inv = {
            tuple(eval(k)): v for k, v in state["bpe_vocab_inv"].items()
        }
        print(f"[FAST] Loaded tokenizer: {len(tok.bpe_merges)} merges, "
              f"H={tok.horizon}, D={tok.action_dim}")
        return tok

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size (base + BPE merges)."""
        return self._base_vocab_size + len(self.bpe_merges)


# ============================================================================
# Utility: train tokenizer from RLDS/LIBERO data
# ============================================================================

def train_fast_tokenizer_from_dataset(
    data_root: str,
    dataset_name: str,
    horizon: int = 8,
    action_dim: int = 7,
    scale: float = 10.0,
    vocab_size: int = 1024,
    max_chunks: int = 50000,
    save_path: Optional[str] = None,
) -> FASTTokenizer:
    """Train a FAST tokenizer on action data from an RLDS dataset.

    This loads action chunks from the dataset statistics or raw data
    and trains a BPE tokenizer.

    Args:
        data_root: Root directory containing RLDS datasets.
        dataset_name: Name of the RLDS dataset.
        horizon: Action chunk length.
        action_dim: Number of action dimensions.
        scale: DCT quantization scale.
        vocab_size: BPE vocabulary size.
        max_chunks: Maximum number of action chunks to use for training.
        save_path: Optional path to save the trained tokenizer.

    Returns:
        Trained FASTTokenizer.
    """
    import tensorflow_datasets as tfds

    print(f"[FAST] Loading action data from {dataset_name}...")

    # Load dataset
    builder = tfds.builder(dataset_name, data_dir=data_root)
    ds = builder.as_dataset(split="train")

    # Collect action chunks
    chunks = []
    for episode in ds.take(max_chunks * 2):  # take extra in case some are short
        steps = list(episode["steps"])
        if len(steps) < horizon:
            continue
        for start in range(0, len(steps) - horizon + 1, horizon):
            chunk = np.array([steps[start + i]["action"] for i in range(horizon)])
            if chunk.shape == (horizon, action_dim):
                chunks.append(chunk)
                if len(chunks) >= max_chunks:
                    break
        if len(chunks) >= max_chunks:
            break

    chunks = np.array(chunks[:max_chunks])
    print(f"[FAST] Collected {len(chunks)} action chunks of shape {chunks.shape}")

    # Train tokenizer
    tokenizer = FASTTokenizer(
        scale=scale,
        vocab_size=vocab_size,
        horizon=horizon,
        action_dim=action_dim,
    )
    tokenizer.fit(chunks)

    # Save if requested
    if save_path:
        tokenizer.save(save_path)

    return tokenizer
