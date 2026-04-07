"""ESM-2 protein language model wrapper.

Wraps facebook/esm2_t33_650M_UR50D via HuggingFace transformers for
per-residue embeddings and ligandability scoring.

Gracefully degrades to a fallback mode that returns simulated embeddings
when PyTorch or transformers is not installed.
"""

from __future__ import annotations

import hashlib
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from covalent_agent.config import settings

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Attempt to import torch + transformers; set a module-level flag
# ---------------------------------------------------------------------------

_HAS_TORCH = False

try:
    import torch  # noqa: F401
    from transformers import AutoModel, AutoTokenizer  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants for fallback mode
# ---------------------------------------------------------------------------

_FALLBACK_HIDDEN_DIM = 1280  # ESM2-650M hidden dimension
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Intrinsic nucleophilicity scores by residue identity (0-1 scale).
# Cysteine thiol is the most nucleophilic; lysine amine and
# hydroxyl/carboxylate side chains follow.
_RESIDUE_NUCLEOPHILICITY: dict[str, float] = {
    "C": 0.92,
    "K": 0.60,
    "Y": 0.48,
    "S": 0.42,
    "T": 0.40,
    "D": 0.35,
    "E": 0.33,
    "H": 0.30,
    "R": 0.25,
    "N": 0.22,
    "Q": 0.20,
}

# Default score for residues with low/no intrinsic nucleophilicity.
_DEFAULT_NUCLEOPHILICITY = 0.10


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_esm_model(model_name: str) -> tuple:
    """Load ESM-2 tokenizer and model. Cached to avoid repeated loading.

    Returns (tokenizer, model) tuple, or (None, None) if torch is unavailable.
    """
    if not _HAS_TORCH:
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# ESMWrapper
# ---------------------------------------------------------------------------

class ESMWrapper:
    """Wrapper around facebook/esm2_t33_650M_UR50D.

    If PyTorch or transformers is not installed, the wrapper operates in
    *fallback mode*: it returns deterministic simulated embeddings derived
    from sequence hashing and residue-identity heuristics, and prints a
    one-time warning.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.esm_model
        self.fallback_mode = not _HAS_TORCH

        if self.fallback_mode:
            warnings.warn(
                "PyTorch/transformers not available. ESMWrapper running in "
                "fallback mode with simulated embeddings.",
                stacklevel=2,
            )
            self._tokenizer = None
            self._model = None
        else:
            self._tokenizer, self._model = _load_esm_model(self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sequence_embeddings(self, sequence: str) -> np.ndarray:
        """Return per-residue embeddings.

        Returns:
            numpy array of shape ``(seq_len, hidden_dim)``.
        """
        if self.fallback_mode:
            return self._fallback_embeddings(sequence)

        return self._real_embeddings(sequence)

    def get_residue_embedding(self, sequence: str, position: int) -> np.ndarray:
        """Return the embedding vector for a single residue (0-indexed).

        Args:
            sequence: amino acid sequence.
            position: 0-indexed residue position.

        Returns:
            numpy array of shape ``(hidden_dim,)``.
        """
        embeddings = self.get_sequence_embeddings(sequence)
        if position < 0 or position >= len(sequence):
            raise IndexError(
                f"Position {position} out of range for sequence length {len(sequence)}"
            )
        return embeddings[position]

    def score_residue_ligandability(self, sequence: str, position: int) -> float:
        """Heuristic 0-1 ligandability score for a residue.

        In real mode: combines embedding norm and local attention variance.
        In fallback mode: uses residue identity, solvent-exposure proxy
        (distance from termini), and context diversity.

        Args:
            sequence: amino acid sequence.
            position: 0-indexed residue position.

        Returns:
            float between 0.0 and 1.0.
        """
        if self.fallback_mode:
            return self._fallback_ligandability(sequence, position)

        return self._real_ligandability(sequence, position)

    def get_context_window(
        self, sequence: str, position: int, window: int = 15
    ) -> str:
        """Return the amino acid context window around *position*.

        Args:
            sequence: amino acid sequence.
            position: 0-indexed residue position.
            window: number of residues on each side.

        Returns:
            Substring of *sequence* centred on *position*.
        """
        start = max(0, position - window)
        end = min(len(sequence), position + window + 1)
        return sequence[start:end]

    # ------------------------------------------------------------------
    # Real-model helpers
    # ------------------------------------------------------------------

    def _real_embeddings(self, sequence: str) -> np.ndarray:
        """Compute per-residue embeddings using the real ESM-2 model."""
        inputs = self._tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.last_hidden_state shape: (1, seq_len+2, hidden_dim)
        # Strip the <cls> and <eos> tokens.
        hidden = outputs.last_hidden_state[0, 1:-1, :].numpy()
        return hidden

    def _real_ligandability(self, sequence: str, position: int) -> float:
        """Score ligandability using real embedding norms and variance."""
        embeddings = self.get_sequence_embeddings(sequence)
        if position < 0 or position >= embeddings.shape[0]:
            raise IndexError(
                f"Position {position} out of range for sequence length "
                f"{embeddings.shape[0]}"
            )

        residue_emb = embeddings[position]

        # Component 1: normalised embedding norm (higher norm -> more
        # "informative" representation, loosely correlated with functional
        # importance).
        all_norms = np.linalg.norm(embeddings, axis=1)
        max_norm = all_norms.max() if all_norms.max() > 0 else 1.0
        norm_score = float(np.linalg.norm(residue_emb) / max_norm)

        # Component 2: local embedding variance in a context window.
        window = 15
        start = max(0, position - window)
        end = min(embeddings.shape[0], position + window + 1)
        local_embs = embeddings[start:end]
        local_var = float(np.var(local_embs))
        global_var = float(np.var(embeddings)) if np.var(embeddings) > 0 else 1.0
        var_score = min(local_var / global_var, 1.0)

        # Component 3: intrinsic nucleophilicity.
        aa = sequence[position].upper()
        nuc_score = _RESIDUE_NUCLEOPHILICITY.get(aa, _DEFAULT_NUCLEOPHILICITY)

        # Weighted combination.
        score = 0.35 * norm_score + 0.25 * var_score + 0.40 * nuc_score
        return round(min(max(score, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    def _fallback_embeddings(self, sequence: str) -> np.ndarray:
        """Generate deterministic simulated per-residue embeddings.

        Uses a seeded PRNG derived from the sequence hash so the same
        sequence always produces the same output.
        """
        seed = int(hashlib.sha256(sequence.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(len(sequence), _FALLBACK_HIDDEN_DIM).astype(np.float32)

        # Modulate each residue vector slightly by amino-acid identity so
        # that different residue types have distinguishable embeddings.
        for i, aa in enumerate(sequence):
            aa_seed = ord(aa.upper()) / 128.0
            embeddings[i] *= 0.8 + 0.4 * aa_seed

        return embeddings

    def _fallback_ligandability(self, sequence: str, position: int) -> float:
        """Heuristic ligandability without a real model.

        Combines:
          - Intrinsic nucleophilicity of the amino acid.
          - Solvent-exposure proxy (residues far from both termini are
            more likely buried; residues near ends are more exposed).
          - Local sequence diversity (more diverse = less structured = more
            accessible).
        """
        if position < 0 or position >= len(sequence):
            raise IndexError(
                f"Position {position} out of range for sequence length {len(sequence)}"
            )

        seq_len = len(sequence)
        aa = sequence[position].upper()

        # 1. Intrinsic nucleophilicity.
        nuc = _RESIDUE_NUCLEOPHILICITY.get(aa, _DEFAULT_NUCLEOPHILICITY)

        # 2. Solvent-exposure proxy: residues in the first/last ~15% of the
        #    chain are more likely to be surface-exposed in globular proteins.
        frac = position / max(seq_len - 1, 1)
        exposure = 1.0 - 2.0 * abs(frac - 0.5)  # 1 at termini, 0 at centre
        exposure = 0.3 + 0.7 * exposure  # remap to [0.3, 1.0]

        # 3. Local diversity: count unique residue types in context window.
        ctx = self.get_context_window(sequence, position, window=10)
        unique_fraction = len(set(ctx)) / max(len(ctx), 1)
        diversity = 0.4 + 0.6 * unique_fraction  # remap to [0.4, 1.0]

        score = 0.50 * nuc + 0.25 * exposure + 0.25 * diversity
        return round(min(max(score, 0.0), 1.0), 4)
