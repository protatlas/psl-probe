"""Uniform wrapper over the protein language models we probe.

ESM-2 and ProtT5 differ in tokenizer conventions, special-token placement and
encoder API. :class:`PLMWrapper` normalises these differences so that downstream
code always receives, for every layer ``l``, a tensor of shape
``(batch, n_residues, hidden)`` in which position ``i`` corresponds to residue
``i`` of the *amino-acid* sequence -- special tokens already stripped. Getting
this alignment right is a prerequisite for the residue-level analyses; it is
verified in ``tests/test_alignment.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor

from psl_probe.config import ModelConfig
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

#: Rare/ambiguous residues that ProtT5's vocabulary does not cover.
_PROTT5_UNK_PATTERN = re.compile(r"[UZOB]")


@dataclass
class PLMOutput:
    """Per-layer hidden states aligned to amino-acid positions.

    Attributes
    ----------
    hidden_states:
        Tensor of shape ``(n_layers, batch, max_residues, hidden)``. Layer 0 is the
        input embedding layer; layer ``n`` is the output of transformer block ``n``.
    mask:
        Boolean tensor ``(batch, max_residues)``; ``True`` at real residue positions.
    attentions:
        Optional tuple of ``(batch, heads, max_residues, max_residues)`` tensors,
        one per layer, already cropped to residue positions. ``None`` unless the
        model was loaded with ``output_attentions=True``.
    """

    hidden_states: Tensor
    mask: Tensor
    attentions: tuple[Tensor, ...] | None = None


class PLMWrapper:
    """Load a pLM and expose residue-aligned, layer-wise representations."""

    def __init__(self, cfg: ModelConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = _DTYPES[cfg.dtype]
        self.family = cfg.family.lower()
        self.tokenizer, self.model = self._load()
        self.model.eval()

        # ``n_layers + 1`` hidden states are returned (embeddings + one per block).
        self.n_hidden_states = self.model.config.num_hidden_layers + 1
        self.hidden_size = self.model.config.hidden_size
        logger.info(
            "Loaded %s (%s): %d layers, hidden=%d, dtype=%s",
            cfg.tag, cfg.hf_id, self.model.config.num_hidden_layers,
            self.hidden_size, cfg.dtype,
        )

    # -- loading ------------------------------------------------------------
    def _load(self):
        from transformers import AutoTokenizer

        # Transformers defaults to the fused SDPA attention kernel, which cannot
        # return attention probabilities: with SDPA, ``output_attentions=True`` is
        # silently ignored and ``outputs.attentions`` comes back empty. The
        # attention-alignment analysis therefore requires the eager implementation,
        # which we select only when attentions are actually needed -- eager is
        # noticeably slower and far more memory-hungry at long sequence lengths.
        load_kwargs: dict = {"dtype": self.dtype}
        if self.cfg.output_attentions:
            load_kwargs["attn_implementation"] = "eager"

        if self.family == "prott5":
            from transformers import T5EncoderModel

            tokenizer = AutoTokenizer.from_pretrained(self.cfg.hf_id, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(self.cfg.hf_id, **load_kwargs)
        elif self.family == "ankh":
            from transformers import T5EncoderModel

            tokenizer = AutoTokenizer.from_pretrained(self.cfg.hf_id)
            model = T5EncoderModel.from_pretrained(self.cfg.hf_id, **load_kwargs)
        elif self.family == "esm2":
            from transformers import AutoModel

            tokenizer = AutoTokenizer.from_pretrained(self.cfg.hf_id)
            # ``add_pooling_layer=False`` suppresses a randomly initialised pooler
            # head that we never use and that would otherwise be reported as
            # missing weights on every load.
            model = AutoModel.from_pretrained(
                self.cfg.hf_id, add_pooling_layer=False, **load_kwargs
            )
        else:
            raise ValueError(f"Unsupported model family: {self.cfg.family!r}")

        return tokenizer, model.to(self.device)

    # -- tokenisation --------------------------------------------------------
    def _prepare(self, sequences: list[str]) -> list[str]:
        """Apply family-specific input formatting."""
        truncated = [s[: self.cfg.max_length] for s in sequences]
        if self.family in {"prott5", "ankh"}:
            # T5-based protein models expect whitespace-separated residues and map
            # the rare residues U/Z/O/B to the unknown token.
            return [" ".join(_PROTT5_UNK_PATTERN.sub("X", s)) for s in truncated]
        return truncated

    def _residue_slice(self, attention_mask: Tensor) -> tuple[int, Tensor]:
        """Return ``(offset, residue_mask)`` mapping tokens to residue positions.

        ESM-2 wraps the sequence in ``<cls> ... <eos>``; ProtT5 appends a single
        ``</s>`` with no prefix token. In both cases the residue-aligned span is
        obtained by dropping the leading offset and every padding/terminal token.
        """
        offset = 1 if self.family == "esm2" else 0
        mask = attention_mask.bool()[:, offset:]
        # Drop the trailing terminal token of each sequence.
        lengths = mask.sum(dim=1)
        residue_mask = torch.zeros_like(mask)
        for i, length in enumerate(lengths.tolist()):
            residue_mask[i, : max(length - 1, 0)] = True
        return offset, residue_mask

    # -- inference -----------------------------------------------------------
    @torch.inference_mode()
    def encode(self, sequences: list[str]) -> PLMOutput:
        """Embed a batch of sequences and return residue-aligned hidden states."""
        prepared = self._prepare(sequences)
        encoded = self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length + 2,
            add_special_tokens=True,
        ).to(self.device)

        outputs = self.model(
            **encoded,
            output_hidden_states=True,
            output_attentions=self.cfg.output_attentions,
        )

        offset, residue_mask = self._residue_slice(encoded["attention_mask"])
        n_residues = int(residue_mask.shape[1])

        hidden = torch.stack(
            [h[:, offset : offset + n_residues, :] for h in outputs.hidden_states], dim=0
        )
        hidden = hidden * residue_mask[None, :, :, None].to(hidden.dtype)

        attentions = None
        if self.cfg.output_attentions:
            raw = getattr(outputs, "attentions", None)
            if not raw:
                raise RuntimeError(
                    "output_attentions=True but the model returned no attentions. "
                    "This happens when a fused attention kernel (SDPA/flash) is "
                    "active; load the model with attn_implementation='eager'."
                )
            attentions = tuple(
                a[:, :, offset : offset + n_residues, offset : offset + n_residues]
                for a in raw
            )

        return PLMOutput(hidden_states=hidden, mask=residue_mask, attentions=attentions)


def load_plm(cfg: ModelConfig, device: str = "cuda") -> PLMWrapper:
    """Convenience constructor mirroring the config-driven CLI style."""
    return PLMWrapper(cfg, device=device)
