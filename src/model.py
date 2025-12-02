"""
HOPE Model: Full architecture implementation.

HOPE (Hierarchical Optimization with Persistent Experience) is a
self-referential learning module that combines:
1. Self-modifying Titans for memory-augmented sequence processing
2. Continuum Memory System for multi-timescale knowledge storage
3. Nested learning optimization at multiple levels

Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
import math

from src.config import HopeConfig
from src.modules.hope_block import HopeBlock, HopeBlockStack


class Hope(nn.Module):
    """
    HOPE: Hierarchical Optimization with Persistent Experience.

    A self-referential sequence model that learns to:
    1. Store and retrieve information using neural memory
    2. Update memory based on surprise (prediction error)
    3. Maintain multi-scale knowledge through continuum memory

    Architecture:
        Embedding -> [HopeBlock x N] -> LayerNorm -> Output Projection

    Each HopeBlock contains:
        - Self-Modifying Titans (memory attention)
        - Continuum Memory System (multi-frequency FFN)
    """

    def __init__(self, config: HopeConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.dim)

        # Input dropout
        self.input_dropout = nn.Dropout(config.dropout)

        # HOPE blocks
        self.blocks = nn.ModuleList([
            HopeBlock(
                dim=config.dim,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_memory_levels=config.num_memory_levels,
                chunk_sizes=config.chunk_sizes,
                ffn_expansion=config.ffn_expansion,
                learning_rate=config.learning_rate_memory,
                momentum=config.momentum_decay,
                use_delta_rule=config.use_delta_rule,
                dropout=config.dropout,
                eps=config.eps,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.dim)

        # Output projection (tied with embedding by default)
        self.output_proj = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.output_proj.weight = self.embedding.weight

        # Initialize weights
        self.applyInit()

    def applyInit(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # Initialize other layers
        self.apply(self._initWeights)

    def _initWeights(self, module):
        """Initialize weights for a single module."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[List[torch.Tensor]] = None,
        step: Optional[int] = None,
        return_memory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through HOPE model.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            memory_states: Optional list of memory states for each block
            step: Current global step (for CMS update scheduling)
            return_memory: Whether to return updated memory states

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            memory_states: Updated memory states (if return_memory=True)
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = self.embedding(input_ids)

        # Add positional encoding
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_embedding(positions)

        # Input dropout
        x = self.input_dropout(x)

        # Initialize memory states if needed
        if memory_states is None:
            memory_states = [None] * len(self.blocks)

        # Process through HOPE blocks
        new_memory_states = []
        for i, block in enumerate(self.blocks):
            x, memory_state = block(
                x,
                memory_state=memory_states[i],
                step=step,
                return_memory=return_memory,
            )
            new_memory_states.append(memory_state)

        # Final normalization
        x = self.final_norm(x)

        # Output projection
        logits = self.output_proj(x)

        if return_memory:
            return logits, new_memory_states
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        memory_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            memory_states: Optional initial memory states

        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get logits for last position
                logits, memory_states = self.forward(
                    generated,
                    memory_states=memory_states,
                    step=step,
                    return_memory=True,
                )
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    values, _ = torch.topk(logits, top_k)
                    min_value = values[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < min_value,
                        torch.full_like(logits, float("-inf")),
                        logits,
                    )

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    cumsum_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_mask = cumsum_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False

                    indices_to_remove = sorted_mask.scatter(
                        1, sorted_indices, sorted_mask
                    )
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

                # Truncate if exceeds max_seq_len
                if generated.shape[1] > self.config.max_seq_len:
                    generated = generated[:, -self.config.max_seq_len:]

        return generated

    def getNumParams(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.weight.numel()
        return n_params

    def resetMemory(self) -> List[None]:
        """Reset all memory states."""
        return [None] * len(self.blocks)


class HopeForCausalLM(Hope):
    """
    HOPE model wrapper for causal language modeling.

    Adds loss computation and convenience methods for training.
    """

    def __init__(self, config: HopeConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        memory_states: Optional[List[torch.Tensor]] = None,
        step: Optional[int] = None,
        return_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            labels: Target token IDs for loss computation (batch, seq_len)
            memory_states: Optional memory states
            step: Current global step
            return_memory: Whether to return memory states

        Returns:
            Dictionary with 'logits' and optionally 'loss', 'memory_states'
        """
        if return_memory:
            logits, memory_states = super().forward(
                input_ids,
                memory_states=memory_states,
                step=step,
                return_memory=True,
            )
        else:
            logits = super().forward(
                input_ids,
                memory_states=memory_states,
                step=step,
                return_memory=False,
            )
            memory_states = None

        output = {"logits": logits}

        if labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        if return_memory:
            output["memory_states"] = memory_states

        return output


class HopeForSequenceClassification(Hope):
    """
    HOPE model for sequence classification tasks.
    """

    def __init__(self, config: HopeConfig, num_labels: int):
        super().__init__(config)
        self.num_labels = num_labels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        memory_states: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sequence classification.

        Uses the last token representation for classification.
        """
        # Get sequence representations
        logits = super().forward(
            input_ids, memory_states=memory_states, return_memory=False
        )

        # Use last token for classification
        # Shape: (batch, seq_len, vocab_size) -> get hidden states
        # We need hidden states before output projection
        # For simplicity, we pool over sequence
        hidden_states = self.final_norm(
            self.embedding(input_ids) +
            self.pos_embedding(
                torch.arange(input_ids.shape[1], device=input_ids.device)
            )
        )

        for block in self.blocks:
            hidden_states, _ = block(hidden_states)

        # Mean pooling
        pooled = hidden_states.mean(dim=1)

        # Classify
        class_logits = self.classifier(pooled)

        output = {"logits": class_logits}

        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss = F.mse_loss(class_logits.squeeze(), labels.float())
            else:
                # Classification
                loss = F.cross_entropy(class_logits, labels)
            output["loss"] = loss

        return output


def createHopeModel(
    model_size: str = "base",
    vocab_size: int = 32000,
    **kwargs,
) -> Hope:
    """
    Factory function to create HOPE models.

    Args:
        model_size: One of 'small', 'base', 'large', 'xl'
        vocab_size: Vocabulary size
        **kwargs: Additional config overrides

    Returns:
        Hope model instance
    """
    from src.config import (
        HopeSmallConfig,
        HopeBaseConfig,
        HopeLargeConfig,
        HopeXLConfig,
    )

    config_map = {
        "small": HopeSmallConfig,
        "base": HopeBaseConfig,
        "large": HopeLargeConfig,
        "xl": HopeXLConfig,
    }

    if model_size not in config_map:
        raise ValueError(f"Unknown model size: {model_size}")

    config_class = config_map[model_size]
    config = config_class(vocab_size=vocab_size, **kwargs)

    return Hope(config)
