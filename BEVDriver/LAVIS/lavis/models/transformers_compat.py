"""
Compatibility helpers for huggingface transformers utilities.

Transformers occasionally relocates helpers like `apply_chunking_to_forward`
or pruning utilities. Centralising the fallbacks here keeps the rest of LAVIS
code unchanged even when the upstream API shifts.
"""

from typing import Callable, List, Set, Tuple

import torch
from torch import nn
from transformers import modeling_utils

# Base class is still exposed but load via getattr to avoid AttributeError on
# older/newer releases that move things around.
PreTrainedModel = getattr(modeling_utils, "PreTrainedModel")

apply_chunking_to_forward = getattr(modeling_utils, "apply_chunking_to_forward", None)
if apply_chunking_to_forward is None:  # pragma: no cover - executed via HF internals

    def apply_chunking_to_forward(
        forward_fn: Callable, chunk_size: int, chunk_dim: int, *input_tensors: torch.Tensor
    ):
        """
        Minimal re-implementation of transformers' helper used to reduce memory
        by slicing long sequences into smaller chunks.
        """

        if chunk_size <= 0:
            return forward_fn(*input_tensors)

        tensor_shape = input_tensors[0].shape[chunk_dim]
        for tensor in input_tensors:
            if tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError("All input tensors must have the same shape")
        if tensor_shape % chunk_size != 0:
            raise ValueError("Chunk size must divide sequence length")
        num_chunks = tensor_shape // chunk_size
        input_chunks = [tensor.chunk(num_chunks, dim=chunk_dim) for tensor in input_tensors]
        output_chunks = [forward_fn(*chunk) for chunk in zip(*input_chunks)]
        return torch.cat(output_chunks, dim=chunk_dim)


find_pruneable_heads_and_indices = getattr(modeling_utils, "find_pruneable_heads_and_indices", None)
if find_pruneable_heads_and_indices is None:

    def find_pruneable_heads_and_indices(
        heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
    ) -> Tuple[Set[int], torch.LongTensor]:
        """Matches the behaviour of the legacy helper from transformers."""

        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index: torch.LongTensor = torch.arange(len(mask))[mask].long()
        return heads, index


prune_linear_layer = getattr(modeling_utils, "prune_linear_layer", None)
if prune_linear_layer is None:

    def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
        """Keep only entries specified by `index` along dimension `dim`."""

        index = index.to(layer.weight.device)
        weight = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            bias = layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(weight.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(bias.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer
