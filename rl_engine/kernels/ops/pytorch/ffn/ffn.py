# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bias-free gated FFN assembled from deterministic CUDA kernels."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

QWEN3_8B_HIDDEN_SIZE = 4096
QWEN3_8B_INTERMEDIATE_SIZE = 12288

_REQUIRED_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_db",
    "swiglu_forward",
    "swiglu_backward",
)


def _require_ffn_kernels() -> None:
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(_C, name)]
    if not _EXT_AVAILABLE or _C is None or missing:
        suffix = f" Missing symbols: {', '.join(missing)}." if missing else ""
        raise RuntimeError(
            "qwen3_ffn requires the compiled deterministic GEMM and "
            f"SwiGLU CUDA kernels.{suffix}"
        )


def _require_parallel_group(group: Any, name: str):
    if group is None:
        return None

    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError(f"{name}-parallel FFN requires torch.distributed.")
    if not dist.is_initialized():
        raise RuntimeError(f"{name}-parallel FFN requires an initialized process group.")
    if dist.get_world_size(group=group) <= 1:
        raise ValueError(f"{name}_group must contain at least two ranks.")
    return dist


def _validate_ffn_inputs(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> None:
    tensors = {
        "rmsnorm_output": rmsnorm_output,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}.")

    if rmsnorm_output.dim() < 1:
        raise ValueError("rmsnorm_output must have at least one dimension.")
    if rmsnorm_output.numel() == 0:
        raise ValueError("rmsnorm_output must contain at least one token.")
    for name, weight in (
        ("gate_weight", gate_weight),
        ("up_weight", up_weight),
        ("down_weight", down_weight),
    ):
        if weight.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(weight.shape)}.")

    hidden_size = rmsnorm_output.size(-1)
    intermediate_size = gate_weight.size(0)
    expected_shapes = {
        "gate_weight": (intermediate_size, hidden_size),
        "up_weight": (intermediate_size, hidden_size),
        "down_weight": (hidden_size, intermediate_size),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}.")

    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16, got {tensor.dtype}.")
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be on a CUDA device, got '{tensor.device}'.")
        if tensor.device != rmsnorm_output.device:
            raise RuntimeError(
                f"all FFN inputs must be on {rmsnorm_output.device}, "
                f"got {name} on {tensor.device}."
            )


def _all_gather_tokens(tensor: Tensor, dist, group: Any) -> Tensor:
    world_size = dist.get_world_size(group=group)
    output = torch.empty(
        (world_size * tensor.size(0), *tensor.shape[1:]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # TODO: NCCL AllGather can expose cross-configuration mismatch. Replace it
    # with the custom deterministic AllGather and compare both paths.
    dist.all_gather_into_tensor(output, tensor.contiguous(), group=group)
    return output


def _reduce_scatter_tokens(tensor: Tensor, dist, group: Any) -> Tensor:
    world_size = dist.get_world_size(group=group)
    if tensor.size(0) % world_size != 0:
        raise ValueError(
            "the gathered token count must be divisible by the tensor-parallel "
            f"world size, got {tensor.size(0)} and {world_size}."
        )

    local_tokens = tensor.size(0) // world_size
    output = torch.empty(
        (local_tokens, *tensor.shape[1:]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # TODO: NCCL ReduceScatter can change the reduction order and cause
    # mismatch. Replace it with the custom deterministic ReduceScatter and
    # compare both paths.
    dist.reduce_scatter_tensor(
        output,
        tensor.contiguous(),
        op=dist.ReduceOp.SUM,
        group=group,
    )
    return output


class _DeterministicFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rmsnorm_output: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        tp_group: Any,
        cp_group: Any,
        sequence_parallel: bool,
    ) -> Tensor:
        tp_dist = _require_parallel_group(tp_group, "tensor")
        _require_parallel_group(cp_group, "context")
        if sequence_parallel and tp_dist is None:
            raise ValueError("sequence_parallel requires a tensor-parallel group.")
        if sequence_parallel and str(tp_dist.get_backend(tp_group)) != "nccl":
            raise RuntimeError("sequence-parallel FFN currently requires an NCCL process group.")

        input_shape = rmsnorm_output.shape
        rmsnorm_output_2d = rmsnorm_output.reshape(-1, input_shape[-1]).contiguous()
        if sequence_parallel:
            rmsnorm_output_2d = _all_gather_tokens(rmsnorm_output_2d, tp_dist, tp_group)

        # The model stores projection weights as [out, in]; GEMM consumes [K, N].
        gate = _C.det_gemm_fwd(rmsnorm_output_2d, gate_weight.t().contiguous())
        up = _C.det_gemm_fwd(rmsnorm_output_2d, up_weight.t().contiguous())
        activated = _C.swiglu_forward(gate, up)
        output = _C.det_gemm_fwd(activated, down_weight.t().contiguous())

        if sequence_parallel:
            output = _reduce_scatter_tokens(output, tp_dist, tp_group)
        elif tp_dist is not None:
            # TODO: CUDA currently uses NCCL. Replace it with the custom
            # deterministic AllReduce and compare both communication paths.
            tp_dist.all_reduce(output, op=tp_dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(
            rmsnorm_output_2d,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        )
        ctx.input_shape = input_shape
        ctx.tp_group = tp_group
        ctx.cp_group = cp_group
        ctx.sequence_parallel = sequence_parallel
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            rmsnorm_output,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        ) = ctx.saved_tensors
        tp_dist = _require_parallel_group(ctx.tp_group, "tensor")
        cp_dist = _require_parallel_group(ctx.cp_group, "context")
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
        if ctx.sequence_parallel:
            grad_output = _all_gather_tokens(grad_output, tp_dist, ctx.tp_group)

        # Down weight gradients use the same coordinates across CP ranks.
        grad_down_weight = _C.det_gemm_db(activated, grad_output).t().contiguous()
        if cp_dist is not None:
            # TODO: CUDA currently uses NCCL. Replace it with the custom
            # deterministic AllReduce and compare both communication paths.
            cp_dist.all_reduce(
                grad_down_weight,
                op=cp_dist.ReduceOp.SUM,
                group=ctx.cp_group,
            )

        # Down input-gradient shards concatenate across TP; no TP reduction.
        grad_activated = _C.det_gemm_fwd(grad_output, down_weight)
        grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate, up)

        # Gate/Up weight-gradient shards concatenate across TP and reduce across CP.
        grad_gate_weight = _C.det_gemm_db(rmsnorm_output, grad_gate).t().contiguous()
        if cp_dist is not None:
            cp_dist.all_reduce(
                grad_gate_weight,
                op=cp_dist.ReduceOp.SUM,
                group=ctx.cp_group,
            )

        grad_up_weight = _C.det_gemm_db(rmsnorm_output, grad_up).t().contiguous()
        if cp_dist is not None:
            cp_dist.all_reduce(
                grad_up_weight,
                op=cp_dist.ReduceOp.SUM,
                group=ctx.cp_group,
            )

        # Gate/Up input gradients reduce across TP, then add locally.
        grad_rmsnorm_from_gate = _C.det_gemm_fwd(grad_gate, gate_weight)
        if ctx.sequence_parallel:
            grad_rmsnorm_from_gate = _reduce_scatter_tokens(
                grad_rmsnorm_from_gate,
                tp_dist,
                ctx.tp_group,
            )
        elif tp_dist is not None:
            # TODO: CUDA currently uses NCCL. Replace it with the custom
            # deterministic AllReduce and compare both communication paths.
            tp_dist.all_reduce(
                grad_rmsnorm_from_gate,
                op=tp_dist.ReduceOp.SUM,
                group=ctx.tp_group,
            )

        grad_rmsnorm_from_up = _C.det_gemm_fwd(grad_up, up_weight)
        if ctx.sequence_parallel:
            grad_rmsnorm_from_up = _reduce_scatter_tokens(
                grad_rmsnorm_from_up,
                tp_dist,
                ctx.tp_group,
            )
        elif tp_dist is not None:
            tp_dist.all_reduce(
                grad_rmsnorm_from_up,
                op=tp_dist.ReduceOp.SUM,
                group=ctx.tp_group,
            )

        grad_rmsnorm_output = grad_rmsnorm_from_gate.add_(grad_rmsnorm_from_up)
        return (
            grad_rmsnorm_output.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
            None,
            None,
            None,
        )


def qwen3_ffn(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    tp_group: Any = None,
    cp_group: Any = None,
    sequence_parallel: bool = False,
) -> Tensor:
    """Apply a bias-free SiLU-gated FFN with deterministic backward kernels.

    Args:
        rmsnorm_output: RMSNorm output, shape ``[..., H]``.
        gate_weight: Gate projection weight in ``[out, in]`` layout, shape
            ``[I_local, H]``.
        up_weight: Up projection weight in ``[out, in]`` layout, shape
            ``[I_local, H]``.
        down_weight: Down projection weight in ``[out, in]`` layout, shape
            ``[H, I_local]``.
        tp_group: Optional tensor-parallel process group. Gate and Up are
            column-parallel; Down is row-parallel.
        cp_group: Optional context-parallel process group. Each rank owns
            different token rows and the same local weight shards.
        sequence_parallel: Whether ``rmsnorm_output`` and the returned output
            are sharded on the flattened token dimension across ``tp_group``.

    Returns:
        FFN output with shape ``[..., H]``.
    """
    _validate_ffn_inputs(rmsnorm_output, gate_weight, up_weight, down_weight)
    _require_ffn_kernels()
    if not isinstance(sequence_parallel, bool):
        raise TypeError("sequence_parallel must be a bool.")
    return _DeterministicFFNFunction.apply(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        tp_group,
        cp_group,
        sequence_parallel,
    )
