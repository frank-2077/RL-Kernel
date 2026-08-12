# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the deterministic Qwen3 dense FFN autograd assembly."""

from __future__ import annotations

import queue
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import rl_engine.kernels.ops.pytorch.ffn.ffn as ffn_module
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.ffn.ffn import (
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_INTERMEDIATE_SIZE,
    qwen3_ffn,
)

_REQUIRED_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_db",
    "swiglu_forward",
    "swiglu_backward",
)
_HAS_SM90_FFN = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 9
    and _EXT_AVAILABLE
    and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)
)

requires_cuda_ffn = pytest.mark.skipif(
    not _HAS_SM90_FFN,
    reason="FFN optimized-path validation requires SM90 and the GEMM/SwiGLU extension symbols",
)


def _gloo_available():
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


requires_gloo = pytest.mark.skipif(
    not _gloo_available(),
    reason="parallel FFN CPU test requires torch.distributed Gloo",
)


class _TorchKernelStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def det_gemm_fwd(self, a, b):
        self.calls.append("det_gemm_fwd")
        return a @ b

    def det_gemm_db(self, a, grad_output):
        self.calls.append("det_gemm_db")
        return a.t().contiguous() @ grad_output

    def swiglu_forward(self, gate, up):
        self.calls.append("swiglu_forward")
        return gate * torch.sigmoid(gate) * up

    def swiglu_backward(self, grad_output, gate, up):
        self.calls.append("swiglu_backward")
        sigmoid = torch.sigmoid(gate)
        grad_gate = grad_output * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_up = grad_output * gate * sigmoid
        return grad_gate, grad_up


def _reference(hidden_states, gate_weight, up_weight, down_weight):
    gate = hidden_states @ gate_weight.t()
    up = hidden_states @ up_weight.t()
    activated = F.silu(gate) * up
    return (activated @ down_weight.t()), gate, up, activated


def _randn(shape, *, seed, device="cpu", dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02
    return value.to(device=device, dtype=dtype)


def _tp_ffn_backward_gloo_worker(rank, world_size, init_method, result_queue):
    try:
        import torch.distributed as dist

        torch.set_num_threads(1)
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        stub = _TorchKernelStub()
        ffn_module._C = stub
        ffn_module._EXT_AVAILABLE = True
        ffn_module._validate_ffn_inputs = lambda *args: None

        token_count, hidden_size, intermediate_size = 6, 5, 12
        local_intermediate = intermediate_size // world_size
        shard_start = rank * local_intermediate
        shard_end = shard_start + local_intermediate

        rmsnorm_output = _randn((token_count, hidden_size), seed=30)
        gate_weight = _randn((intermediate_size, hidden_size), seed=31)
        up_weight = _randn((intermediate_size, hidden_size), seed=32)
        down_weight = _randn((hidden_size, intermediate_size), seed=33)
        grad_output = _randn((token_count, hidden_size), seed=34)

        reference_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (rmsnorm_output, gate_weight, up_weight, down_weight)
        ]
        reference_output, _, _, _ = _reference(*reference_inputs)
        reference_output.backward(grad_output)

        actual_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (
                rmsnorm_output,
                gate_weight[shard_start:shard_end].contiguous(),
                up_weight[shard_start:shard_end].contiguous(),
                down_weight[:, shard_start:shard_end].contiguous(),
            )
        ]
        actual_output = qwen3_ffn(
            *actual_inputs,
            tp_group=dist.group.WORLD,
        )
        actual_output.backward(grad_output)

        expected_grads = (
            reference_inputs[0].grad,
            reference_inputs[1].grad[shard_start:shard_end],
            reference_inputs[2].grad[shard_start:shard_end],
            reference_inputs[3].grad[:, shard_start:shard_end],
        )
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "max_errors": [float((actual_output - reference_output).abs().max().item())]
                + [
                    float((actual.grad - expected).abs().max().item())
                    for actual, expected in zip(actual_inputs, expected_grads, strict=True)
                ],
            }
        )
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _tp_cp_ffn_backward_gloo_worker(rank, world_size, init_method, result_queue):
    try:
        import torch.distributed as dist

        torch.set_num_threads(1)
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        tp_rank = rank % 2
        cp_rank = rank // 2
        tp_group = tp_groups[cp_rank]
        cp_group = cp_groups[tp_rank]

        stub = _TorchKernelStub()
        ffn_module._C = stub
        ffn_module._EXT_AVAILABLE = True
        ffn_module._validate_ffn_inputs = lambda *args: None

        token_count, hidden_size, intermediate_size = 8, 5, 12
        local_tokens = token_count // 2
        local_intermediate = intermediate_size // 2
        token_start = cp_rank * local_tokens
        token_end = token_start + local_tokens
        feature_start = tp_rank * local_intermediate
        feature_end = feature_start + local_intermediate

        rmsnorm_output = _randn((token_count, hidden_size), seed=40)
        gate_weight = _randn((intermediate_size, hidden_size), seed=41)
        up_weight = _randn((intermediate_size, hidden_size), seed=42)
        down_weight = _randn((hidden_size, intermediate_size), seed=43)
        grad_output = _randn((token_count, hidden_size), seed=44)

        reference_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (rmsnorm_output, gate_weight, up_weight, down_weight)
        ]
        reference_output, _, _, _ = _reference(*reference_inputs)
        reference_output.backward(grad_output)

        local_grad_output = grad_output[token_start:token_end].contiguous()
        actual_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (
                rmsnorm_output[token_start:token_end].contiguous(),
                gate_weight[feature_start:feature_end].contiguous(),
                up_weight[feature_start:feature_end].contiguous(),
                down_weight[:, feature_start:feature_end].contiguous(),
            )
        ]
        actual_output = qwen3_ffn(
            *actual_inputs,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        actual_output.backward(local_grad_output)

        expected_grads = (
            reference_inputs[0].grad[token_start:token_end],
            reference_inputs[1].grad[feature_start:feature_end],
            reference_inputs[2].grad[feature_start:feature_end],
            reference_inputs[3].grad[:, feature_start:feature_end],
        )
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "max_errors": [
                    float(
                        (actual_output - reference_output[token_start:token_end]).abs().max().item()
                    )
                ]
                + [
                    float((actual.grad - expected).abs().max().item())
                    for actual, expected in zip(actual_inputs, expected_grads, strict=True)
                ],
            }
        )
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_qwen3_8b_dimensions_are_pinned():
    assert QWEN3_8B_HIDDEN_SIZE == 4096
    assert QWEN3_8B_INTERMEDIATE_SIZE == 12288


def test_backward_matches_autograd_reference(monkeypatch):
    stub = _TorchKernelStub()
    monkeypatch.setattr(ffn_module, "_C", stub)
    monkeypatch.setattr(ffn_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(ffn_module, "_validate_ffn_inputs", lambda *args: None)

    hidden = _randn((2, 3, 8), seed=0)
    gate_weight = _randn((12, 8), seed=1)
    up_weight = _randn((12, 8), seed=2)
    down_weight = _randn((8, 12), seed=3)
    grad_output = _randn(hidden.shape, seed=4)

    ref_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected, _, _, _ = _reference(*ref_inputs)
    expected.backward(grad_output)

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected.detach())
    for actual_input, reference in zip(actual_inputs, ref_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, reference.grad)

    assert stub.calls.count("det_gemm_fwd") == 6
    assert stub.calls.count("det_gemm_db") == 3
    assert stub.calls.count("swiglu_forward") == 1
    assert stub.calls.count("swiglu_backward") == 1


@requires_gloo
def test_tensor_parallel_backward_matches_full_reference_cpu_gloo_2_ranks():
    ctx = mp.get_context("spawn")
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "gloo_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_tp_ffn_backward_gloo_worker,
                args=(rank, world_size, init_method, result_queue),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()

        results = []
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=45))
        except queue.Empty:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail("timed out waiting for tensor-parallel Gloo workers")
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()

    for result in sorted(results, key=lambda item: item["rank"]):
        assert result["ok"], result.get("traceback")
        assert max(result["max_errors"]) < 1e-6
    for process in processes:
        assert process.exitcode == 0


@requires_gloo
def test_tensor_context_parallel_backward_matches_reference_cpu_gloo_4_ranks():
    ctx = mp.get_context("spawn")
    world_size = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "gloo_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_tp_cp_ffn_backward_gloo_worker,
                args=(rank, world_size, init_method, result_queue),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()

        results = []
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=45))
        except queue.Empty:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail("timed out waiting for TP+CP Gloo workers")
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()

    for result in sorted(results, key=lambda item: item["rank"]):
        assert result["ok"], result.get("traceback")
        assert max(result["max_errors"]) < 1e-6
    for process in processes:
        assert process.exitcode == 0


def test_rejects_non_huggingface_weight_layout():
    hidden = torch.empty((2, 8), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 12), dtype=torch.bfloat16)
    up_weight = torch.empty((12, 8), dtype=torch.bfloat16)
    down_weight = torch.empty((8, 12), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="gate_weight must have shape"):
        qwen3_ffn(
            hidden,
            gate_weight,
            up_weight,
            down_weight,
        )


@requires_cuda_ffn
def test_cuda_forward_backward_matches_fp32_reference():
    hidden = _randn((2, 3, 64), seed=10, device="cuda", dtype=torch.bfloat16)
    gate_weight = _randn((128, 64), seed=11, device="cuda", dtype=torch.bfloat16)
    up_weight = _randn((128, 64), seed=12, device="cuda", dtype=torch.bfloat16)
    down_weight = _randn((64, 128), seed=13, device="cuda", dtype=torch.bfloat16)
    grad_output = _randn(hidden.shape, seed=14, device="cuda", dtype=torch.bfloat16)

    ref_inputs = [
        value.detach().cpu().float().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected, _, _, _ = _reference(*ref_inputs)
    expected.backward(grad_output.cpu().float())

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs)
    actual.backward(grad_output)

    torch.testing.assert_close(
        actual.cpu().float(),
        expected.detach(),
        atol=5e-2,
        rtol=2e-2,
    )
    for actual_input, reference in zip(actual_inputs, ref_inputs, strict=True):
        torch.testing.assert_close(
            actual_input.grad.cpu().float(),
            reference.grad,
            atol=5e-2,
            rtol=2e-2,
        )


@requires_cuda_ffn
def test_cuda_forward_and_input_gradient_are_batch_invariant():
    gate_weight = _randn((128, 64), seed=20, device="cuda", dtype=torch.bfloat16)
    up_weight = _randn((128, 64), seed=21, device="cuda", dtype=torch.bfloat16)
    down_weight = _randn((64, 128), seed=22, device="cuda", dtype=torch.bfloat16)
    hidden = _randn((6, 64), seed=23, device="cuda", dtype=torch.bfloat16)
    grad_output = _randn(hidden.shape, seed=24, device="cuda", dtype=torch.bfloat16)

    full_hidden = hidden.detach().clone().requires_grad_(True)
    full_output = qwen3_ffn(full_hidden, gate_weight, up_weight, down_weight)
    full_output.backward(grad_output)

    slice_hidden = hidden[2:4].detach().clone().requires_grad_(True)
    slice_output = qwen3_ffn(slice_hidden, gate_weight, up_weight, down_weight)
    slice_output.backward(grad_output[2:4])

    assert torch.equal(slice_output, full_output[2:4])
    slice_grad_hidden = slice_hidden.grad
    full_grad_hidden = full_hidden.grad
    assert torch.equal(slice_grad_hidden, full_grad_hidden[2:4])
