import contextlib
import math
import os
from collections.abc import Callable, Iterable
from typing import cast

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from loguru import logger
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    _cp_options,
    set_rotate_method,
)
from torch.distributed.tensor.parallel import loss_parallel

from torchfeather.config import TORCH_DTYPE_MAP
from torchfeather.distributed.parallel_dims import ParallelDims
from torchfeather.tools.device_utils import device_module


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None,
) -> int | float:
    if isinstance(x, DTensor):
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    assert x.numel() == 1
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> int | float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_sum(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> int | float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_mean(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg
    )


def set_determinism(
    parallel_dims: ParallelDims | None,
    device: torch.device,
    seed: int | None = None,
    deterministic: bool = False,
    distinct_seed_mesh_dims: list[str] | None = None,
) -> None:
    if distinct_seed_mesh_dims is None:
        distinct_seed_mesh_dims = ["pp"]

    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if seed is None:
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = cast(int, seed_tensor.to("cpu").view(torch.uint64).item())

    if parallel_dims is not None and c10d.get_world_size() > 1:
        distinct_meshes = [
            mesh
            for dim in distinct_seed_mesh_dims
            if (mesh := parallel_dims.get_optional_mesh(dim)) is not None
        ]

        if distinct_meshes:
            seed_offset = 0
            cumulative_size = 1
            for distinct_mesh in distinct_meshes:
                local_rank = distinct_mesh.get_local_rank()
                seed_offset += local_rank * cumulative_size
                cumulative_size *= distinct_mesh.size()

            seed += seed_offset
            seed %= 2**64

            logger.debug(
                f"Distinct dims {distinct_seed_mesh_dims}, Global rank {c10d.get_rank()} using seed: {seed}"
            )
    else:
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str, # whether to do rotate of k,v from other ranks or do all-gather k,v with current rank's q block
    cp_load_balance: bool = True, # because each rank only do their local mask based on the length of local q block so the computation cost is different, can ask rank 1 do q block 1, 3; rank 2 do q block 2 etc. for load balancing
):
    set_rotate_method(cp_rotate_method)
    _cp_options.enable_load_balance = cp_load_balance
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(
    enable_loss_parallel: bool, enable_compiled_autograd: bool
) -> Callable[..., contextlib.AbstractContextManager]:
    @contextlib.contextmanager
    def context(cp_context: contextlib.AbstractContextManager | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context:
                stack.enter_context(cp_context)

            yield

    return context


def maybe_enable_amp(
    parallel_dims: ParallelDims, mixed_precision_param: str, device_type: str
) -> contextlib.AbstractContextManager:
    if parallel_dims.fsdp_enabled:
        logger.info("Mixed precision training is handled by fully_shard")
        return contextlib.nullcontext()
    else:
        if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
            logger.warning(
                "Mixed precision training with TP or PP is only supported when FSDP/HSDP/CP is enabled."
            )
            logger.info("Mixed precision training is disabled")
        return contextlib.nullcontext()
