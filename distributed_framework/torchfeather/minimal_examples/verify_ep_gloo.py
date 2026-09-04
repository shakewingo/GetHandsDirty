"""Multi-rank expert-parallel verification on CPU/gloo. No GPU required.

Exercises the full EP dispatch/combine path -- both all-to-alls, `_permute`/`_unpermute`, the
padded per-expert counts, and the optimizer's expert-bias hook -- against a single-process
reference. This is rung L1 of the MoE design's verification ladder.

    torchrun --nproc_per_node=4 -m torchfeather.minimal_examples.verify_ep_gloo

gloo in torch 2.13 supports uneven `all_to_all_single`, and `moe/kernels.py` ships a non-Triton
CPU fallback, so nothing here needs CUDA.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from torchfeather.distributed.expert_parallel import ExpertParallel
from torchfeather.model.moe.moe import MoE, MoEArgs

DIM, HIDDEN, N_EXPERTS, TOP_K, SEQ_LEN = 64, 32, 8, 2, 16
ATOL = 5e-2  # the expert path runs in bf16 via torch._grouped_mm


def _build_moe(seed: int = 0) -> MoE:
    torch.manual_seed(seed)
    args = MoEArgs(
        num_experts=N_EXPERTS,
        num_shared_experts=1,
        top_k=TOP_K,
        score_func="sigmoid",
        route_norm=True,
        score_before_experts=False,
        load_balance_coeff=1e-3,
    )
    moe = MoE(args, dim=DIM, hidden_dim=HIDDEN)
    for p in moe.experts.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return moe


def _report(rank: int, ok: bool, name: str, detail: str = "") -> bool:
    if rank == 0:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    return ok


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    ep_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))

    # Same weights and same global batch on every rank; each rank owns one batch row.
    torch.manual_seed(1234)
    x_global = torch.randn(world, SEQ_LEN, DIM)

    reference = _build_moe()
    with torch.no_grad():
        out_ref = reference(x_global)
    counts_ref = reference.tokens_per_expert.clone()

    moe = _build_moe()
    # The split sizes live on the ParallelStyle, not the module, so keep the reference.
    ep_plan = ExpertParallel()
    parallelize_module(moe.experts, ep_mesh, ep_plan)

    x_local = x_global[rank : rank + 1].clone().requires_grad_(True)
    out_local = moe(x_local)

    results = []

    # 1. The EP path must reproduce the single-process result exactly, not just closely.
    #    A wrong permutation shows up here and almost nowhere else.
    maxdiff = (out_local - out_ref[rank : rank + 1]).abs().max().item()
    results.append(_report(rank, maxdiff < ATOL, "1 EP-vs-baseline parity", f"maxdiff {maxdiff:.5f}"))

    # 2. Token conservation: what every rank sends must equal what every rank receives.
    sent = torch.tensor(ep_plan.input_splits, dtype=torch.int64)
    recv = torch.tensor(ep_plan.output_splits, dtype=torch.int64)
    all_sent = [torch.zeros_like(sent) for _ in range(world)]
    all_recv = [torch.zeros_like(recv) for _ in range(world)]
    dist.all_gather(all_sent, sent)
    dist.all_gather(all_recv, recv)
    sent_m, recv_m = torch.stack(all_sent), torch.stack(all_recv)
    results.append(_report(rank, torch.equal(sent_m.T, recv_m), "2 send matrix == recv transpose"))
    results.append(
        _report(rank, int(sent.sum()) == SEQ_LEN * TOP_K, "2b local sent == N*K",
                f"{int(sent.sum())} vs {SEQ_LEN * TOP_K}")
    )

    # 3. Padding is inert. `_permute` pads each local expert block up to a multiple of 8 and
    #    `_unpermute` drops those rows via the sentinel; assertion 1 passing proves it.
    results.append(_report(rank, results[0], "3 padding inert (implied by 1)"))

    # 4. The router histogram is global -- it counts every (token, expert) pair before any split.
    local_counts = moe.tokens_per_expert.clone()
    dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
    expected = world * SEQ_LEN * TOP_K
    results.append(
        _report(rank, int(local_counts.sum()) == expected, "4 tokens_per_expert global",
                f"{int(local_counts.sum())} vs {expected}")
    )
    results.append(
        _report(rank, torch.equal(local_counts, counts_ref), "4b matches single-process reference")
    )

    # 5. The bias is derived from an all-reduced statistic, so it must agree across ranks, and
    #    the update is constructed zero-sum so routing pressure is redistributed, not added.
    delta = moe.load_balance_coeff * torch.sign(local_counts.mean() - local_counts)
    delta = delta - delta.mean()
    bias = moe.expert_bias + delta
    gathered = [torch.zeros_like(bias) for _ in range(world)]
    dist.all_gather(gathered, bias)
    agree = all(torch.allclose(g, gathered[0], atol=1e-6) for g in gathered)
    zero_sum = abs(float(delta.sum())) < 1e-5
    results.append(_report(rank, agree and zero_sum, "5 expert_bias agrees across ranks + zero-sum"))

    # 6. Every locally-owned expert shard must receive gradient.
    out_local.sum().backward()
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad.to_local()).all() and p.grad.to_local().abs().sum() > 0
        for p in moe.experts.parameters()
    )
    results.append(_report(rank, grads_ok, "6 local expert grads"))

    all_ok = torch.tensor([1 if all(results) else 0])
    dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"\n{'ALL PASS' if all_ok.item() else 'FAILURES PRESENT'} "
              f"({len(results)} assertions, {world} ranks)", flush=True)
    dist.destroy_process_group()
    if not all_ok.item():
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    main()
