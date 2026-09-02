# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.profiler import ProfilerActivity, profile, record_function

"""
Workflow for pp=2, dp=2
I'll use m = 2 microbatches so the table fits (2m + 3 = 7 slots); with the file's real constants m = 4 and the middle rows just repeat. Shapes are (2, 8) = (MICRO_BATCH_SIZE, HIDDEN_SIZE).

t	rank 0 / rank 1 (stage 0)	rank 2 / rank 3 (stage 1)	tensor crossing
0	FWD(0), ISEND_F(0)	IRECV_F(0)	a₀ → (tag 100)
1	FWD(1), WAIT_SEND_F(0), ISEND_F(1)	WAIT_RECV_F(0), IRECV_F(1), FWD(0)	a₁ → (tag 101)
2	WAIT_SEND_F(1) — idle (bubble)	WAIT_RECV_F(1), FWD(1)	
3	IRECV_B(1) — idle (bubble)	BWD(1), ISEND_B(1)	← ∂L₁/∂a₁ (tag 201)
4	WAIT_RECV_B(1), IRECV_B(0), BWD(1)	BWD(0), WAIT_SEND_B(1), ISEND_B(0)	← ∂L₀/∂a₀ (tag 200)
5	WAIT_RECV_B(0), BWD(0)	WAIT_SEND_B(0)	
6	ALL_REDUCE_GRADS over {0, 1}	ALL_REDUCE_GRADS over {2, 3}	dp sync
The FWD(i) cells in bold are where your selected code runs. Tracing microbatch 0 through the seam concretely:

t0  rank 0: x₀ → block0 → a₀        ; isend(a₀.detach(), dst=2, tag=100)
t1  rank 2: buf₀ filled with a₀     ; buf₀.requires_grad_()   ← the line you asked about
            block1(buf₀) → out₀     ; loss₀ = mse(out₀, y₀)/2
t4  rank 2: loss₀.backward()        → block1.grad += … ; buf₀.grad = ∂L₀/∂a₀
            isend(buf₀.grad.detach(), dst=0, tag=200)
t5  rank 0: a₀.backward(g₀)         → block0.grad += …
t6  ranks 0,1: all_reduce(block0.grad) / 2   ; ranks 2,3: all_reduce(block1.grad) / 2

"""

LOCAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 2
PP_SIZE = 2
DP_REPLICATE_SIZE = 1
EXPECTED_WORLD_SIZE = PP_SIZE * DP_REPLICATE_SIZE
MODEL_SEED = 0
DATA_SEED = 100
FORWARD_TAG_BASE = 100
BACKWARD_TAG_BASE = 200
HIDDEN_SIZE = 8
TRACE_DIR = Path("./traces")


class ActionKind(Enum):
    IRECV_FORWARD = auto()
    WAIT_RECV_FORWARD = auto()
    FORWARD = auto()
    ISEND_FORWARD = auto()
    WAIT_SEND_FORWARD = auto()
    IRECV_BACKWARD = auto()
    WAIT_RECV_BACKWARD = auto()
    BACKWARD = auto()
    ISEND_BACKWARD = auto()
    WAIT_SEND_BACKWARD = auto()
    ALL_REDUCE_GRADS = auto()

@dataclass(frozen=True)
class Action:
    kind: ActionKind
    microbatch: int | None = None

dp_coordinate = 1
generator = torch.Generator(device="cpu").manual_seed(
            DATA_SEED + dp_coordinate
        )
inputs = torch.randn(
            LOCAL_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
        )
targets = torch.randn(
            LOCAL_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
        )

# %%
inputs

# %%
class RepeatedDataLoader:
    def __init__(self, dp_coordinate: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(
            DATA_SEED + dp_coordinate
        )
        self.inputs = torch.randn(
            LOCAL_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
        )
        self.targets = torch.randn(
            LOCAL_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
        )

    def __iter__(self) -> RepeatedDataLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs.clone(), self.targets.clone()

# %%
class MLPBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear(x))
        return x

# %%
def make_block(pp_coordinate: int, device: torch.device) -> MLPBlock:
    # Built under the CPU RNG and then moved, so every rank holding the same
    # pipeline stage starts from identical weights whatever device it lands on.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED + pp_coordinate)
        return MLPBlock().to(device)

# %%
def build_stage_zero_schedule(num_microbatches: int) -> list[list[Action]]:
    schedule = [[] for _ in range(2 * num_microbatches + PP_SIZE + 1)] # each element is the list of action done per timestamp, the extra part PP_SIZE + 1 is the extra waiting time plus all_reduce_grads

    for microbatch in range(num_microbatches):
        actions = [Action(ActionKind.FORWARD, microbatch)]
        if microbatch > 0: 
            actions.append(Action(ActionKind.WAIT_SEND_FORWARD, microbatch - 1))
        actions.append(Action(ActionKind.ISEND_FORWARD, microbatch))
        schedule[microbatch].extend(actions)

    # extra timestamp for last microbatch's forward pass to send to next stage
    schedule[num_microbatches].append(
        Action(ActionKind.WAIT_SEND_FORWARD, num_microbatches - 1)
    )
    
    # extra timestamp for first microbatch's backward pass to be received from next stage
    first_backward = num_microbatches + 1
    schedule[first_backward].append(
        Action(ActionKind.IRECV_BACKWARD, num_microbatches - 1)
    )
    for offset, microbatch in enumerate(reversed(range(num_microbatches))):
        timestamp = first_backward + 1 + offset
        schedule[timestamp].append(
            Action(ActionKind.WAIT_RECV_BACKWARD, microbatch)
        )
        if microbatch > 0:
            schedule[timestamp].append(
                Action(ActionKind.IRECV_BACKWARD, microbatch - 1)
            )
        schedule[timestamp].append(Action(ActionKind.BACKWARD, microbatch))

    # extra timestamp for all-reduce
    schedule[-1].append(Action(ActionKind.ALL_REDUCE_GRADS))
    return schedule



# %%
def build_stage_one_schedule(num_microbatches: int) -> list[list[Action]]:
    schedule = [[] for _ in range(2 * num_microbatches + PP_SIZE + 1)] # each element is the list of action done per timestamp, the extra part PP_SIZE + 1 is the extra waiting time plus all_reduce_grads
    schedule[0].append(Action(ActionKind.IRECV_FORWARD, 0))

    for microbatch in range(num_microbatches):
        timestamp = microbatch + 1
        schedule[timestamp].append(
            Action(ActionKind.WAIT_RECV_FORWARD, microbatch)
        )
        if microbatch + 1 < num_microbatches:
            schedule[timestamp].append(
                Action(ActionKind.IRECV_FORWARD, microbatch + 1)
            )
        schedule[timestamp].append(Action(ActionKind.FORWARD, microbatch))

    first_backward = num_microbatches + 1
    for offset, microbatch in enumerate(reversed(range(num_microbatches))):
        timestamp = first_backward + offset
        schedule[timestamp].append(Action(ActionKind.BACKWARD, microbatch))
        if offset > 0:
            schedule[timestamp].append(
                Action(ActionKind.WAIT_SEND_BACKWARD, microbatch + 1)
            )
        schedule[timestamp].append(Action(ActionKind.ISEND_BACKWARD, microbatch))
    schedule[-2].append(Action(ActionKind.WAIT_SEND_BACKWARD, 0))
    schedule[-1].append(Action(ActionKind.ALL_REDUCE_GRADS))
    return schedule

# %%
def build_schedule(
    pp_coordinate: int,
    num_microbatches: int,
) -> list[list[Action]]:
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    if pp_coordinate == 0:
        return build_stage_zero_schedule(num_microbatches)
    if pp_coordinate == 1:
        return build_stage_one_schedule(num_microbatches)
    raise ValueError(f"expected pipeline coordinate 0 or 1, got {pp_coordinate}")


# %%
def format_schedule(rank: int, schedule: list[list[Action]]) -> str:
    lines = [f"[rank {rank}] schedule:"]
    for timestamp, actions in enumerate(schedule):
        descriptions = [
            action.kind.name
            if action.microbatch is None
            else f"{action.kind.name}({action.microbatch})"
            for action in actions
        ]
        lines.append(f"  t={timestamp:02d}: {', '.join(descriptions)}")
    return "\n".join(lines)


def require_work(work: dist.Work | None, name: str) -> dist.Work:
    if work is None:
        raise RuntimeError(f"{name} did not return a work handle")
    return work


def require_gradient(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.grad is None:
        raise RuntimeError(f"missing gradient for {name}")
    return tensor.grad

def average_gradients(module: nn.Module, group: dist.ProcessGroup) -> None:
    for parameter in module.parameters():
        gradient = require_gradient(parameter, "parameter")
        dist.all_reduce(gradient, group=group)
        gradient.div_(DP_REPLICATE_SIZE)

def peer_rank(mesh: DeviceMesh, pp_coordinate: int, dp_coordinate: int) -> int:
    return int(mesh.mesh[1 - pp_coordinate, dp_coordinate].item())

# %%
def run_reference(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pp_coordinate: int,
    dp_group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, MLPBlock]:
    first_block = make_block(0, device)
    second_block = make_block(1, device)
    output = second_block(first_block(inputs))
    loss = F.mse_loss(output, targets)
    loss.backward()
    selected_block = first_block if pp_coordinate == 0 else second_block
    average_gradients(selected_block, dp_group)
    return output, loss, selected_block

def assert_parameter_gradients_match(
    block: MLPBlock,
    reference_block: MLPBlock,
) -> None:
    for (name, parameter), (reference_name, reference_parameter) in zip(
        block.named_parameters(),
        reference_block.named_parameters(),
        strict=True,
    ):
        assert name == reference_name, (name, reference_name)
        torch.testing.assert_close(
            require_gradient(parameter, name),
            require_gradient(reference_parameter, reference_name),
        )

# %%
def run_schedule(pp_coordinate,
                pipeline_peer,
                dp_group,
                block,
                schedule,
                input_microbatches,
                target_microbatches,
                device):
    num_microbatches = len(input_microbatches)
    forward_outputs: dict[int, torch.Tensor] = {}
    received_activations: dict[int, torch.Tensor] = {}
    losses: dict[int, torch.Tensor] = {}

    forward_recv_buffers: dict[int, torch.Tensor] = {}
    forward_recv_requests: dict[int, dist.Work] = {}
    forward_send_buffers: dict[int, torch.Tensor] = {}
    forward_send_requests: dict[int, dist.Work] = {}
    backward_recv_buffers: dict[int, torch.Tensor] = {}
    backward_recv_requests: dict[int, dist.Work] = {}
    backward_send_buffers: dict[int, torch.Tensor] = {}
    backward_send_requests: dict[int, dist.Work] = {}

    # NCCL point-to-point ignores `tag` and matches peer messages in post order.
    # Both stages post their sends and receives in the same microbatch order, so
    # the tags below only do real matching work under gloo.
    for timestamp, actions in enumerate(schedule):
        for action in actions:
            microbatch = action.microbatch
            action_label = action.kind.name.lower()
            if microbatch is not None:
                action_label = f"{action_label}/microbatch_{microbatch}"
            with record_function(f"timestamp_{timestamp}/{action_label}"):
                # forward
                if action.kind is ActionKind.FORWARD:
                    assert microbatch is not None, action
                    if pp_coordinate == 0:
                        forward_outputs[microbatch] = block(
                            input_microbatches[microbatch]
                        )
                    else:
                        activation = forward_recv_buffers[microbatch] # get output from peer pipeline rank
                        activation.requires_grad_() # flag requires_grad=True
                        received_activations[microbatch] = activation
                        output = block(activation)
                        forward_outputs[microbatch] = output
                        losses[microbatch] = F.mse_loss(
                            output,
                            target_microbatches[microbatch],
                        ) / num_microbatches # per pp rankd and dp rank's loss that across all microbatches, i.e. a complete batch
                elif action.kind is ActionKind.ISEND_FORWARD:
                    assert microbatch is not None, action
                    buffer = forward_outputs[microbatch].detach()
                    forward_send_buffers[microbatch] = buffer
                    forward_send_requests[microbatch] = require_work(
                        dist.isend(
                            buffer,
                            dst=pipeline_peer,
                            tag=FORWARD_TAG_BASE + microbatch,
                        ),
                        "forward isend",
                    )
                elif action.kind is ActionKind.WAIT_SEND_FORWARD:
                    assert microbatch is not None, action
                    forward_send_requests.pop(microbatch).wait()
                    del forward_send_buffers[microbatch]
                elif action.kind is ActionKind.WAIT_RECV_FORWARD:
                    assert microbatch is not None, action
                    forward_recv_requests.pop(microbatch).wait()
                elif action.kind is ActionKind.IRECV_FORWARD:
                    assert microbatch is not None, action
                    buffer = torch.empty(MICRO_BATCH_SIZE, HIDDEN_SIZE, device=device)
                    forward_recv_buffers[microbatch] = buffer
                    forward_recv_requests[microbatch] = require_work(
                        dist.irecv(
                            buffer,
                            src=pipeline_peer,
                            tag=FORWARD_TAG_BASE + microbatch,
                        ),
                        "forward irecv",
                    )

                # backend
                elif action.kind is ActionKind.BACKWARD:
                    assert microbatch is not None, action
                    if pp_coordinate == 1:
                        losses[microbatch].backward()
                        backward_send_buffers[microbatch] = require_gradient(
                            received_activations[microbatch],
                            f"activation {microbatch}",
                        ).detach()
                    else:
                        forward_outputs[microbatch].backward(
                            backward_recv_buffers[microbatch]
                        )
                elif action.kind is ActionKind.ISEND_BACKWARD:
                    assert microbatch is not None, action
                    buffer = backward_send_buffers[microbatch]
                    backward_send_requests[microbatch] = require_work(
                        dist.isend(
                            buffer,
                            dst=pipeline_peer,
                            tag=BACKWARD_TAG_BASE + microbatch,
                        ),
                        "backward isend",
                    )
                elif action.kind is ActionKind.WAIT_SEND_BACKWARD:
                    assert microbatch is not None, action
                    backward_send_requests.pop(microbatch).wait()
                    del backward_send_buffers[microbatch]
                elif action.kind is ActionKind.WAIT_RECV_BACKWARD:
                    assert microbatch is not None, action
                    backward_recv_requests.pop(microbatch).wait()
                elif action.kind is ActionKind.IRECV_BACKWARD:
                    assert microbatch is not None, action
                    buffer = torch.empty(MICRO_BATCH_SIZE, HIDDEN_SIZE, device=device)
                    backward_recv_buffers[microbatch] = buffer
                    backward_recv_requests[microbatch] = require_work(
                        dist.irecv(
                            buffer,
                            src=pipeline_peer,
                            tag=BACKWARD_TAG_BASE + microbatch,
                        ),
                        "backward irecv",
                    )
                elif action.kind is ActionKind.ALL_REDUCE_GRADS:
                    average_gradients(block, dp_group) # per pp rank and dp rank's gradients across all microbatches, need to do all-reduce to get avg grads from partial grads
                else:
                    raise AssertionError(action)
    return forward_outputs, losses

# %%
def main() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "launch with: uv run torchrun --standalone "
            f"--nproc-per-node={EXPECTED_WORLD_SIZE} "
            "./minimal_examples/my_pp_gpipe.py (one GPU per rank)"
        )
    if LOCAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0:
        raise RuntimeError(
            "LOCAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"this example expects {EXPECTED_WORLD_SIZE} ranks, "
                f"got {world_size}"
            )

        mesh = init_device_mesh(
            "cuda",
            (PP_SIZE, DP_REPLICATE_SIZE),
            mesh_dim_names=("pp", "dp_replicate"),
        )
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError(f"rank {rank} is not part of the device mesh")
        pp_coordinate, dp_coordinate = coordinate
        print(f'pp_coordinate: {pp_coordinate}, dp_coordinate: {dp_coordinate}')
        pipeline_peer = peer_rank(mesh, pp_coordinate, dp_coordinate)
        print(f'pipeline_peer: {pipeline_peer}')
        dp_group = mesh.get_group("dp_replicate")
        print(f'dp_group: {dp_group}')

        inputs, targets = next(iter(RepeatedDataLoader(dp_coordinate)))
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_microbatches = inputs.split(MICRO_BATCH_SIZE)
        target_microbatches = targets.split(MICRO_BATCH_SIZE)
        num_microbatches = LOCAL_BATCH_SIZE // MICRO_BATCH_SIZE
        assert len(input_microbatches) == num_microbatches, (
            len(input_microbatches),
            num_microbatches,
        )
        assert len(target_microbatches) == num_microbatches, (
            len(target_microbatches),
            num_microbatches,
        )

        reference_output, reference_loss, reference_block = run_reference(
            inputs,
            targets,
            pp_coordinate,
            dp_group,
            device,
        )
        block = make_block(pp_coordinate, device)
        schedule = build_schedule(pp_coordinate, num_microbatches)
        print(
            f"[rank {rank}] coordinate={(pp_coordinate, dp_coordinate)} "
            f"peer={pipeline_peer}\n{format_schedule(rank, schedule)}",
            flush=True,
        )

        if rank == 0:
            TRACE_DIR.mkdir(exist_ok=True)
        # dist.barrier() is a collective synchronization point: every rank in the group must call it - indicate "everyone has reached this line"
        # in this case, every rank is waiting for a side effect (mkdir) on the filesystem did by rank 0
        dist.barrier(device_ids=[local_rank])
        trace_path = TRACE_DIR / f"pp-gpipe-rank-{rank}.json"

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            forward_outputs, losses = run_schedule(
                pp_coordinate,
                pipeline_peer,
                dp_group,
                block,
                schedule,
                input_microbatches,
                target_microbatches,
                device,
            )

        profiler.export_chrome_trace(str(trace_path))
        assert_parameter_gradients_match(block, reference_block)

        if pp_coordinate == PP_SIZE - 1:
            pipeline_output = torch.cat(
                [forward_outputs[i].detach() for i in range(num_microbatches)]
            )
            pipeline_loss = torch.stack(
                [losses[i].detach() for i in range(num_microbatches)]
            ).sum()
            torch.testing.assert_close(pipeline_output, reference_output)
            torch.testing.assert_close(pipeline_loss, reference_loss)

        print(f"[rank {rank}] trace: {trace_path}", flush=True)
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            print(
                "PASS: GPipe outputs and averaged gradients match the reference",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            
if __name__ == "__main__":
    main()

# %%



