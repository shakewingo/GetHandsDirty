import os

from torchfeather.config.job_config import JobConfig
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.moe.moe import MoEArgs


def get_deepseek_v3_model_args() -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="sigmoid",
            route_norm=True,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        moe_enabled=True
    )


def get_torchfeather_1b_model_args() -> DeepSeekV3ModelArgs:
    """1.38B total / 0.41B active MoE -- the model Phase 3 actually trains.

    Sized so the whole thing fits comfortably on 8x24GB with room for activations, while still
    exercising every parallelism dimension: 32 experts divide by ep in {2,4,8}, 8 heads divide
    by tp=2, and `dim` / `moe_inter_dim` are multiples of 8 so `torch._grouped_mm` gets the
    16-byte-aligned strides it requires.
    """
    return DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=1024,
        inter_dim=2816,
        moe_inter_dim=704,
        n_layers=16,
        n_dense_layers=1,
        n_heads=8,
        moe_args=MoEArgs(
            num_experts=32,
            num_shared_experts=1,
            top_k=4,
            score_func="sigmoid",
            route_norm=True,
            score_before_experts=False,
            load_balance_coeff=1e-3,
        ),
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        max_seq_len=2048,
        original_seq_len=4096,
        mscale=1.0,
        max_batch_size=1,      # the KV cache is unused in training; do not allocate it
        attn_impl="naive",     # 3-4x cheaper attention than absorb, and the only SDPA path
        moe_enabled=True,
    )


def get_torchfeather_1b_base_config() -> JobConfig:
    """Shared base for the Phase 3 runs. Parallelism degrees are set by the callers below."""
    config = JobConfig()

    config.job.dump_folder = "./outputs"

    config.profiling.enable_profiling = False
    config.profiling.profile_freq = 100

    config.metrics.log_freq = 10

    config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"
    config.model.args = get_torchfeather_1b_model_args()

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 4e-4
    config.optimizer.eps = 1e-8

    config.lr_scheduler.warmup_steps = 300
    config.lr_scheduler.decay_ratio = None
    config.lr_scheduler.decay_type = "cosine"
    config.lr_scheduler.min_lr_factor = 0.1

    config.training.dataset = "fineweb_edu"
    config.training.seq_len = 2048
    config.training.local_batch_size = 8
    config.training.global_batch_size = 64      # 64 * 2048 = 131,072 tokens per optimizer step
    config.training.max_norm = 1.0
    config.training.steps = 10000
    config.training.dataloader_num_workers = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 8
    config.parallelism.expert_tensor_parallel_degree = 1

    config.checkpoint.enable = True
    config.checkpoint.interval = 150
    config.checkpoint.keep_latest_k = 3         # ~16.6 GiB each -- 10 would be 166 GiB
    config.checkpoint.async_mode = "async"
    # F7: an async save killed mid-write leaves step-N without .metadata, so the loader
    # skips it. keep_latest_k >= 2 makes that survivable by falling back to the previous
    # complete checkpoint -- but before the first one exists there is nothing to fall back
    # to, and a kill in that window restarts from zero. Saving at step 1 closes it.
    config.checkpoint.enable_first_step_checkpoint = True

    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "op"

    # Data-dependent shapes in the MoE make this expensive to compile and cheap to get wrong.
    # Turn it on as a measured experiment (Stage 2), not as a default.
    config.compile.enable = False

    return config


def get_torchfeather_1b_smoke_config() -> JobConfig:
    """Stage 1: one GPU, 100 steps. Everything here exists to be measured, not trained."""
    config = get_torchfeather_1b_base_config()
    config.model.args.n_layers = 4
    config.training.local_batch_size = 2
    config.training.global_batch_size = 2
    config.training.steps = 100
    config.metrics.log_freq = 1
    config.checkpoint.interval = 50
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.expert_parallel_degree = 1
    return config


def get_torchfeather_1b_run_config() -> JobConfig:
    """Stage 3: the real run. 8 GPUs, FSDP + EP, tp=1 so `attn_impl="naive"` is safe."""
    return get_torchfeather_1b_base_config()


def get_deepseek_v3_base_config() -> JobConfig:
    config = JobConfig()

    config.job.dump_folder = "./outputs"

    config.profiling.enable_profiling = False
    config.profiling.save_traces_folder = "profile_trace"
    config.profiling.profile_freq = 10
    config.profiling.enable_memory_snapshot = False
    config.profiling.save_memory_snapshot_folder = "memory_snapshot"

    config.metrics.log_freq = 1
    config.metrics.save_folder = "metrics"

    config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"
    config.model.args = get_deepseek_v3_model_args()

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.2e-4
    config.optimizer.eps = 1e-8

    config.lr_scheduler.warmup_steps = 50
    config.lr_scheduler.decay_ratio = 0.8
    config.lr_scheduler.decay_type = "cosine"
    config.lr_scheduler.min_lr_factor = 0.1

    config.training.local_batch_size = 5
    config.training.seq_len = 4096
    config.training.max_norm = 1.0
    config.training.steps = 500
    config.training.dataset = "fineweb"

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.fsdp_reshard_after_forward = "default"
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1

    config.checkpoint.enable = True
    config.checkpoint.folder = "checkpoint"
    config.checkpoint.interval = 100
    config.checkpoint.async_mode = "async"

    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "op"

    config.compile.enable = True
    config.compile.components = ["model", "loss"]

    return config


def get_deepseek_v3_pp_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 4

    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    return config


def get_deepseek_v3_hsdp_ep_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1

    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = 8
    return config


def get_deepseek_v3_fsdp_cp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.context_parallel_degree = 2
    return config


def get_deepseek_v3_fsdp_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    # Reduce number of layers and batch size to fit in memory with DDP
    config.model.args.n_layers = 6
    config.training.local_batch_size = 8

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 8
    return config


def get_deepseek_v3_hsdp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 8
    return config


def get_deepseek_v3_ddp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()

    # Reduce number of layers and batch size to fit in memory with DDP
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1

    config.parallelism.data_parallel_replicate_degree = 16
    config.parallelism.data_parallel_shard_degree = 1
    return config


def get_deepseek_v3_fsdp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 16
    return config


def get_deepseek_v3_fsdp_ep_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 1
    return config


def get_deepseek_v3_fsdp_ep_etp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 2
    return config




def get_torchfeather_1b_reshard_config() -> JobConfig:
    """Checkpoint-resharding probe. One config name, so both runs share a dump folder.

    EP degree comes from TF_RESHARD_EP so the same checkpoint can be written at one
    expert-parallel degree and read back at another -- the check that catches a shard
    dim declared wrong, which DCP otherwise resolves silently and wrongly.
    """
    config = get_torchfeather_1b_base_config()
    config.training.steps = int(os.environ.get("TF_RESHARD_STEPS", "10"))
    config.training.seed = 1234
    config.metrics.log_freq = 1
    config.checkpoint.enable = True
    config.checkpoint.interval = 5
    config.checkpoint.enable_first_step_checkpoint = False
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = int(os.environ.get("TF_RESHARD_EP", "2"))
    return config

# Stage 2 parallelism matrix. Seven world_size=8 configs, same seed and same global
# batch, so their loss curves are directly comparable against config 1. Each is 50
# steps with checkpointing off -- these exist to exercise the plans, not to train.
_STAGE2_MATRIX = {
    #        dp_shard, tp, ep, etp, pp   exercises
    "m1": (8, 1, 1, 1, 1),   # FSDP baseline, no expert plan
    "m2": (8, 1, 2, 1, 1),   # ExpertParallel + EFSDP wrapping
    "m3": (8, 1, 8, 1, 1),   # ep == world; the Shard(1) branch in apply_fsdp
    "m4": (4, 2, 1, 1, 1),   # TensorParallel expert plan, no EP
    "m5": (4, 2, 4, 1, 1),   # EP borrows TP -> ReordererSequenceParallel
    "m6": (4, 2, 2, 2, 1),   # ExpertTensorParallel
    "m7": (2, 2, 2, 1, 2),   # PP + EP
}


def _make_stage2_config(key: str):
    dp_shard, tp, ep, etp, pp = _STAGE2_MATRIX[key]

    def _build() -> JobConfig:
        config = get_torchfeather_1b_base_config()
        config.training.steps = 50
        config.training.seed = 1234
        config.training.deterministic = False
        config.metrics.log_freq = 5
        # Same global batch on every row, so the curves are comparable. local_batch_size
        # is chosen per dp degree; gradient accumulation absorbs the difference.
        config.training.global_batch_size = 64
        config.training.local_batch_size = 8
        config.checkpoint.enable = False

        config.parallelism.data_parallel_replicate_degree = 1
        config.parallelism.data_parallel_shard_degree = dp_shard
        config.parallelism.tensor_parallel_degree = tp
        config.parallelism.expert_parallel_degree = ep
        config.parallelism.expert_tensor_parallel_degree = etp
        config.parallelism.pipeline_parallel_degree = pp
        return config

    return _build


config_map = {
    "tf1b_smoke": get_torchfeather_1b_smoke_config,
    "tf1b_reshard": get_torchfeather_1b_reshard_config,
    "tf1b_m1": _make_stage2_config("m1"),
    "tf1b_m2": _make_stage2_config("m2"),
    "tf1b_m3": _make_stage2_config("m3"),
    "tf1b_m4": _make_stage2_config("m4"),
    "tf1b_m5": _make_stage2_config("m5"),
    "tf1b_m6": _make_stage2_config("m6"),
    "tf1b_m7": _make_stage2_config("m7"),
    "tf1b_run": get_torchfeather_1b_run_config,
    "hsdp": get_deepseek_v3_hsdp_config,
    "ddp": get_deepseek_v3_ddp_config,
    "fsdp": get_deepseek_v3_fsdp_config,
    "pp_tp": get_deepseek_v3_pp_tp_config,
    "fsdp_tp": get_deepseek_v3_fsdp_tp_config,
    "fsdp_cp": get_deepseek_v3_fsdp_cp_config,
    "hsdp_ep": get_deepseek_v3_hsdp_ep_config,
    "fsdp_ep_tp": get_deepseek_v3_fsdp_ep_tp_config,
    "fsdp_ep_etp": get_deepseek_v3_fsdp_ep_etp_config,
}


def get_config(name: str) -> JobConfig:
    return config_map[name]()