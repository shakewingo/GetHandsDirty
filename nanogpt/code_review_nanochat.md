
# From Karpathy's nanochat repo
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CORE COMPONENTS ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

nanochat/gpt.py
    └─► GPT model (Transformer)
        ├─► CausalSelfAttention (with KV cache support)
        ├─► MLP (ReLU² activation)
        ├─► Rotary embeddings (RoPE)
        └─► QK normalization

nanochat/engine.py
    └─► Inference engine
        ├─► KVCache: Maintains KV cache across tokens
        ├─► generate(): Token-by-token generation loop
        └─► Supports tool use (Python execution)

nanochat/tokenizer.py
    └─► RustBPETokenizer
        ├─► encode/decode: Basic tokenization
        ├─► render_conversation(): Converts chat format → token IDs + mask
        └─► Special tokens for conversations

nanochat/checkpoint_manager.py
    └─► Save/load model checkpoints
        ├─► save_checkpoint(): Model + optimizer + metadata
        ├─► load_checkpoint(): Resume training
        └─► load_model(): Load for inference/eval

tasks/*.py
    └─► Task classes (Task base class)
        ├─► Each task implements __getitem__() → conversation dict
        ├─► TaskMixture: Combines multiple tasks
        └─► Used by mid_train.py and chat_sft.py
```
```
Focus Areas for Core Training/Inference:
1. nanochat/gpt.py — Transformer architecture
2. scripts/base_train.py — Pretraining loop
3. scripts/mid_train.py — Midtraining loop
4. scripts/chat_sft.py — SFT loop
5. nanochat/engine.py — Inference with KV cache
6. nanochat/dataloader.py — Data loading and batching
```
## `nonachat/gpt.py`
### Difference from `bigram_v1.py`
1. Fixed position embedding obtained from sin/cos -> RoPE
    - the simpl 2D rotation formula is:
        ```
        x1′​​=x1​cosθ−x2​sinθ
        x2′=x1​sinθ+x2​cosθ
        ```
2. kv cache supported
3. Temperature considered in inference
    - softmax makes larger scaled value even largerm, so if logits = logits/temperature (0-1), smaller temperature -> larger logits -> larger prob that highly chance to be selected each time -> output is less variant

## `nanochat/engine.py`
### Difference from generator() in `gpt.py`
1. Basically same thing but considered kv cache in inferece, kv_cache_prefill
2. Add RowState, num of tokens etc.
3. Force inject python code execution's output as tokens into model etc. 
    ```
    Model samples: <|python_start|>2+2<|python_end|>
                        ↓
    System evaluates: 2+2 = 4
                        ↓
    System forces: <|output_start|>4<|output_end|>
                        ↓
    Model continues sampling from there...
    ```
4. Batch generation

## `nanochat/tokenizer.py`
1. Use RustBPE to train tokenizer in traning and tiktoken in inference, HF's tokenizer can do both training and inference but the author found it's confusing

## `script/base_train.py`
1. In code, it has "once in a while: sample from the model (only on master process)" to generate result, why sampling is needed?

## `script/mid_train.py`
1. Built from `base_train.py`, the base dataset is "FineWeb-Edu 100BT dataset" and then in `mid_train` it uses benchmark datasets to further train, and more importantly, takes care of structed conversion(e.g. conversation_rendering), tool usage etc. 
2.`base_train` only learns how to speak NL but `mid_train` foucs on how to answer specific questions for nanochat

## `script/chat_sft.py`
1. Built from `mid_train.py`
    ```
    Input: Same conversation format, but higher quality/curated

    Format: Same special tokens, but with **loss masking (only train on assistant tokens)**

    Goal: Refine conversation quality, reduce errors, improve helpfulness
    Learning Rate: init_lr_frac = 0.02 (2% of base - very gentle)

    Data: 23K rows (smaller, curated dataset)

    Key Difference: Uses loss masking - only computes loss on tokens the assistant should generate
    ```
## `script/chat_rl.py`
1. Simple GRPO version

# From Karpathy's video "Let's build GPT2"
1. Speed up: cpu -> gpu
    - add autocast which changes partial training components from float32 to float16 (see [CUDA Op-Specific Behavior](https://docs.pytorch.org/docs/stable/amp.html#cuda-op-specific-behavior))
    - torch.compile(model), 2X speed up
    - FlashAttention
    - urgly number -> nice number (2's power) then will utilize GPU more (block tiles is 2's power)

2. Cosine learning rate is used in GPT2/GPT3
3. In terms of  weight decay, one thumb of rule is that usually do not do weight-decay for 1-dim params such as bias, layer normalization  but do it for matrix manipulation
4. Gradient accumulation 
    - want to simulate the same batch size used by GPT3 as other params is already referred from it and batch_size has high relationship with other params but because the GPU we used is small, can't just copy and paste, so comes up with gradient accumulation
    - basically just need to loop current batch size! But be careful that loss per iteration is a sum of current batch size so need to be divided by gradient_accumulation_steps
5.  Distributed Data Parallel
    - use `torchrun`
    - need to change `compute_init` for ddp; DataLoader w `process_rank`, `num_processes`
6. Dataset
    - HF provides a high quality dataset: FineWeb-Edu, which filters common crawl (the dataset used by GPT2 and GPT3) and the filter is used llamba 7B to judge which content is educational
7. How to do validation, login, visualize losses
    - Karpathy uses 'helloswag' dataset to compare model's performance with GPT2
    - During evaluation, he cares about why trainging curve has deeper drop and then goes back later, guess is that based on poor sharding..
    - Add checkpoint after 5000 steps, e.g. `model.state_dict()` / `optimizer.state_dict()`
    - If want to build a chat-model on top of that, just need to fine-tune based on more convensational format data, e.g. user: XXX, system: XXX
