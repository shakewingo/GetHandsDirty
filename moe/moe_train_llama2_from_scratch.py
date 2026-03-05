"""
Author: 
Date: 25/12/16
"""
import math
import random
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        result = self.weight * (hidden_states / (torch.rsqrt(torch.mean(hidden_states * hidden_states, dim=-1, keepdim=True) + self.variance_epsilon)))
        return result

# RoPOE
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    
    cos = cos.unsqueeze(unsqueeze_dim) # (1, seq_len, 1, dim)
    sin = sin.unsqueeze(unsqueeze_dim) # (1, seq_len, 1, dim)
   
    q_embed = (q*cos) + (rotate_half(q)*sin)  # (batch_size, seq_len, head_num, dim) * (1, seq_len, 1, dim) = (batch_size, seq_len, head_num, dim) 广播
    k_embed = (k*cos) + (rotate_half(k)*sin)  # (batch_size, seq_len, head_num, dim) * (1, seq_len, 1, dim) = = (batch_size, seq_len, head_num, dim) 广播
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    # Here is slight different than the slides, originally is [x1, x2, x3, x4,. ...] * [cos(m * theta_1), cos(m * theta_1), cos(m * theta_2), cos(m * theta_2), ..cos(m * theta_d/2)] 
    # + [-x2, x1, -x4, x3, ...] * [sin(m * theta_1), sin(m * theta_1), sin(m * theta_2), sin(m * theta_2), ..sin(m * theta_d/2)], which is equivalently to be 
    # [x1, x3, ..., x2, x4,. ...] * [cos(m * theta_1), cos(m * theta_2), ..., cos(m * theta_1), cos(m * theta_2), ..cos(m * theta_d/2)] 
    # + [-x2, -x4,..., x1, x3, ...] * [sin(m * theta_1), sin(m * theta_2), ..., sin(m * theta_1), sin(m * theta_2), ..sin(m * theta_d/2)]
    def __init__(self, dim, max_seq_len=2048):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2)
        t = torch.arange(max_seq_len).float().unsqueeze(1)  # (max_seq_len, 1)
        freqs = t @ inv_freq.unsqueeze(0)  #(max_seq_len, 1)*(1, dim/2) = (max_seq_len, dim/2), e.g. m * theta_i part in the slides
        freqs = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)  # (1, seq_len, dim)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)  # (1, seq_len, dim)
        return apply_rotate_pos_emb(q, k, cos, sin)
    
# Config
class Config(PretrainedConfig):
    model_type = "custom_gpt" # for later on: AutoConfig.register("custom_gpt", Config)

    def __init__(
        self,
        vocab_size=6400,
        hidden_size=512,
        n_layers = 8,
        num_attention_heads=16,
        num_key_value_heads = 8,
        flash_attn = False,
        attention_bias = False,
        max_seq_len = 512,
        intermediate_size = 2048,
        mlp_bias = False,
        dropout = 0.0,
        expert_num = 4,
        topk = 2,
        output_router_logits = True,
        aux_loss_coef = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef

def repeat_kv(hidden_states, num_key_value_groups):
    B, S, NUM_KV_H, H = hidden_states.shape # at this moment, the k/v has been linearly projected in consideration of num_key_value_heads
    if num_key_value_groups == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(B, S, NUM_KV_H, num_key_value_groups, H)
    return hidden_states.reshape(B, S, NUM_KV_H * num_key_value_groups, H)

# GPT architecture
class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.heads_dim = self.hidden_size // self.num_attention_heads # simpily using default ones instead of customized heads_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.dropout_prob = config.dropout
        self.flash_attn = self.config.flash_attn
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.dropout = nn.Dropout(self.dropout_prob) # simpily using the same value instead of distinguishing attention_dropout and residual_dropout 
        self.rotary_emb = RotaryEmbedding(self.heads_dim)
        
        # multi-group transform
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.heads_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.heads_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.heads_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.heads_dim, self.hidden_size, bias=config.attention_bias)
    
    def forward(self, hidden_states, use_kv_cache=False):
        B, S, H = hidden_states.shape # H: heads_dim
        if use_kv_cache and self.eval(): # model.eval() is used to freeze model in inference phase only
            if self.k_cache is None:
                q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            else:
                last_token = hidden_states[:, -1, :]
                q = torch.cat((torch.zeros_like(hidden_states[:, :-1, :]), self.q_proj(last_token)), dim=1)
                k = torch.cat((self.k_cache, self.k_proj(last_token)), dim=1)
                v = torch.cat((self.v_cache, self.v_proj(last_token)), dim=1)
                
                # update kv cache
                self.k_cache, self.v_cache = k, v
                self.register_buffer("k_cache", self.k_cache)
                self.register_buffer("v_cache", self.v_cache)
        else:
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q = q.view(B, S, self.num_attention_heads, self.heads_dim)
        k = k.view(B, S, self.num_key_value_heads, self.heads_dim)
        v = v.view(B, S, self.num_key_value_heads, self.heads_dim)
        # use RoPOE
        q, k = self.rotary_emb(q, k)
        # use repetitive k and v for multi-grouped q
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q = q.transpose(1, 2) # (B, NUM_H, S, H)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            # TODO: 照抄的
            # q*k转置，（b, self.num_heads, s, self.head_dim）* (b, self.num_heads, self.head_dim，s) = （b, self.num_heads, s, s）
            # q*k/sqrt(self.head_dim)*v  （b, self.num_heads, s, s）* (b, self.num_heads, s, self.head_dim) = b, self.num_heads, s, self.head_dim
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                    dropout_p=self.dropout_prob if self.training else 0.0, 
                                                    is_causal=self.is_causal)  # TODO: not using attn_mask may cause some issue
        else:
            # Create causal mask on the SAME device as q/k (fixes mps:0 vs cpu)
            # Shape: (1, 1, S, S) so it broadcasts over (B, NUM_H, S, S)
            causal_mask = torch.triu(
                torch.ones((S, S), device=q.device, dtype=torch.bool),
                diagonal=1,
            )[None, None, :, :]

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.heads_dim)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)  # (B, NUM_H, S, H) * (B, NUM_H, H, S) -> (B, NUM_H, S, S)

            scores = F.softmax(scores.float(), dim=-1).type_as(q) # TODO: not sure why dim=-1. Here considers converting to FP32 then converting back to dtype(q)
            scores = self.dropout(scores)
            output = torch.matmul(scores, v) # (B, NUM_H, S, H) 
        output = output.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    # Note: why not using nn.Sequential() to implement SwiGLU? - cuz it's not linear pipeline but including parallel structure and a multiplicative operation
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout = config.dropout
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states):
        down_proj = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return down_proj

def load_balancing_loss_func(gate_logits, expert_num, topk):
    # gate_logits: list of (batch_size*seq_len, expert_num) 
    concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0) # (n_layers x batch_size x seq_len, expert_num)
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1) # (n_layers x batch_size x seq_len, expert_num)
    _, selected_indices = torch.topk(routing_weights, topk, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_indices, expert_num) # (n_layers x batch_size x seq_len, topk, expert_num) - one hot encoding for selected experts, where the value is 1 for the selected experts and 0 for the unselected experts
    # tokens_per_expert = torch.mean(expert_mask.float(), dim=0) #  (topk, expert_num)
    tokens_per_expert = expert_mask.sum(dim=(0, 1)) / (expert_mask.shape[0] * expert_mask.shape[1]) # (expert_num) - the percentage being selected among all tokens for each expert
    router_prob_per_expert = torch.mean(routing_weights, dim=0) # the overall weights each expert takes 
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * expert_num

class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(config.hidden_size, config.expert_num)

    def forward(self, hidden_states):
        gate_logit = self.gate(hidden_states) # (batch_size, seq_len, expert_num), (1, 1, expert_num) returns each expert's logit per token
        topk_gate_logit, indices = gate_logit.topk(k=self.topk, dim=-1) # (batch_size, seq_len, topk), (batch_size, seq_len, topk)
        zeros = torch.full_like(gate_logit, float("-inf")) # create a tensor with the same shape as gate_logit and fill it with -inf, which will be used to create sparse logits for selected experts ONLY
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=topk_gate_logit)  # fill the selected experts value based on their indices
        sparse_probs = F.softmax(sparse_logits, dim=-1) # (batch_size, seq_len, expert_num), (1, 1, expert_num) returns the prob of each experts per token where sum of the the selected experts's prob = 1
        gate_logit_flat = gate_logit.view(-1, self.expert_num) # (batch_size*seq_len, expert_num) - the purpose of this is probably ease for load balancing loss calculation
        return sparse_probs, indices, gate_logit_flat

# class Expert(nn.Module):
#     """
#     Same as previous FNN for single expert
#     """
#     pass

class MoE(nn.Module):
    """
    My understanding of MoE: based on multiple experts, return the weigthed result from top K experts for each token for each layer of FNN.
    Later on, using load balancing loss to measure the balance of the expert utilization across all layers and add it as part of the objective function.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gating = Gating(config)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.expert_num)])

    def forward(self, hidden_states):
        sparse_probs, indices, gate_logit_flat = self.gating(hidden_states) # (batch_size, seq_len, expert_num), (batch_size, seq_len, expert_num), (batch_size*seq_len, expert_num)
        final_outputs = torch.zeros_like(hidden_states) 
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])  # (batch_size * seq_len, hidden_size)
        sparse_probs_flat = sparse_probs.view(-1, sparse_probs.shape[-1])  # (batch_size * seq_len, expert_num)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)  # (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1) # (batch_size * seq_len)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # (seq_true, hidden_size), give each expert only use the tokens that select it as input
                export_output = expert(expert_input)  # (seq_true, hidden_size)
                
                gate_scores = sparse_probs_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true) --> (seq_true, 1)
                
                weighted_output = export_output * gate_scores  # (seq_true, hidden_size)
                
                final_outputs[expert_mask] += weighted_output
        
        return final_outputs, gate_logit_flat
        

class DecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = Attention(config)
        # For MOE: add MOE layer
        self.moe = MoE(config)
        self.ffn = FeedForward(config)
        self.input_layernorm = RMSNorm(self.hidden_size)
        self.post_attention_layernorm = RMSNorm(self.hidden_size)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, use_kv_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, use_kv_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.layer_idx % 2 == 0:
            hidden_states = self.ffn(hidden_states)
            gate_logit = None

        else:
            hidden_states, gate_logit = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, gate_logit

class LLM(PreTrainedModel):
    config_class = Config  # for later on: AutoModelForCausalLM.register(Config, LLM)
    
    def __init__(self, config):
        super().__init__(config)
        self.vocat_size = self.config.vocab_size
        self.n_layers = self.config.n_layers
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk
        
        self.dropout = nn.Dropout(self.config.dropout) 
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = torch.nn.ModuleList() 
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(config, layer_idx)) 
        self.layernorm = RMSNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) # each token generated's shape is (hidden_size, vocab_size)
        self.apply(self._init_weights) 
        self.loss = None 
        self.aux_loss = None

        # TODO: Have no idea why it looks like this, looks so hacky - explained by GPT: 
        # the loop over self.named_parameters() looks for tensor names ending with w3.weight (the MLP’s down-projection in a SwiGLU block) 
        # or wo.weight (the attention output projection) and rescales them with a smaller std, 0.02 / sqrt(2 * n_layers), to match the RMSNorm-residual scaling used in LLaMA-style models.
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 


    def forward(self, input_ids, labels, use_kv_cache=False):
        gate_logits = () if self.config.output_router_logits else None

        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states, gate_logit = layer(hidden_states, use_kv_cache=use_kv_cache)
            if gate_logit is not None:
                gate_logits += (gate_logit, )  
        hidden_states = self.layernorm(hidden_states) 

        if labels is None:
            logits = self.output(hidden_states[:, [-1], :])  # [B, 1, V]
            self.loss = None
        else:
            # During training, compute full sequence logits
            logits = self.output(hidden_states)  # [B, S, V]
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            self.loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=0
            )
        
        # add load balancing loss for MoE to ensure all experts will be selected and trained
        if self.config.output_router_logits:
            self.aux_loss = load_balancing_loss_func(gate_logits, self.expert_num, self.topk)
            if labels is not None:
                self.loss += self.config.aux_loss_coef * self.aux_loss.to(self.loss.device)

        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        s = input_ids.shape[1]
        while input_ids.shape[1] < max_new_tokens - 1:  
            inference_res = self.forward(input_ids, labels, use_kv_cache=use_kv_cache)  
            logits = inference_res.logits 
            logits = logits[:, -1, :] 

            # apply penaly for repetitive tokens
            for token in set(input_ids.tolist()[0]):  
                logits[:, token] /= repetition_penalty

            if temperature == 0.0: 
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  
                if top_k is not None:  
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf') 

                probs = F.softmax(logits, dim=-1)  
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

            if idx_next == eos:  
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            if stream:  
                yield input_ids[:, s:]  

        if not stream:  
            yield input_ids[:, s:] 

class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        line = self.data[index]
        line = json.loads(line)
        # text example: "<s><|im_start|>这篇文章的题目是：“旧金山公立图书馆取消罚款政策”。根据该新闻文章，旧金山市公立图书馆现在将取消罚款政策，使人们更容易获得知识并保证公平性。取消罚款政策将为学生和低收入家庭提供巨大帮助，并且不会影响图书馆的资金。旧金山市公立图书馆是美国第一个完全取消罚款的城市图书馆，赢得了全国媒体的赞誉和注意。旧金山市公立图书馆取消罚款政策的目的是什么？答案：目的是为了使人们更容易获得知识、保持公平，并帮助学生和低收入家庭。<|im_end|> <|im_start|>原始文章: 多年来，医学研究表明，过量饮酒会给健康带来危害。据调查，每年因酗酒导致死亡的人数高达数百万。研究人员表示，饮酒过量对身体器官的伤害主要包括肝脏、胰腺和大脑等。在美国，饮酒过量导致的损失和费用高达数百亿美元。因此，为了保持身体健康，建议避免酗酒。医学研究表明，过量饮酒对身体器官的伤害包括肝脏、胰腺和大脑等，而每年因酗酒导致死亡的人数高达数百万。为了避免饮酒过量造成的健康危害和经济损失，建议避免酗酒。<|im_end|></s>"
        text = self.tokenizer.bos_token + line['text'] + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(text)
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - text_len)
        input_ids = np.array(input_ids)
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        return {
            'input_ids': torch.from_numpy(X),
            'labels': torch.from_numpy(Y),
        }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
                
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("./sft/tokenizer")
    tokenizer.pad_token_id = 0
    # tokenizer.bos_token = '<|im_start|>' # based on original pretrain data
    # tokenizer.eos_token = '<|im_end|>'

    dataset = LLMDataset("./sft/dataset/pretrain_hq.jsonl", tokenizer, max_seq_len=512)
    # print('Pretrain data sample: ')
    # print(tokenizer.decode(dataset[1]['input_ids']))
    # print(tokenizer.decode(dataset[1]['labels']))

    config = Config(flash_attn = True)
    model = LLM(config)

    params_num = count_parameters(model)
    print(f'params_num: {params_num}')

    args = TrainingArguments(output_dir='./moe/result', 
                        num_train_epochs=10, 
                        do_train=True, 
                        per_device_train_batch_size=2, # 128
                        gradient_accumulation_steps=1, # 8
                        max_steps=10,
                        logging_steps=100,
                        report_to='tensorboard',
                        save_total_limit=5,
                        bf16=True,
                        learning_rate=2e-4,
                        lr_scheduler_type='cosine',
                        dataloader_num_workers=8,
                        dataloader_pin_memory=True,
                        save_safetensors=False)

    data_collator = DefaultDataCollator()       
    trainer = Trainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, data_collator=data_collator)
    trainer.train(resume_from_checkpoint=False)

    trainer.save_model('./moe/model')
    trainer.save_state()

    # eval result
    AutoConfig.register("custom_gpt", Config)
    AutoModelForCausalLM.register(Config, LLM)
    reload_model = AutoModelForCausalLM.from_pretrained('./moe/model').to(device)
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode("1+1等于几?")
    input_data = {'input_ids': torch.tensor(input_ids, device=device).unsqueeze(0), "labels":None} # unsqueeze(0) to insert a dim at index 0 for batch
    for token in reload_model.generate(inputs=input_data, eos=tokenizer.eos_token_id, max_new_tokens=100, stream=False):
        print(tokenizer.decode(token[0])) 