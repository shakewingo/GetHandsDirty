# %%
"""
Author:
Date: 25/12/24
"""

from dataclasses import dataclass
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from typing import List, Callable, Dict, Union, Optional
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
# from trl.trainer import grpo_trainer
from grpo.grpo_reward_func import correctness_reward, digit_reward, hard_format_reward, mark_reward


# %%
class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):

        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data["train"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # prompt = self.tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        answer = sample["answer_only"]
        prompt = sample["question"]
        return {"prompt": prompt, "answer": answer}


# %%
args = TrainingArguments()
writer = SummaryWriter("./grpo/result/runs")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', fix_mistral_regex=True) # Added leading slash
policy_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', fix_mistral_regex=True)

# %%
dataset = GSM8KDataset('./dataset/gsm8k_chinese', tokenizer)

# %%
dataset[0]

# %%
@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: str
    answer: str
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[torch.Tensor, int]
    response_length: torch.Tensor


@dataclass
class Experience:
    prompt_response_ids: torch.Tensor
    action_log_probs: torch.Tensor
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    # kl: Optional[torch.Tensor] = None

# %%
SYSTEM_PROMPT = """
    Use below format to answer questions
    <think>
    your thinking process
    </think>
    <answer>
    your answer
    </answer>
    """

# %%
class GRPOTrainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        tokenizer,
        reward_funcs: List[Callable] = None, # here doesn't include loaded reward model from_pretrained
        ref_model=None,
        **kwargs
    ):
        self.model = model.to(model.device)
        print(f'model_device: {model.device}')
        self.args = args
        self.ref_model = (
            ref_model.to(model.device).eval()
            if ref_model is not None
            else model.to(model.device).eval()
        )
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.reward_funcs = reward_funcs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.max_steps = self.args.max_steps if hasattr(self.args, 'max_steps') else None

    def generate_samples(
        self,
        inputs: List[Dict[str, str]],
    ) -> List[Samples]:
        samples_list = []
        self.model.eval()
        for prompt, answer in zip(inputs["prompt"], inputs["answer"]):
            total_length = self.args.max_generate_length + self.args.max_prompt_length
            input_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.tokenizer(
                input_text,
                padding="max_length",
                max_length=self.args.max_prompt_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                prompt_response_ids = self.model.generate(
                    **inputs.to(self.model.device),
                    max_new_tokens=self.args.max_generate_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=self.args.num_sample_generations, # generate a group responses
                    do_sample=True,
                )
            if (
                prompt_response_ids.size(1)
                >= total_length
            ):
                prompt_response_ids = prompt_response_ids[
                    :, : total_length
                ]
            else:
                prompt_response_ids = torch.cat(
                    [
                        prompt_response_ids,
                        torch.full(
                            (
                                prompt_response_ids.size(0),
                                total_length
                                - prompt_response_ids.size(1),
                            ),
                            fill_value=self.tokenizer.pad_token_id,
                            device=prompt_response_ids.device,
                        ),
                    ],
                    dim=1,
                )
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(
                dtype=torch.long
            )  # masked since beginning of the prompt
            response_ids = prompt_response_ids[:, input_ids.size(1) :]
            action_mask = (
                response_ids.ne(self.tokenizer.pad_token_id)
                & response_ids.ne(self.tokenizer.eos_token_id)
            ).to(
                dtype=torch.long
            )  # masked since beginning of the answer

            # each samples is a "group" per prompt in GRPO
            samples = Samples(
                prompt_response_ids=prompt_response_ids, # shape: (num_sample_generations, total_length)
                response_ids=response_ids,
                prompt=prompt, # shape: (1)
                answer=answer, # shape: (1)
                attention_mask=attention_mask, # shape: (num_sample_generations, total_length)
                action_mask=action_mask, # shape: (num_sample_generations, max_generate_length)
                num_actions=action_mask.size(1), # shape: (1)
                response_length=action_mask.float().sum(dim=-1), # shape: (num_sample_generations, 1)
            )
            samples_list.append(samples)
        return samples_list

    def generate_experiences(self, samples_list: List[Samples]):
        self.model.eval()
        experiences = []
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_sample_generations, total_length)
            prompt = samples.prompt # shape: (1)
            answer = samples.answer # shape: (1)
            response_ids = samples.response_ids 
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            num_actions = samples.num_actions # shape: (1)
            with torch.no_grad():
                # get output logits
                outputs = self.model(prompt_response_ids, attention_mask=attention_mask)
                log_probs = F.softmax(outputs.logits[:, :-1, :], dim=-1)  # (B, S, V)
                log_probs_labels = log_probs.gather(
                    dim=-1, index=prompt_response_ids[:, 1:].unsqueeze(-1)
                )  # (B, S, 1)
                action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:] # truncate for only output part

                responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                
                # calculate reward
                # here ignores weights across different reward functions
                prompts = [prompt] * response_ids.size(0)
                answers = [answer] * response_ids.size(0)
                final_reward = torch.zeros(response_ids.size(0), device=self.model.device)
                for reward_func in self.reward_funcs:
                    reward = reward_func(prompts, responses, answers)
                    # Convert list to tensor and accumulate
                    final_reward += torch.tensor(reward, device=self.model.device, dtype=torch.float)
                
                final_reward /= len(self.reward_funcs)
                print(f'prompt: {prompt}, grouped rewards: {final_reward}')

                # noramlize group adv
                final_reward_mean = final_reward.mean()
                final_reward_std = final_reward.std()
                advantages = (final_reward - final_reward_mean) / (final_reward_std + 1e-8)  # (B)

                experiences.append(
                    Experience(
                        prompt_response_ids,
                        action_log_probs.detach(),
                        advantages,
                        attention_mask,
                        action_mask,
                        num_actions
                    )
                )

        return experiences
    
    @staticmethod
    def compute_approx_kl(
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ):
        # GRPO uses K3 KL divergence
        log_ratio = log_probs.float() - ref_log_probs.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        kl = log_ratio.exp() - 1 - log_ratio
        return kl
    
    def compute_core_loss(self, new_probs, old_probs, advantages, kl, action_mask=None):
        advantages = advantages.unsqueeze(1)  # shape: (B) -> (B, 1)
        ratio = (new_probs - old_probs).exp()
        loss = (
            -torch.min(
                ratio * advantages, ratio.clamp(1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
            )
            + self.args.beta * kl
        )
        if action_mask is None:
            return loss.mean(-1).mean()
        return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

    
    def compute_loss(self, experiences):
        prompt_response_ids = torch.cat([exp.prompt_response_ids for exp in experiences], dim=0)
        old_action_log_probs = torch.cat([exp.action_log_probs for exp in experiences], dim=0)
        advantages = torch.cat([exp.advantages for exp in experiences], dim=0)
        attention_mask = torch.cat([exp.attention_mask for exp in experiences], dim=0)
        action_mask = torch.cat([exp.action_mask for exp in experiences], dim=0)
        num_actions = experiences[0].num_actions
        
        logits = self.model(prompt_response_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=prompt_response_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

        # get ref's output logits
        ref_output = self.ref_model(prompt_response_ids, attention_mask=attention_mask).logits
        ref_log_probs = F.log_softmax(ref_output[:, :-1, :], dim=-1)
        ref_log_probs_labels = ref_log_probs.gather(
            dim=-1, index=prompt_response_ids[:, 1:].unsqueeze(-1)
        )  # seqs[:, 1:] shape: [batch_size, seq_len-1] – these are the target token IDs (next token at each position) # .unsqueeze(-1) makes that [batch_size, seq_len-1, 1] so it can be used as an index.
        ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:] # truncate for only output part

        kl = self.compute_approx_kl(
            action_log_probs, ref_action_log_probs, action_mask=action_mask
        )
        loss = self.compute_core_loss(
            action_log_probs, old_action_log_probs, advantages, kl, action_mask=action_mask
        )
        return loss
    
    def train_step(self, experiences):
        self.model.train()
        loss = self.compute_loss(experiences)
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        print(f"step: {self.step}  policy_loss: {loss.item():.3f}")

        if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"step {self.step}: optimizer updated!")
        else:
            print(f"step {self.step}: gradients accumulated...")
        
        if (self.step + 1) == self.global_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def train(self):
        self.global_steps = self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        self.step = 0
        for _ in range(self.args.epoch):
            dataloader = DataLoader(
                self.train_dataset, batch_size=self.args.batch_size, shuffle=True
            ) # -> Dict[List], e.g. {'prompt': [...], 'answer': [...]}
            for _, batch_input in enumerate(dataloader):
                samples = self.generate_samples(batch_input)
                # print(samples)
                experiences = self.generate_experiences(samples)         
                # self.exps_buffer[idx % self.args.gradient_accumulation_steps] = experiences # TODO: not sure why?
                # for step, exp_buffer in enumerate(self.exps_buffer):
                self.train_step(experiences)
                self.step += 1
                if self.max_steps is not None and self.step >= self.max_steps:
                    print(f"Reached max steps of {self.max_steps}, stopping training.")
                    return
                if self.step % self.args.save_steps == 0:
                    self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.step}')
                    self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.step}')
                del samples 

# %%
class GRPOArguments:
    output_dir="./grpo/result"
    epoch=1
    lr = 0.000001
    batch_size=2  # 16
    gradient_accumulation_steps=2  # $$\text{Effective Batch Size} = \text{Batch Size per GPU} \times \text{Gradient Accumulation Steps} \times \text{Number of GPUs}$$
    save_steps=10
    num_sample_generations=3  # num_sample genereated per prompt
    max_prompt_length=256
    max_generate_length=512
    reward_weights=None  # only applied if multiple reward funcs are used
    clip_eps=0.2
    beta=0.1
    max_steps = 10

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
args = GRPOArguments()
trainer = GRPOTrainer(model=policy_model,
                        args=args,
                        train_dataset=dataset,
                        tokenizer=tokenizer,
                        reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward])
trainer.train()


# %%



