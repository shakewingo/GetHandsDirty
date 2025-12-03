"""
Author: 
Date: 25/12/02
"""
from dataclasses import dataclass
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Union, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.trainer import ppo_trainer


class PromptDataset(Dataset): 
    def __init__(self, prompts, tokenizer, apply_chat_template=True):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []
        for prompt in self.prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    content, tokenize=False, apply_chat_template=True
                )
            else:
                prompt = self.tokenizer.bos_token + prompt
            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.final_prompts[idx]


# value model - initiallized from policy model, plus a linear head
# output: (batch_size, seq_len, 1), i.e. each reward per token in per seq


class Critic(nn.Module):
    def __init__(self, policy_model):
        super().__init__()
        self.policy_model = policy_model
        self.policy_model.eval()  # freeze policy model
        self.value_head = nn.Linear(policy_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, num_actions):
        hidden_states = self.policy_model(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
        value_model_output = self.value_head(hidden_states)
        # remove last dimensions of a tensor if last dimensions has a size of 1
        # only keep the last num_actions tokens's value as previous are prompts
        values = value_model_output.squeeze(-1)[:, -num_actions:]
        return values


def compute_value_loss(
    values, old_values, returns, clip_eps: float = None, action_mask=None
):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_policy_loss(
    old_probs, new_probs, advantages, action_mask=None, clip_eps=0.2
):
    ratio = (new_probs - old_probs).exp()
    loss = -torch.min(
        ratio * advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    )
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):
    # importance sampling & off-policy learning
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio


def compute_reward(kl, r, action_mask, kl_ctl, clip_reward_value):
    # TODO: Don't understand the reason behind?

    kl_divergence_estimate = -kl_ctl * kl
    reward = kl_divergence_estimate

    ends = action_mask.sum(1) + 1

    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)

    reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)
    batch_size = r.size(0)
    for j in range(batch_size):
        reward[j, : ends[j]][-1] += reward_clip[j, 0]
    return reward


def get_advantages_and_returns(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    lamda: float,
):
    # delta(t) = R(t) + gam*V(t+1) - V(t)
    # GAE:A(t) = delta(t) + gam*lam*A(t+1)
    # 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
    # A(T-1) = delta(T-1) + gam*lam*A(T) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推得到A(t)
    # TODO: returns =  advantages + values
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    response_length = rewards.size(1)
    advantages_reversed = []
    last_gae = 0

    for t in reversed(range(response_length)):
        values = action_mask * values
        rewards = action_mask * rewards
        next_value = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lamda * last_gae
        advantages_reversed.append(last_gae)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions",
        )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value

        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer) - self.limit :]

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)


@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[torch.Tensor, int]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor


@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None


@dataclass
class BufferItem:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]


def collate_fn(batch):

    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []

    for x in batch:
        seqs.append(x["seqs"])
        action_log_probs.append(x["action_log_probs"])
        values.append(x["values"])
        returns.append(x["returns"])
        advantages.append(x["advantages"])
        attention_mask.append(x["attention_mask"])
        action_mask.append(x["action_mask"])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)

    return BufferItem(
        seqs,
        action_log_probs,
        values,
        returns,
        advantages,
        attention_mask,
        action_mask,
        action_mask.size(1),
    )


def generate_samples(
    prompts,
    policy_model,
    micro_rollout_batch_size,
    n_samples_per_prompt,
    max_new_tokens,
    max_length,
):
    samples_list = []
    policy_model.eval()
    all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i : i + micro_rollout_batch_size]
        inputs = policy_tokenizer(
            prompts,
            padding=PADDING,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        seqs = policy_model.generate(
            **inputs.to(device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, : max_new_tokens + max_length]
        else:
            seqs = torch.cat(
                [
                    seqs,
                    torch.full(
                        (seqs.size(0), max_new_tokens + max_length - seqs.size(1)),
                        fill_value=pad_token_id,
                        device=seqs.device,
                    ),
                ],
                dim=1,
            )
        attention_mask = (seqs.ne(pad_token_id)).to(
            dtype=torch.long
        )  # masked since beginning of the prompt
        ans = seqs[:, input_ids.size(1) :]
        action_mask = (ans.ne(pad_token_id) & ans.ne(eos_token_id)).to(
            dtype=torch.long
        )  # masked since beginning of the answer

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
    samples_list.append(samples)
    return samples_list


def generate_expeirences(samples_list):
    policy_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()
    experiences = []
    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        with torch.no_grad():
            # get output logits
            output = policy_model(seqs, attention_mask=attention_mask)
            log_probs = F.softmax(output.logits[:, :-1, :], dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

            # get ref's output logits
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_log_probs = F.softmax(ref_output.logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(
                dim=-1, index=seqs[:, 1:].unsqueeze(-1)
            )  # seqs[:, 1:] shape: [batch_size, seq_len-1] – these are the target token IDs (next token at each position) # .unsqueeze(-1) makes that [batch_size, seq_len-1, 1] so it can be used as an index.
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]

            # get output's value
            values = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            seq_texts = policy_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # calculate output's reward from reward model
            reward_model_inputs = reward_tokenizer(
                seq_texts, return_tensors="pt", padding=True
            )
            r = reward_model(**reward_model_inputs.to(device)).logits

            # calculate KL divergence
            kl = compute_approx_kl(
                action_log_probs, ref_action_log_probs, action_mask=action_mask
            ).to(device)
            # calculate the modified output's reward
            rewards = compute_reward(
                kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2
            )

            # calculate the adv and return
            advantages, returns = get_advantages_and_returns(
                values, rewards, action_mask, 0.1, 0.2
            )
            experiences.append(
                Experience(
                    seqs,
                    action_log_probs.detach(),
                    values.detach(),
                    returns.detach(),
                    advantages.detach(),
                    attention_mask,
                    action_mask,
                    r.detach(),
                    samples.response_length,
                    samples.total_length,
                    num_actions,
                    kl.detach(),
                )
            )

    return experiences


def train_step(experience, steps=0):
    seqs = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns

    policy_model.train()
    optimizer_policy.zero_grad()

    logits = policy_model(seqs, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    policy_loss = compute_policy_loss(
        action_log_probs, old_action_log_probs, advantages, action_mask=action_mask
    )
    policy_loss.backward()
    optimizer_policy.step()
    writer.add_scalar("policy_loss", policy_loss.item(), steps)

    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(seqs, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(
        f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}"
    )


def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for eposide in range(episodes):
        for rand_prompts in prompts_dataloader:
            samples = generate_samples(
                rand_prompts,
                policy_model,
                micro_rollout_batch_size,
                n_samples_per_prompt,
                max_new_tokens,
                max_length,
            )
            # generate experience that includes adv, rewards, returns
            experiences = generate_expeirences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(
                buffer,
                batch_size=micro_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            torch.cuda.empty_cache()
            for _ in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            buffer.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # epochs
    episodes = 3
    # how many epochs taken when generating experience in training
    max_epochs = 5
    # how many records used to generate samples each time
    rollout_batch_size = 8
    # how many records used once to generate experience each time (due to memory limitation, parallel execution required)
    micro_rollout_batch_size = 2
    n_samples_per_prompt = 2
    # max generated length
    max_new_tokens = 50
    max_length = 256
    # actual size used to generated experience in training
    micro_train_batch_size = 2

    # logging
    writer = SummaryWriter("./runs")
    # model def
    POLICY_MODEL_NAME = REF_MODEL_NAME = "./rl/Qwen2.5-0.5B-Instruct"
    REWARD_MODEL_NAME = "./rl/reward-model-deberta-v3-large-v2"
    policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME
    ).to(device)

    policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        REWARD_MODEL_NAME, fix_mistral_regex=True
    )
    # policy_model.save_pretrained("./rl/Qwen2.5-0.5B-Instruct")
    # policy_tokenizer.save_pretrained("./Qwen2.5-0.5B-Instruct")
    # reward_model.save_pretrained("./rl/reward-model-deberta-v3-large-v2")
    # reward_tokenizer.save_pretrained("./reward-model-deberta-v3-large-v2")
    critic_model = Critic(policy_model.base_model).to(device)

    # initilization
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)

    # padding
    PADDING = "max_length"
    policy_tokenizer.padding_side = "left"
    eos_token_id = policy_tokenizer.eos_token_id
    pad_token_id = policy_tokenizer.pad_token_id
    prompt_list = [
        "请问1+1等于多少？",
        "PowerShell，如何知道BIOS中的虚拟化是否已禁用",
        "为什么人们喜欢在水族馆里游泳，而不是在游泳池里？",
        "你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。",
        "你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。",
        "你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。",
        "为什么所有的镜子都是矩形的？",
        "我们在受感染的植物根部可以找到哪一种，臭氧还是金子？",
    ]
    prompts_dataset = PromptDataset(
        prompt_list, policy_tokenizer, apply_chat_template=True
    )
    prompts_dataloader = DataLoader(
        prompts_dataset, batch_size=rollout_batch_size, shuffle=True
    )

    train()
