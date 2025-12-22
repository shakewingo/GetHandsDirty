# use model traied from ./sft/sft_train.py
"""
Author: 
Date: 25/12/20
"""
import os
import json
import math
import random
from typing import List, Optional, Tuple, Union
from prompt_toolkit import prompt
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.utils.data import IterableDataset, Dataset
import numpy as np
# from trl import DPOConfig, DPOTrainer
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from sft.train_llama2_from_scratch import Config, LLM

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = '<pad>'
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        self.max_seq_len = max_seq_len
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas= []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.datas.append(obj)
        
    def __getitem__(self, index):
        item = self.datas[index]
        chosen = item['chosen']  
        rejected = item['rejected']

        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        ) # {"content":"continue","role":"user"},

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        ) # {"content":"continue","role":"user"},
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_seq_len, padding='max_length'
        ) # encoded: {"content":"As Recharge Retreats grows, we plan to expand our team with additional event coordinators and marketing specialists. 
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_seq_len, padding='max_length'
        ) # encoded: {"content":"With Recharge Retreats grows, 

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        y_chosen = y_chosen * mask_chosen

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        y_rejected = y_rejected * mask_rejected

        input_ids = torch.cat([x_chosen, x_rejected], dim=0) # concentrate chosen and reject to be one single tensor
        labels = torch.cat([y_chosen, y_rejected], dim=0)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
        

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_seq_len)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __len__(self):
        return len(self.datas)
    

class DPOTrainer(Trainer):
    def __init__(self, model, args, ref_model=None, beta=0.1, **kwargs):
        self.beta = beta
        self.ref_model = ref_model.to(model.device).eval() if ref_model is not None else model.to(model.device).eval()
        super().__init__(model, args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids'] # (B, S)
        labels = inputs['labels']
        len_chosen = int(input_ids.shape[-1] // 2)
        chosen_input_ids = input_ids[:, :len_chosen]
        reject_input_ids = input_ids[:, len_chosen:]
        chosen_labels = labels[:, :len_chosen]
        reject_labels = labels[:, len_chosen:]

        chosen_probs, chosen_ref_probs = self.calculate_logprobs(model, self.ref_model, chosen_input_ids, chosen_labels)
        reject_probs, reject_ref_probs = self.calculate_logprobs(model, self.ref_model, reject_input_ids, reject_labels)

        chosen_logratios = chosen_probs - chosen_ref_probs
        reject_logratios = reject_probs - reject_ref_probs
        logits = chosen_logratios - reject_logratios

        #  Add numerical stability
        # logits = torch.clamp(logits, min=-10.0, max=10.0)

        loss = -F.logsigmoid(self.beta * logits)
        # print(f'loss: {loss.mean()}')
        return loss.mean()
    
    @staticmethod
    def mask_logits(logits, labels):
        # logits shape: (B, S, V)
        new_logits = []
        for logit, label in zip(logits, labels):
            new_logits.append(logit[label != 0].sum().unsqueeze(0)) # sum() is because each token's prob is log prob, so multiple of prob => sum of log prob
        return torch.concat(new_logits, dim=0) # (B)
    
    @staticmethod
    def calculate_logprobs(model, ref_model, input_ids, labels):
        logits = model.forward(input_ids=input_ids, labels=labels).logits # (B, S, V)
        probs = F.log_softmax(logits, dim=-1) # (B, S, V)
        probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) #  selects only the log probability of the actual target token at each position -> (B, S)
        probs = DPOTrainer.mask_logits(probs, labels) # return a list that has length equal to B whcih calculates the probs to generate the whole sentence of answer -> (B, 1)
        
        with torch.no_grad():
            ref_logits = ref_model.forward(input_ids=input_ids, labels=labels).logits
        ref_probs = F.log_softmax(ref_logits, dim=-1)
        ref_probs = DPOTrainer.mask_logits(ref_probs, labels)
        return probs, ref_probs
    
   
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "mps"
    AutoConfig.register("custom_gpt", Config)
    AutoModelForCausalLM.register(Config, LLM)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='/Users/yingyao/Desktop/Code/GetHandsDirty.nosync/sft/result/sft/checkpoint-6441').to(device)

    print(f'params_num: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    ref_model = AutoModelForCausalLM.from_pretrained('/Users/yingyao/Desktop/Code/GetHandsDirty.nosync/sft/result/sft/checkpoint-6441').to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained("./sft/tokenizer", use_fast=True)

    args = TrainingArguments(output_dir='./sft/result/dpo', 
                            num_train_epochs=1,
                            do_train=True, 
                            per_device_train_batch_size=16, # 16
                            gradient_accumulation_steps=4,
                            # max_steps=100,
                            logging_steps=50, # 50
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001, 
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100)          
    dpo_dataset = DPODataset('./sft/dataset/dpo.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    dpo_dataset[0]
    data_collator = DefaultDataCollator()
    trainer = DPOTrainer(model=model, args=args, train_dataset=dpo_dataset, processing_class=tokenizer, data_collator=data_collator, beta=0.1, ref_model=ref_model)
    # trainer = DPOTrainer(model=model, args=args, train_dataset=dpo_dataset, tokenizer=tokenizer, data_collator=data_collator, ref_model=ref_model)
    trainer.train(resume_from_checkpoint=False)

    # # Use tl DPOTrainer: https://huggingface.co/docs/trl/v0.9.3/dpo_trainer - TODO: has issue to run, dataset format needs to adapt
    # dpo_dataset = load_dataset('json', data_files='./sft/dataset/dpo.jsonl', split="train")
    # args = DPOConfig(
    #     output_dir='./sft/result/dpo', 
    #     num_train_epochs=1,
    #     per_device_train_batch_size=8,#16
    #     gradient_accumulation_steps=4,
    #     max_steps=5,
    #     logging_steps=1,#50
    #     report_to='tensorboard',
    #     save_total_limit=3,
    #     bf16=True,
    #     learning_rate=0.00001, 
    #     lr_scheduler_type='cosine',
    #     save_safetensors=False,
    #     save_steps=100,
    # )
    # trainer = DPOTrainer(model=model, args=args, processing_class=tokenizer, train_dataset=dpo_dataset)
    # trainer.train()


    
