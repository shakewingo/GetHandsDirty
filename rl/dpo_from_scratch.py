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
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = '<pad>'
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = [json.loads(line) for line in f]

    def __getitem__(self, index):
        sample = self.datas[index]
        prompt = sample['chosen'][0]['content']
        chosen = sample['chosen'][1]['content']
        rejected = sample['rejected'][1]['content']
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(text=text)['input_ids']
        rejected_inputs = self.tokenizer(text=rejected)['input_ids'] + [self.tokenizer.eos_token_id]
        chosen_inputs = self.tokenizer(text=chosen)['input_ids'] + [self.tokenizer.eos_token_id]
        return [prompt_inputs, chosen_inputs, rejected_inputs]
    
    def __len__(self):
        return len(self.datas)
    

class DPODataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, features):
        inputs_ids = []
        labels = []
        
        for feature in features:
            inputs_ids.append(feature[0] + feature[1])
            labels.append([0]*len(feature[0]) + feature[1])
        for feature in features:
            inputs_ids.append(feature[0] + feature[2])
            labels.append([0]*len(feature[0]) + feature[2])

        # input_ids: List[[chose_complete_1], [chose_complete_1], [reject_complete_1], [reject_complete_2]]   
        def process(inputs_ids, labels):
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids]
            labels = [label[:self.max_seq_len] for label in labels]
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []
            
            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len:
                    input_ids = input_ids+[0]*(max_len-len(input_ids))
                    label = label+[0]*(max_len-len(label))
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels
        
        inputs_ids, labels = process(inputs_ids, labels)
        
        return {
            "input_ids": torch.tensor(inputs_ids),
            "labels": torch.tensor(labels)
            }
        

def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

def mask_logits(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels_masks shape: (batch_size, seq_len)
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))
    
    return new_logits


def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)
    
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()
    

class DPOTrainer(Trainer):
    def __init__(self, model, args, ref_model=None, beta=0.1, **kwargs):
        self.beta = beta
        self.ref_model = ref_model.to(model.device).eval() if ref_model is not None else model.to(model.device).eval()
        super().__init__(model, args, **kwargs)

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     input_ids = inputs['input_ids']
    #     labels = inputs['labels']
    #     with torch.no_grad():
    #         ref_logits = ref_model(input_ids=input_ids, labels = labels).logits
    #     ref_probs = logits_to_probs(ref_logits, labels)
    #     ref_probs = mask_logits(ref_probs, labels)
    #     logits = model(input_ids=input_ids, labels = labels).logits
    #     probs = logits_to_probs(logits, labels)
    #     probs = mask_logits(probs, labels)
    #     loss = dpo_loss(ref_probs, probs, 0.1)
    #     return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids'] # (B, S)
        labels = inputs['labels']
        probs, ref_probs = DPOTrainer.calculate_logprobs(model, self.ref_model, input_ids, labels)

        len_chosen = int(len(probs) // 2)
        chosen_probs = torch.cat(probs[:len_chosen])
        reject_probs = torch.cat(probs[len_chosen:])
        chosen_ref_probs = torch.cat(ref_probs[:len_chosen])
        reject_ref_probs = torch.cat(ref_probs[len_chosen:])

        chosen_logratios = chosen_probs - chosen_ref_probs
        reject_logratios = reject_probs - reject_ref_probs
        logits = chosen_logratios - reject_logratios

        loss = -F.logsigmoid(self.beta * logits)
        # print(f'loss: {loss.mean()}')
        return loss.mean()
    
    @staticmethod
    def mask_logits(logits, labels):
        # logits shape: (B, S, V)
        new_logits = []
        for logit, label in zip(logits, labels):
            new_logits.append(logit[label != 0].sum().unsqueeze(0)) # sum() is because each token's prob is log prob, so multiple of prob => sum of log prob
        return new_logits # (B)
    
    @staticmethod
    def calculate_logprobs(model, ref_model, input_ids, labels):
        logits = model.forward(input_ids=input_ids, labels=labels).logits # (B, S, V)
        probs = F.log_softmax(logits, dim=-1) # (B, S, V)
        probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) #  selects only the log probability of the actual target token at each position -> (B, S)
        probs = DPOTrainer.mask_logits(probs, labels) # return a list that has length equal to B whcih calculates the probs to generate the whole sentence of answer -> (B, 1)
        
        with torch.no_grad():
            ref_logits = ref_model.forward(input_ids=input_ids, labels=labels).logits
        ref_probs = F.log_softmax(ref_logits, dim=-1) # (B, S, V)
        ref_probs = torch.gather(ref_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
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
                            per_device_train_batch_size=4, # 16
                            gradient_accumulation_steps=4,
                            max_steps=10,
                            logging_steps=5, # 50
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001, 
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100)          
    dpo_dataset = DPODataset('./sft/dataset/dpo.jsonl', tokenizer=tokenizer)
    data_collator = DPODataCollator(tokenizer=tokenizer, max_seq_len=1024)
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


    
