"""
Author: 
Date: 25/12/16
"""
import math
import random
from typing import List, Optional, Tuple, Union
from prompt_toolkit import prompt
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
from train_llama2_from_scratch import Config, LLM

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines() # 几千万条数据
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        line = self.data[index]
        line = json.loads(line)
        # instruction_text = line['instruction']
        conversation = line['conversations']
        # query = instruction_text + input_text
        prompt = [conversation[0]]
        answer = conversation[1]['content'] + self.tokenizer.eos_token
        # messages = []
        # if history:
        #     for i in history:
        #         messages.append({'role': 'user', 'content': i[0]})
        #         messages.append({'role': 'assistant', 'content': i[1]})
        
        # messages.append({'role': 'user', 'content': query}) 
        # prompt example: '<s>user\n生成一个创意，关于未来的家居设计。</s>\n<s>assistant\n'
        # answer example: '未来的家居设计趋势应该是智能化、可持续化、安全性更高。例如集成智能家居系统、使用环保的材料和技术，提高能源利用效率等等。您还有什么关于未来家居设计的具体需求？</s>'
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) 
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        input_ids = prompt_input_ids + answer_input_ids
        labels = [0] * len(prompt_input_ids) + answer_input_ids # the easiest way to not compute loss for prompts
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
            labels = labels + [0] * (self.max_seq_len - text_len)
        
        input_ids = input_ids[:-1]
        labels = labels[1:]
        return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}
    
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AutoConfig.register("custom_gpt", Config)
    AutoModelForCausalLM.register(Config, LLM)
    reload_model = AutoModelForCausalLM.from_pretrained('./sft/model').to(device)
    print(f"Loading model in device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("./sft/tokenizer", use_fast=True)
    # tokenizer.bos_token = '<|im_start|>' # based on original pretrain data
    # tokenizer.eos_token = '<|im_end|>'

    # sample: {"conversations": [{"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"}, {"role": "assistant", "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响：\n\n1. **造纸术**：据史书记载，东汉时期的蔡伦改进了造纸工艺，使得纸张的生产更加便捷、成本更低，质量也更加优良。这一发明极大地促进了文化的传播和保存。\n\n2. **印刷术**：中国古代的印刷术最早可以追溯到唐代的雕版印刷，到了宋代发展出了活字印刷技术。印刷术的发明极大地促进了知识的传播，降低了书籍的成本，对教育和文化的发展起到了重要的推动作用。\n\n3. **火药**：火药最初是在唐代被发现的，最初可能用于医疗或炼金术。到了宋代，人们开始将火药用于军事目的，发明了各种火器。火药的发明改变了战争的面貌，对世界军事技术的发展产生了重大影响。\n\n4. **指南针**：指南针最初在中国被用于风水测量，后来逐渐发展成为航海导航的重要工具。这一发明极大地促进了海上航行技术的发展，对于新航路的开辟和世界地理大发现起到了关键作用。\n\n这四项发明不仅在中国历史上占有重要地位，而且对全世界的科技进步和文明发展都产生了深远的影响。"}]}
    sft_dataset = SFTDataset('./sft/dataset/sft_mini_512_sub.jsonl', tokenizer=tokenizer, max_seq_len=512)
    # print('SFT data sample: ')
    # print(tokenizer.decode(sft_dataset[1]['input_ids']))
    # print(tokenizer.decode(sft_dataset[1]['labels']))

    data_collator = DefaultDataCollator()
    args = TrainingArguments(output_dir='./sft/esult/sft', 
                            num_train_epochs=3,
                            do_train=True, 
                            per_device_train_batch_size=2,#64
                            gradient_accumulation_steps=8,
                            max_steps=100,
                            logging_steps=100, #100
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=5e-5,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False)       
    print(f"Trainer will use device: {args.device}")   
    trainer = Trainer (model=reload_model, args=args, train_dataset=sft_dataset, processing_class=tokenizer, data_collator=data_collator)
    trainer.train(resume_from_checkpoint=False)  # resume_from_checkpoint=True meanings same training process, if you interrupt last time you can continue on the last checkpoint, but not for a new traninig like SFT
    trainer.save_model('./sft/model/sft')
    trainer.save_state()

    # eval result
    AutoConfig.register("custom_gpt", Config)
    AutoModelForCausalLM.register(Config, LLM)
    reload_model = AutoModelForCausalLM.from_pretrained('./sft/model/sft').to(device)
    query = "1+1等于多少？"
    input = f'<s>user\n{query}</s>\n<s>assistant\n'
    input_ids = tokenizer.encode(input)
    input_data = {'input_ids': torch.tensor(input_ids, device=device).unsqueeze(0), "labels":None} # unsqueeze(0) to insert a dim at index 0 for batch
    for token in reload_model.generate(inputs=input_data, eos=tokenizer.eos_token_id, max_new_tokens=100, stream=False):
        print(tokenizer.decode(token[0]))