# Note: foundation is easy to understand but coding part is kind of tedious..

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset
import torch.nn.functional as F
import json
import gc
import os
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.utils.rnn import pad_sequence


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        answer = item['answer']
   
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        
        input_ids = prompt_ids + answer_ids
        labels = answer_ids
    
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
class MyDataCollator:
    # DefaultDataCollator in transformers requires equal length inputs when stack them as batch, 
    # but here I do not want it due to padding with different tokenizer.pad_token_id instead of -100m
    # so use the cutomized data collator without padding
    def __init__(self):
        pass
        
    def __call__(self, features):
        input_ids = [feature['input_ids'] for feature in features]
        labels = [feature['labels'] for feature in features]
        return {'input_ids': input_ids, 'labels': labels} 

    
class ULDLoss(nn.Module):

    def __init__(self, std_tokenizer, tch_tokenizer, temperature=0.2):
        super().__init__()
        self.std_tokenizer = std_tokenizer
        self.tch_tokenizer = tch_tokenizer
        self.temperature = temperature
        vocab_mapping, teacher_matched_ids, student_matched_ids = (
            self.init_vocab_mapping()
        )
        self.vocab_mapping = vocab_mapping
        self.teacher_matched_ids = teacher_matched_ids
        self.student_matched_ids = student_matched_ids

    def init_vocab_mapping(self):
        """
        Returns: vacab_mapping: dict mapping teacher token IDs to student token IDs
                teacher_matched_ids: set of matched teacher token IDs
                student_matched_ids: set of matched student token IDs
        """

        student_vocab = self.std_tokenizer.get_vocab()
        teacher_vocab = self.tch_tokenizer.get_vocab()

        student_token_to_id = dict(student_vocab.items())
        vocab_mapping = {}

        teacher_matched_ids = set()
        student_matched_ids = set()

        for token_str, teacher_token_id in teacher_vocab.items():
            # the criteria of matching is based on token str instead of token id
            if token_str in student_token_to_id:
                student_token_id = student_token_to_id[token_str]
                vocab_mapping[teacher_token_id] = student_token_id
                teacher_matched_ids.add(teacher_token_id)
                student_matched_ids.add(student_token_id)

        return vocab_mapping, teacher_matched_ids, student_matched_ids

    def get_alignment_groups_from_ids(self, std_token_ids, tch_token_ids) -> Tuple[List[List[int]], List[List[int]]]:
        # std_token_ids / tch_token_ids: answer tokens w eos excluded 
        def to_canonical_pieces(tok, ids):
            pieces = []
            prev = ""
            for k in range(len(ids)):
                cur = tok.decode(
                    ids[: k + 1],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                pieces.append(cur[len(prev) :])
                prev = cur
            return pieces

        s_pieces = to_canonical_pieces(self.std_tokenizer, std_token_ids) # e.g. : ['我喜欢', '打', '篮球', '。']
        t_pieces = to_canonical_pieces(self.tch_tokenizer, tch_token_ids) # e,g. : ['我', '喜欢', '打', '篮球', '。']

        i = j = 0
        s_buf = t_buf = ""
        s_group = []
        t_group = []
        s_groups = []
        t_groups = []

        def flush():
            if s_group and t_group:
                s_groups.append(s_group.copy())
                t_groups.append(t_group.copy())

        while i < len(s_pieces) or j < len(t_pieces):
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf = t_buf = ""
                s_group = []
                t_group = []
                continue

            if s_buf == "" and i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
                continue
            if t_buf == "" and j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
                continue

            if len(s_buf) <= len(t_buf):
                if i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
            else:
                if j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1

        if s_buf == t_buf and s_group and t_group:
            flush()
        elif s_group or t_group:

            if s_group or t_group:
                if not s_group:
                    s_group = []
                if not t_group:
                    t_group = []
                if s_group or t_group:
                    s_groups.append(s_group.copy() if s_group else [])
                    t_groups.append(t_group.copy() if t_group else [])
        # e.g. s_groups: [[0], [1], [2, 3], [4]]; t_groups: [[0, 1], [2], [3], [4, 5]]
        return s_groups, t_groups

    @staticmethod
    def merge_prob_with_alignment_groups(probs, alignment_groups):

        if not alignment_groups:
            return probs

        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device)

        for group_idx, group in enumerate(alignment_groups):
            if len(group) > 1:
                eps = 1e-8
                logp = torch.log(probs[group[0]].clamp_min(eps))
                for idx in group[1:]:
                    if idx < probs.size(0):
                        logp = logp + torch.log(probs[idx].clamp_min(eps))
                aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs
    
    @staticmethod
    def get_answer_start_and_len(answers, tokenizer) -> Tuple[List[int], List[int]]:
        answers_index = []
        answers_size = []

        for answer in answers:
            answer_mask = answer.ne(tokenizer.pad_token_id)
            if not answer_mask.any():
                answers_index.append(0)
                answers_size.append(0)
                continue

            indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size
    
    def compute_kl_loss(self, std_probs, tch_probs):
        _, num_matched = std_probs.shape

        std_probs = std_probs.view(-1, num_matched)
        tch_probs = tch_probs.view(-1, num_matched)
 
        std_probs = std_probs / self.temperature
        tch_probs = tch_probs / self.temperature
        
        std_log_probs = F.log_softmax(std_probs, dim=-1)
        tch_log_probs = F.log_softmax(tch_probs, dim=-1)
        kl_loss = F.kl_div(std_log_probs, tch_log_probs, reduction="none", log_target=True)
        return kl_loss.mean()
    
    def compute_hybrid_uld_loss(self, std_aligned, tch_aligned):
        
        device = std_aligned.device
        student_vocab_size = std_aligned.size(-1)
        teacher_vocab_size = tch_aligned.size(-1)

        if self.teacher_matched_ids:
            teacher_matched_token_ids = torch.tensor(sorted(self.teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor(
                [self.vocab_mapping[token_id.item()] for token_id in teacher_matched_token_ids], dtype=torch.long, device=device
            )
        else:
            teacher_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)

        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_token_ids) > 0:
            teacher_matched_mask[teacher_matched_token_ids] = True
            student_matched_mask[student_matched_token_ids] = True

        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_token_ids) > 0:
    
            teacher_matched_probs = tch_aligned[:, teacher_matched_token_ids]  # [seq_len, num_matched]
            student_matched_probs = std_aligned[:, student_matched_token_ids]  # [seq_len, num_matched]
            matched_token_count = teacher_matched_probs.size(-1)
            matched_loss = self.compute_kl_loss(student_matched_probs, teacher_matched_probs)

        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = tch_aligned[:, teacher_unmatched_mask]  # [S, V_unmatched]
        student_unmatched_probs = std_aligned[:, student_unmatched_mask]  # [S, V_unmatched]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
         
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            unmatched_loss = F.l1_loss(student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum")
            unmatched_loss /= std_aligned.size(0)  
        
 
        matched_weight = matched_token_count / max(1, teacher_vocab_size)
        unmatched_weight = 1.0 - matched_weight
 
        total_loss = matched_weight * matched_loss + unmatched_weight * unmatched_loss

        return total_loss
    def compute_uld_loss(self, std_logits, tch_logits, std_labels, tch_labels, std_input_ids, tch_input_ids):
        # align text length
        std_ans_index, std_ans_size = self.get_answer_start_and_len(std_labels, self.std_tokenizer)
        tch_ans_index, tch_ans_size = self.get_answer_start_and_len(tch_labels, self.tch_tokenizer)
        B = std_logits.shape[0]
        losses = []
        for b in range(B):
            # keep only ans part
            std_ans_logits = std_logits[b, std_ans_index[b] : std_ans_index[b] + std_ans_size[b], :] 
            tch_ans_logits = tch_logits[b, tch_ans_index[b] : tch_ans_index[b] + tch_ans_size[b], :] 

            student_probs = F.softmax(std_ans_logits / self.temperature, dim=-1) # (S, V)
            teacher_probs = F.softmax(tch_ans_logits / self.temperature, dim=-1)

            std_token_ids = std_input_ids[b, std_ans_index[b] : std_ans_index[b] + std_ans_size[b]].tolist()  
            tch_token_ids = tch_input_ids[b, tch_ans_index[b] : tch_ans_index[b] + tch_ans_size[b]].tolist()
            std_alignment_groups, tch_alignment_groups = self.get_alignment_groups_from_ids(std_token_ids[:-1], tch_token_ids[:-1])
            
            # calculate the prob that generate same chunk of text for std model and tch model seperately
            std_aligned = self.merge_prob_with_alignment_groups(student_probs[:-1], std_alignment_groups)
            tch_aligned = self.merge_prob_with_alignment_groups(teacher_probs[:-1], tch_alignment_groups)
            std_aligned = torch.cat([std_aligned, student_probs[-1:, :]], dim=0)
            tch_aligned = torch.cat([tch_aligned, teacher_probs[-1:, :]], dim=0)

            # align vocab size - use KL loss to train for matched tokens; use sort + pad and L1 loss to train for unmatched tokens
            aligned_loss = self.compute_hybrid_uld_loss(std_aligned, tch_aligned)
            losses.append(aligned_loss) 
        loss = torch.stack(losses).mean()
        return loss

    def forward(self, std_logits, tch_logits, std_labels, tch_labels, std_input_ids, tch_input_ids):
        loss = self.compute_uld_loss(std_logits, tch_logits, std_labels, tch_labels, std_input_ids, tch_input_ids)
        return loss
    
    
class KDTrainer(Trainer):

    def __init__(
        self,
        model=None,
        tch_model=None,
        tch_tokenizer=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        processing_class=None,
        max_length=512,
        **kwargs,
    ):
        self.tch_model = tch_model.eval() # sometimes model.device will automatically be detected as 'mps', so mnaulaly define it
        self.tch_tokenizer = tch_tokenizer
        self.max_length = max_length
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **kwargs,
        )
        self.uld_loss = ULDLoss(std_tokenizer=processing_class, tch_tokenizer=tch_tokenizer)

    def get_inputs_from_text(self, tokenizer, prompt_texts, ans_texts):
        sequences = []
        labels_list = []
        attention_masks = []
        for prompt_text, ans_text in zip(prompt_texts, ans_texts):
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(ans_text, add_special_tokens=False)
            sequence = prompt_ids + answer_ids
            attention_mask = [1] * len(sequence)
            labels = [tokenizer.pad_token_id] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
            if len(sequence) >= self.max_length:
                sequence = sequence[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                labels = labels[: self.max_length - 1] + [tokenizer.eos_token_id]
            else:
                attention_mask += [1] * (self.max_length - len(sequence))
                labels += [tokenizer.pad_token_id] * (self.max_length - len(sequence) - 1)
                sequence += [tokenizer.pad_token_id] * (self.max_length - len(sequence))
            sequences.append(sequence)
            attention_masks.append(attention_mask)
            labels_list.append(labels)
        sequences = torch.tensor(sequences).contiguous().to(device)
        attention_masks = torch.tensor(attention_masks).contiguous().to(device)
        labels = torch.tensor(labels_list).contiguous().to(device)
        return sequences, attention_masks, labels


    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        prompt_ids = [input_id[:len(input_id) - len(label)] for input_id, label in zip(input_ids, labels)]
        prompt_texts = self.processing_class.batch_decode(prompt_ids) # input: list of str
        answer_texts = self.processing_class.batch_decode(labels)

        std_input_ids, std_attention_mask, std_labels  = self.get_inputs_from_text(self.processing_class, prompt_texts, answer_texts)
        tch_input_ids, tch_attention_mask, tch_labels = self.get_inputs_from_text(self.tch_tokenizer, prompt_texts, answer_texts)

        outputs = model(input_ids = std_input_ids, attention_mask=std_attention_mask)
        with torch.no_grad():
            tch_outputs = self.tch_model(input_ids = tch_input_ids, attention_mask=tch_attention_mask)
        logits = outputs.logits
        tch_logits = tch_outputs.logits

        loss = self.uld_loss(logits, tch_logits, std_labels, tch_labels, std_input_ids, tch_input_ids)
        print(f"loss: {loss:.4f}")
        return (loss, logits) if return_outputs else loss

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    gc.collect()
    torch.cuda.empty_cache()

    DATA_PATH = '/knowledge_distillation/cross_tokenizer/example.json'
    STD_MODEL_PATH = '/gz-data/Qwen2.5-0.5B-Instruct'
    TCH_MODEL_PATH = '/gz-data/GLM-4-9B-0414'# GLM-4-9B-0414
    OUTPUT_DIR = '/knowledge_distillation/cross_tokenizer/result'

    std_model = AutoModelForCausalLM.from_pretrained(STD_MODEL_PATH, local_files_only=True, dtype=torch.bfloat16).to(device)
    tch_model = AutoModelForCausalLM.from_pretrained(TCH_MODEL_PATH, local_files_only=True, dtype=torch.bfloat16).to(device)
    std_tokenizer = AutoTokenizer.from_pretrained(STD_MODEL_PATH, use_fast=True, fix_mistral_regex=True)
    tch_tokenizer = AutoTokenizer.from_pretrained(TCH_MODEL_PATH, use_fast=True, fix_mistral_regex=True)

    dataset = SFTDataset(DATA_PATH, std_tokenizer)
    data_collator = MyDataCollator()
    std_model.floating_point_ops = lambda s: 0 # hidden logic to be compatiable with required input format in Trainer 

    args = TrainingArguments(output_dir=OUTPUT_DIR, 
                        num_train_epochs=1, 
                        do_train=True, 
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=8,
                        logging_steps=1,
                        report_to='tensorboard',
                        save_strategy='steps',
                        save_total_limit=3,
                        save_steps=100,
                        bf16=True,
                        learning_rate=0.00001,
                        lr_scheduler_type='cosine',
                        dataloader_num_workers=8,
                        dataloader_pin_memory=True,
                        gradient_checkpointing=True,  # to save memory
                        max_steps = 5,
                        )
    
    trainer = KDTrainer(model=std_model,
                        tch_model=tch_model, 
                        args=args, 
                        train_dataset=dataset, 
                        processing_class=std_tokenizer, 
                        tch_tokenizer=tch_tokenizer,
                        data_collator=data_collator)

    trainer.train(resume_from_checkpoint=False) 