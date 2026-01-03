import os
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_fkl(logits, teacher_logits, labels, padding_id=0, temp=2.0, reduction="mean"):
    logits = logits / temp # (B, S, V, E)
    teacher_logits = teacher_logits / temp

    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32) # (B, S, V)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32) # (B, S, V)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) # (B, S, V)
    # assert teacher_log_probs.exp() == teacher_probs, "These two tensors are not equal"
    kl = teacher_probs * (teacher_log_probs - log_probs) 
    kl = kl.sum(-1) # (B, S)
    pad_mask = labels.eq(padding_id) 
    kl = kl.masked_fill_(pad_mask, 0.0)
    if reduction == "sum": 
        kl = kl.sum(dim=1) # (B,)
    elif reduction == "mean":
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1) # (B,)
    return kl

def compute_rkl(logits, teacher_logits, labels, padding_id=0, temp=2.0, reduction="mean"):
    logits = logits / temp # (B, S, V, E)
    teacher_logits = teacher_logits / temp

    probs = torch.softmax(logits, -1, dtype=torch.float32) # (B, S, V)
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32) # (B, S, V)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) # (B, S, V)
    # assert torch.exp(teacher_log_probs) == teacher_probs, "These two tensors are not equal"
    kl = probs * (log_probs - teacher_log_probs) 
    kl = kl.sum(-1) # (B, S)
    pad_mask = labels.eq(padding_id) 
    kl = kl.masked_fill_(pad_mask, 0.0)
    if reduction == "sum": 
        kl = kl.sum(dim=1) # (B,)
    elif reduction == "mean":
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1) # (B,)
    return kl

def compute_skewed_fkl(logits, teacher_logits, labels, padding_id=0, temp=2.0, reduction="mean", skew_lambda=0.1):
    logits = logits / temp # (B, S, V, E)
    teacher_logits = teacher_logits / temp

    probs = torch.softmax(logits, -1, dtype=torch.float32) # (B, S, V)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32) #
    mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs

    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) # (B, S, V)
    mixed_log_probs = torch.log(mixed_probs)

    kl = teacher_probs * (teacher_log_probs - mixed_log_probs)
    kl = kl.sum(-1) # (B, S)
    pad_mask = labels.eq(padding_id) 
    kl = kl.masked_fill_(pad_mask, 0.0)
    if reduction == "sum": 
        kl = kl.sum(dim=1) # (B,)
    elif reduction == "mean":
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1) # (B,)
    return kl


def compute_skewed_rkl(logits, teacher_logits, labels, padding_id=0, temp=2.0, reduction="mean", skew_lambda=0.1):
    logits = logits / temp # (B, S, V, E)
    teacher_logits = teacher_logits / temp
    
    probs = torch.softmax(logits, -1, dtype=torch.float32) # (B, S, V)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32) #
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs

    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32) # (B, S, V)
    mixed_log_probs = torch.log(mixed_probs)

    kl = probs * (log_probs - mixed_log_probs)
    kl = kl.sum(-1) # (B, S)
    pad_mask = labels.eq(padding_id) 
    kl = kl.masked_fill_(pad_mask, 0.0)
    if reduction == "sum": 
        kl = kl.sum(dim=1) # (B,)
    elif reduction == "mean":
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1) # (B,)
    return kl

