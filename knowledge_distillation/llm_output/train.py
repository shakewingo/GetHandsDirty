import os
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from sft.sft_train import SFTDataset
from knowledge_distillation.llm_output.utils import (
    compute_fkl,
    compute_rkl,
    compute_skewed_fkl,
    compute_skewed_rkl,
)


class KDTrainer(Trainer):
    def __init__(
        self,
        model=None,
        teacher_model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        processing_class=None,
        **kwargs,
    ):
        self.teacher_model = teacher_model
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **kwargs,
        )
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        labels = inputs['labels']
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=self.tokenizer.pad_token_id, temp=2.0).mean()
        loss = kl
        print(f'loss: {loss:.4f}')
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _use_lora_teacher = False
    STD_MODEL_PATH = "/gz-data/Qwen2.5-0.5B-Instruct"
    TCH_MODEL_PATH = "/gz-data/Qwen2.5-1.5B-Instruct" # "./knowledge_distillation/Qwen2.5-7B-Instruct"
    OUTPUT_DIR = "./knowledge_distillation/llm_output/results"
    SFT_DATASET_PATH = "/gz-data/sft_mini_512_sub.jsonl"

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=STD_MODEL_PATH,
        local_files_only=True,
    ).to(device)
    print(
        f"student_model params_num: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # use LoRA for student model's fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=256,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config).to(device)
    print(f"student_model trainable model: {model.print_trainable_parameters()}")

    tokenizer = AutoTokenizer.from_pretrained(STD_MODEL_PATH, use_fast=True)

    teacher_model = (
        AutoModelForCausalLM.from_pretrained(TCH_MODEL_PATH, local_files_only=True)
        .to(device)
        .eval()
    )
    # whether load the one fine-tuned with LoRA
    if _use_lora_teacher == True:
        lora_path = "qwen2.5_7b/lora/sft"
        teacher_model = PeftModel.from_pretrained(teacher_model, lora_path)
        teacher_model.cuda()
        teacher_model.eval()
        print(
            f"teacher_model params_num: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)}"
        )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        do_train=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        logging_steps=10,
        report_to="tensorboard",
        save_strategy="epoch",
        save_total_limit=10,
        bf16=True,
        learning_rate=0.0005,
        lr_scheduler_type="cosine",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        max_steps=5,
    )
    data_collator = DefaultDataCollator()
    dataset = SFTDataset(SFT_DATASET_PATH, tokenizer=tokenizer, max_seq_len=512, include_attention_mask=True)
    dataset_collator = DefaultDataCollator()
    trainer = KDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=False)
