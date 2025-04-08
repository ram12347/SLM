import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import psutil
import os
from multiprocessing import freeze_support

def main():
    # ===== 1. HARDWARE SETUP =====
    torch.set_num_threads(psutil.cpu_count(logical=False))

    # ===== 2. MODEL SELECTION =====
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ===== 3. DATA LOADING =====
    dataset = load_dataset('json', data_files={'train': 'tokenized_data.json'})
    dataset = dataset["train"].train_test_split(test_size=0.1)

    # ===== 4. TRAINING ARGS =====
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        dataloader_num_workers=1,  # Reduced for Windows safety
        optim="adamw_torch_fused",
        num_train_epochs=1,
        use_cpu=True,
        torch_compile=False  # Disabled for multiprocessing safety
    )

    # ===== 5. TRAINER SETUP =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # ===== 6. START TRAINING =====
    print("Starting training...")
    trainer.train()
    trainer.save_model("./final_model")

if __name__ == '__main__':
    freeze_support()  # Required for Windows/macOS multiprocessing
    main()
