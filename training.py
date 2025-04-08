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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(psutil.cpu_count(logical=False))

    # ===== 2. MODEL SELECTION & GPU MOVEMENT =====
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)  # Move model to GPU
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ===== 3. DATA LOADING =====
    dataset = load_dataset('json', data_files={'train': 'tokenized_data.json'})
    dataset = dataset["train"].train_test_split(test_size=0.1)

    # ===== 4. TRAINING ARGS (GPU OPTIMIZED) =====
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,  # Increased for GPU efficiency
        gradient_accumulation_steps=8,
        dataloader_num_workers=4,  # Higher for GPU data loading
        optim="adamw_torch_fused",
        num_train_epochs=1,
        fp16=True,  # Enable mixed-precision training (faster on GPUs)
        torch_compile=True,  # Enable if using CUDA >= 11.7
        report_to="none",  # Disable wandb if not needed
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
    print(f"Training on device: {device}")
    print("Starting training...")
    trainer.train()
    trainer.save_model("./final_model")

if __name__ == '__main__':
    freeze_support()  # Required for Windows/macOS multiprocessing
    main()
