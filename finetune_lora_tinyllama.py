import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from transformers.trainer_utils import set_seed

# === Specify CUDA device explicitly (adjust if needed) ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths for model, dataset, and output ===
model_path = r"models\TinyLlama-1.1B-Chat-v1.0"
data_path = r"data\imdb_finetune_chat.jsonl"  # use chat format data
output_dir = r"models\lora_adapter_tinyllama"

# === Set seed for reproducibility ===
set_seed(42)

# === Load tokenizer and set pad token if not defined ===
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# === Load base model and move it to CUDA device ===
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
model = model.to(device)

# === Configure LoRA for fine-tuning ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# === Load dataset from JSONL file ===
dataset = load_dataset("json", data_files=data_path, split="train")

# === Tokenize using the chat template expected by TinyLlama chat model ===
def tokenize_chat(examples):
    input_ids_list = []
    attention_mask_list = []
    for messages in examples["messages"]:
        input_ids = tokenizer.apply_chat_template(
            messages,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    return {
        "input_ids": input_ids_list,
        "labels": [ids.copy() for ids in input_ids_list],
        "attention_mask": attention_mask_list,
    }

# === Apply tokenization to entire dataset, removing original columns ===
tokenized_dataset = dataset.map(tokenize_chat, batched=True, remove_columns=dataset.column_names)

# === Data collator for padding, no MLM (causal LM) ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# === Define training arguments adapted for limited 4GB GPU ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,         # Reduce batch size to fit in 4GB VRAM
    gradient_accumulation_steps=32,        # Accumulate gradients to simulate larger batch
    num_train_epochs=4,                     # Increase epochs to compensate small batch
    fp16=True,                             # Mixed precision to reduce memory usage
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=1e-4,                    # Slightly lower LR for better convergence
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none",
    save_total_limit=1,
    remove_unused_columns=False,
    optim="adamw_torch",                   # More stable optimizer for LoRA training
    max_grad_norm=1.0                      # Gradient clipping for stable training
)

# === Initialize Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Start training ===
for epoch in range(training_args.num_train_epochs):
    # Memory monitoring: Before training starts for the current epoch
    print(f"Epoch {epoch + 1} starting...")
    print(f"Allocated memory before training: {torch.cuda.memory_allocated() / 1024**2} MB")  # Memory allocated by CUDA (in MB)
    print(f"Reserved memory before training: {torch.cuda.memory_reserved() / 1024**2} MB")  # Memory reserved by CUDA (in MB)

    # Train the model for the current epoch
    trainer.train()

    # Memory monitoring: After training finishes for the current epoch
    print(f"Allocated memory after training: {torch.cuda.memory_allocated() / 1024**2} MB")  # Memory allocated by CUDA (in MB)
    print(f"Reserved memory after training: {torch.cuda.memory_reserved() / 1024**2} MB")  # Memory reserved by CUDA (in MB)

    # Clear CUDA memory cache to free up unused memory
    torch.cuda.empty_cache()
    print(f"Epoch {epoch + 1} finished. Cache cleared.")

