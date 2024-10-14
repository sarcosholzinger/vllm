import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTConfig, SFTTrainer
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm

# 1. Load the base model and tokenizer
# base_model_name = "gpt2"  # You can change this to any other model
base_model_name = "meta-llama/Llama-2-7b-hf"
print("Loading base model and tokenizer...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    # target_modules=["c_attn"],  # for GPT-2, adjust for other models
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Apply LoRA to the model
print("Applying LoRA to the model...")
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 4. Prepare dataset (using a small subset of WikiText for demonstration)
print("Loading and preparing dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
print("Dataset")
print(dataset)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, desc="Tokenizing dataset"
)
print("Tokenized_dataset:")
print(tokenized_dataset)

# 5. Set up training arguments
print("Setting up training arguments...")
# training_args = TrainingArguments(
#     output_dir="./lora_model",
#     num_train_epochs=1,
#     per_device_train_batch_size=8,
#     save_steps=100,
#     save_total_limit=2,
#     logging_steps=10,
# )
sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=512,
    output_dir="/tmp",
)

# 6. Create Trainer and train the model
print("Creating Trainer and starting training...")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=sft_config,
)

trainer.train()

# 7. Save the LoRA model
print("Saving the LoRA model...")
model.save_pretrained("./lora_model_saved", save_adapter=True, save_config=True)

# 8. Load the saved LoRA model
print("Loading the saved LoRA model...")
loaded_model = PeftModel.from_pretrained(base_model, "./lora_model_saved")

# 9. Merge the LoRA weights with the base model
print("Merging LoRA weights with the base model...")
loaded_model.merge_and_unload()

# 10. Generate text with the merged model
print("Generating text with the merged model...")
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = loaded_model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text (merged model):", generated_text)

# 11. Unmerge the LoRA adapter
print("Unmerging the LoRA adapter...")
loaded_model.unmerge_adapter()

# 12. Generate text with the unmerged model (base model behavior)
print("Generating text with the unmerged model...")
output_unmerged = loaded_model.generate(
    input_ids, max_length=50, num_return_sequences=1
)
generated_text_unmerged = tokenizer.decode(output_unmerged[0], skip_special_tokens=True)
print("Generated text (unmerged model):", generated_text_unmerged)

# 13. Re-enable the LoRA adapter
print("Re-enabling the LoRA adapter...")
loaded_model.set_adapter("default")

# 14. Generate text with the LoRA adapter re-enabled
print("Generating text with the LoRA adapter re-enabled...")
output_readapted = loaded_model.generate(
    input_ids, max_length=50, num_return_sequences=1
)
generated_text_readapted = tokenizer.decode(
    output_readapted[0], skip_special_tokens=True
)
print("Generated text (re-adapted model):", generated_text_readapted)