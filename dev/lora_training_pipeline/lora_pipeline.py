import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
import json
import os


# class LoraConfig:
#     def __init__(
#         self,
#         base_model_name_or_path,
#         # is_prompt_learning,
#         r,
#         lora_alpha,
#         target_modules,
#         lora_dropout,
#         bias,
#         task_type,
#     ):
#         self.base_model_name_or_path = base_model_name_or_path
#         self.r = r
#         self.lora_alpha = lora_alpha
#         self.target_modules = target_modules
#         self.lora_dropout = lora_dropout
#         self.bias = bias
#         self.task_type = task_type


class LoRATrainingPipeline:
    def __init__(self, base_model_name, lora_config):
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.tokenizer = None
        self.model = None
        self.tokenized_dataset = None
        # self.lora_config = None

    def load_base_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        print("Loading base model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # def load_lora_config(self):
    #     """Load LoRA configuration from a JSON file."""
    #     print("Configuring LoRA...")
    #     with open(self.lora_config_file, "r") as f:
    #         config = json.load(f)
    #     self.lora_config = LoraConfig(
    #         base_model_name_or_path=config["base_model_name_or_path"],
    #         # is_prompt_learning=config.get('is_prompt_learning', False),
    #         r=config["r"],
    #         lora_alpha=config["lora_alpha"],
    #         target_modules=config["target_modules"],
    #         lora_dropout=config["lora_dropout"],
    #         bias=config["bias"],
    #         task_type=config["task_type"],
    #     )

    def apply_lora(self):
        """Apply LoRA configuration to the model."""
        print("Applying LoRA to the model...")
        # Pass the actual model (self.model) instead of base model name
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def load_and_prepare_dataset(self):
        """Load and tokenize the dataset."""
        print("Loading and preparing dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=128
            )

        self.tokenized_dataset = dataset.map(
            tokenize_function, batched=True, desc="Tokenizing dataset"
        )

    def train_model(self):
        """Train the model using SFTTrainer."""
        print("Creating Trainer and starting training...")
        sft_config = SFTConfig(
            dataset_text_field="text",
            max_seq_length=512,
            output_dir="/tmp",
        )
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.tokenized_dataset,
            args=sft_config,
        )
        trainer.train()

    def save_model(self, save_directory="./lora_model_saved"):
        """Save the trained LoRA model."""
        print("Saving the LoRA model...")
        self.model.save_pretrained(save_directory, save_adapter=True, save_config=True)

    def load_model(self, load_directory="./lora_model_saved"):
        """Load the saved LoRA model."""
        print("Loading the saved LoRA model...")
        self.model = PeftModel.from_pretrained(self.model, load_directory)

    def merge_lora_weights(self):
        """Merge LoRA weights with the base model."""
        print("Merging LoRA weights with the base model...")
        self.model.merge_and_unload()

    def unmerge_lora_weights(self):
        """Unmerge LoRA weights, returning to the base model state."""
        print("Unmerging the LoRA adapter...")
        self.model.unmerge_adapter()

    def reenable_lora(self):
        """Re-enable the LoRA adapter."""
        print("Re-enabling the LoRA adapter...")
        self.model.set_adapter("default")

    def generate_text(self, input_text):
        """Generate text from the model."""
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main():

    # Set current path
    current_directory = os.getcwd()

    # Set up base model and configuration
    base_model_name = "meta-llama/Llama-2-7b-hf"
    # lora_config_file = os.path.join(
    #     current_directory, "dev/lora_training_pipeline/lora_config.conf"
    # )

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

    # Initialize the pipeline
    pipeline = LoRATrainingPipeline(base_model_name, lora_config)

    # Execute the workflow
    pipeline.load_base_model_and_tokenizer()
    # pipeline.load_lora_config()
    pipeline.apply_lora()
    pipeline.load_and_prepare_dataset()
    pipeline.train_model()
    pipeline.save_model()

    # Demonstrate text generation
    pipeline.load_model()
    pipeline.merge_lora_weights()
    print("Generated text (merged model):", pipeline.generate_text("Once upon a time"))
    pipeline.unmerge_lora_weights()
    print(
        "Generated text (unmerged model):", pipeline.generate_text("Once upon a time")
    )
    pipeline.reenable_lora()
    print(
        "Generated text (re-adapted model):", pipeline.generate_text("Once upon a time")
    )


if __name__ == "__main__":
    main()
