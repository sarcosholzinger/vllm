# LoRA Training Pipeline

This repository implements a Low-Rank Adaptation (LoRA) fine-tuning pipeline for HuggingFace models. It applies LoRA to a base model, trains it on a dataset, and supports efficient fine-tuning using techniques like LoRA.

## Features

- **LoRA Fine-Tuning**: Apply LoRA to transformer models for efficient parameter tuning.
- **HuggingFace Integration**: Uses `transformers`, `datasets`, and `trl` for model and dataset handling.
- **Model Saving and Loading**: Save, load, and merge LoRA weights for inference.

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sarcosholzinger/vllm/blob/main/dev/lora_training_pipeline/lora_pipeline.py
   cd lora-training-pipeline
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare LoRA configuration**: 
   
   Create a `lora_config.conf` file:

   ```json
   {
     "r": 8,
     "lora_alpha": 32,
     "target_modules": "all-linear",
     "lora_dropout": 0.05,
     "bias": "none",
     "task_type": "CAUSAL_LM"
   }
   ```

## Usage

### Running as a Script

To run the training pipeline:

```bash
python lora_pipeline.py
```

### Using as a Module

```python
from your_script import LoRATrainingPipeline

pipeline = LoRATrainingPipeline("meta-llama/Llama-2-7b-hf", "lora_config.conf")
pipeline.load_base_model_and_tokenizer()
pipeline.train_model()
```