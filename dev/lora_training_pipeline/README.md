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

### Debugging with `launch.json` in VS Code

Add the following configuration to the `launch.json` file:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Debug LoRA Pipeline",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/dev/lora_training_pipeline/lora_pipeline.py",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "VLLM_ALLOW_DEPRECATED_BEAM_SEARCH": "1"
      },
      "args": []
    }
  ]
}
```

#### Explanation of the Configuration

- **`name`**: The name of the debug configuration. You can change this to something more descriptive, like `"Python: Debug LoRA Pipeline"`.
- **`type`**: This should be `"python"`, specifying that you're debugging a Python script.
- **`request`**: Use `"launch"` to launch and debug the Python file.
- **`program`**: This should point to the Python file you want to debug. Replace `"path/to/your_script.py"` with the relative path to your Python script.
- **`console`**: `"integratedTerminal"` runs the script in VS Code’s terminal.
- **`justMyCode`**: Set this to `false` if you want to debug inside third-party libraries (like `peft` or `transformers`).
- **`env`**: Sets environment variables. For instance, in this case, the `VLLM_ALLOW_DEPRECATED_BEAM_SEARCH` environment variable is set to `"1"`.
- **`args`**: This can be used to pass any additional command-line arguments to your script.

#### 4. Start Debugging

To start debugging:

1. Open the script you want to debug in VS Code.
2. Press `F5` or go to **Run > Start Debugging**.