# Configuration Files

This directory contains YAML configuration files for different RAG (Retrieval-Augmented Generation) experimental setups.

## Files

- `perfect_context.yaml` - Configuration for baseline experiments with ideal context retrieval
- `noise_robustness.yaml` - Configuration for evaluating robustness against noisy or corrupted retrieved documents
- `counterfactual_robustness.yaml` - Configuration for testing resistance to factually similar but incorrect information
- `negative_rejection.yaml` - Configuration for testing the model's ability to reject irrelevant or negative contexts
- `script_pooling.yaml` - Configuration for the experiment pooling manager that executes multiple scripts

## Configuration Structure

All experiment configurations share a common structure:

```yaml
number_of_questions: 100           # Number of questions to evaluate
output_file: "experiment_name"     # Base name for output JSON file
random_seed: 42                    # Seed for reproducibility
dataset_source: "json"             # "json" or "huggingface"
file_path: "../data/en_fact.json"  # Path to dataset file
id_field: "id"                     # Field name for question ID
question_field: "query"            # Field name for question text
answer_field: "answer"             # Field name for ground truth answer

# Additional parameters (experiment-specific)
similarity_threshold: 0.95         # For counterfactual/negative evaluation
noise_ratio: 0.5                   # For noise robustness (0.0 to 1.0)

# Models to evaluate
models:
  - name: "openai/gpt-oss-20b"
    generator: "openai_compatible"  # "openai_compatible" or "google"
    provider: "groq"                 # "groq", "openrouter", "cerebras", or None
  - name: "gemma-3-4b-it"
    generator: "google"
    provider: None
```

## Experiment-Specific Parameters

### Perfect Context (`perfect_context.yaml`)
No additional parameters. Provides only relevant, correct context chunks.

### Noise Robustness (`noise_robustness.yaml`)
- `noise_ratio`: Proportion of irrelevant context (0.0 = no noise, 1.0 = all noise)

### Counterfactual Robustness (`counterfactual_robustness.yaml`)
- `similarity_threshold`: Minimum similarity score for answer evaluation (default: 0.95)

### Negative Rejection (`negative_rejection.yaml`)
- `similarity_threshold`: Maximum similarity to ground truth for successful rejection (default: 0.8)

## Script Pooling Configuration

The `script_pooling.yaml` file controls how the experiment pooling manager executes multiple scripts:

```yaml
scripts:
  - perfect_context.py
  - noise_robustness.py
  - counterfactual_robustness.py
  - negative_rejection.py

max_retries: 3
```

- `scripts`: List of experiment script names to execute (in order)
- `max_retries`: Maximum number of retry attempts for each failed script

## Supported Providers

### OpenAI-Compatible
- **groq**: Fast inference with Groq API
  - Models: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **openrouter**: Access to various open models
  - Models: `qwen/qwen3-4b:free`
- **cerebras**: Cerebras Cloud API
  - Models: `qwen-3-32b`

### Google AI
- Models: `gemma-3-4b-it`, `gemma-3-27b-it`
- No provider needed (uses Google AI Studio)

## Usage

Load configurations using a YAML parser:

```python
import yaml

with open('configs/perfect_context.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access configuration values
num_questions = config['number_of_questions']
models = config['models']
```

## Adding New Models

To add a new model to any experiment:

1. Add a new entry to the `models` list in the configuration file
2. Specify the correct `generator` type and `provider`
3. Ensure the corresponding API key is set in your `.env` file

Example:
```yaml
models:
  - name: "new-model-name"
    generator: "openai_compatible"
    provider: "groq"
```

