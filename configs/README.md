# Configuration Files

This directory contains YAML configuration files for different RAG (Retrieval-Augmented Generation) experimental setups.

## Files

- `script_pooling.yaml` - Configuration for the experiment pooling manager that executes multiple scripts
- `negative_rejection.yaml` - Configuration for testing the model's ability to reject irrelevant or negative contexts
- `noise_robustness.yaml` - Configuration for evaluating robustness against noisy or corrupted retrieved documents
- `perfect_context.yaml` - Configuration for baseline experiments with ideal context retrieval

## Script Pooling Configuration

The `script_pooling.yaml` file controls how the experiment pooling manager executes multiple scripts:

```yaml
scripts:
  - perfect_context.py
  - negative_rejection.py
  - noise_robustness.py

max_retries: 3
```

- `scripts`: List of experiment script names to execute (in order)
- `max_retries`: Maximum number of retry attempts for each failed script

## Usage

Load configurations using a YAML parser:

```python
import yaml

with open('configs/perfect_context.yaml', 'r') as f:
    config = yaml.safe_load(f)
```
