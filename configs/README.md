# Configuration Files

This directory contains YAML configuration files for different RAG (Retrieval-Augmented Generation) experimental setups.

## Files

- `negative_rejection.yaml` - Configuration for testing the model's ability to reject irrelevant or negative contexts
- `noise_robustness.yaml` - Configuration for evaluating robustness against noisy or corrupted retrieved documents
- `perfect_context.yaml` - Configuration for baseline experiments with ideal context retrieval

## Usage

Load configurations using a YAML parser:

```python
import yaml

with open('configs/perfect_context.yaml', 'r') as f:
    config = yaml.safe_load(f)
