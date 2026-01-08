# Experiments

This directory contains experimental scripts for evaluating and testing various aspects of the RAG (Retrieval-Augmented Generation) system.

## Scripts

### `experiment_pooling.py`
A pooling manager that executes multiple experiment scripts sequentially with automatic retry logic and completion tracking.

**Features:**
- Reads script names and configuration from a YAML file
- Executes scripts one by one with proper error handling
- Automatically retries failed scripts up to N times (configurable)
- Tracks completion status to avoid re-running completed experiments
- Skips scripts that have already completed successfully
- Provides detailed execution summary

**Usage:**
```bash
python experiment_pooling.py
# Or with a custom config file:
python experiment_pooling.py /path/to/custom_config.yaml
```

**Configuration:**
Edit `configs/script_pooling.yaml` to specify:
- `scripts`: List of experiment script names to run
- `max_retries`: Maximum number of retry attempts for failed scripts

**Status Tracking:**
The script maintains a `.experiment_status.json` file in the experiments directory to track which scripts have completed successfully. To reset and re-run all experiments, simply delete this file.

### `negative_rejection.py`
Tests the system's ability to reject or handle queries when relevant information is not available in the retrieved context.

### `noise_robustness.py`
Evaluates the robustness of the RAG system when dealing with noisy or irrelevant retrieved documents.

### `perfect_context.py`
Benchmarks the system's performance when provided with ideal, perfectly relevant context for queries.

## Usage

Each script can be run independently:

```bash
python negative_rejection.py
python noise_robustness.py
python perfect_context.py
```

Or run all experiments at once using the pooling manager:

```bash
python experiment_pooling.py
```
