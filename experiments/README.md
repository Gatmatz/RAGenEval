# Experiments

This directory contains experimental scripts for evaluating and testing various aspects of the RAG (Retrieval-Augmented Generation) system.

## Scripts

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
