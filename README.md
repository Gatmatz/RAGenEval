# RAG Study: Robustness and Faithfulness in Retrieval-Augmented Generation

A research project investigating RAG system robustness across different scenarios, focusing on how models handle perfect contexts, noisy information, counterfactual data, and negative contexts.

## 🎯 Core Research Question

**How robust are RAG systems when faced with different context quality scenarios, and how well do they maintain faithfulness to provided context?**

We evaluate RAG systems across four key dimensions:
- **Perfect Context**: Baseline performance with ideal, relevant context
- **Noise Robustness**: Performance degradation with irrelevant information mixed in
- **Counterfactual Robustness**: Resistance to factually similar but incorrect information
- **Negative Rejection**: Ability to recognize when context doesn't contain the answer

## 📊 Evaluation Scenarios

### 1. Perfect Context
**Objective:** Establish baseline performance when the model receives only relevant, accurate context.
**Metrics:** Faithfulness, Answer Relevancy, Answer Correctness

### 2. Noise Robustness
**Objective:** Measure performance degradation as irrelevant context is injected at various ratios.
**Metrics:** Faithfulness, Answer Relevancy, Answer Correctness

### 3. Counterfactual Robustness
**Objective:** Test if models can resist factually similar but incorrect information (e.g., "Facebook acquired Instagram" vs. "Apple acquired Instagram").
**Metrics:** Similarity to true vs. fake answers, Similarity to counterfactual recognition message

### 4. Negative Rejection
**Objective:** Evaluate whether models can recognize when the provided context doesn't contain relevant information.
**Metrics:** Rejection accuracy, false positive rate

## 🗂️ Project Structure

```
rag-study/
├── src/
│   ├── datasets/           # Dataset loaders (UniversalDataset)
│   ├── retriever/          # Retriever implementations for different scenarios
│   │   ├── PerfectContext.py
│   │   ├── NoiseRobustness.py
│   │   ├── CounterfactualRobustness.py
│   │   └── NegativeRejection.py
│   ├── generator/          # LLM wrapper implementations
│   │   ├── OpenAICompatibleGenerator.py
│   │   ├── GoogleGenerator.py
│   │   └── GroqGenerator.py
│   ├── evaluation/         # Evaluation judges
│   │   ├── LLMJudge.py
│   │   ├── NegativeJudge.py
│   │   └── CounterfactualJudge.py
│   ├── instructors/        # System prompts
│   ├── metrics/            # Custom metrics (Faithfulness, BertScore)
│   └── utils/              # Utility functions
├── configs/                # Experiment configurations (YAML)
│   ├── perfect_context.yaml
│   ├── noise_robustness.yaml
│   ├── counterfactual_robustness.yaml
│   ├── negative_rejection.yaml
│   └── script_pooling.yaml
├── experiments/            # Experiment runner scripts
│   ├── perfect_context.py
│   ├── noise_robustness.py
│   ├── counterfactual_robustness.py
│   ├── negative_rejection.py
│   └── experiment_pooling.py
├── analysis/               # Analysis and plotting scripts
├── data/                   # Dataset files
│   └── en_fact.json
├── output/                 # Experiment results
├── requirements.txt
├── LICENSE
└── README.md
```

## 📚 Datasets

### Dataset Structure
The UniversalDataset class supports:
- **HuggingFace datasets**: Direct loading from the HuggingFace hub (e.g., HotpotQA)
- **Local JSON files**: Custom datasets with flexible field mappings
- Configurable field names for `id`, `question`, and `answer`

## 🔧 Components

### Retriever
Four specialized retriever implementations for different evaluation scenarios:
- **PerfectContext**: Returns only relevant, correct context chunks
- **NoiseRobustness**: Injects configurable ratios of irrelevant context
- **CounterfactualRobustness**: Provides factually similar but incorrect context
- **NegativeRejection**: Provides only irrelevant/negative context

### Generator Models
Supports multiple LLM providers through unified interfaces:
- **OpenAI-Compatible API**: Groq, OpenRouter, Cerebras
  - Examples: GPT-OSS-20B, GPT-OSS-120B, Qwen-3-32B
- **Google AI**: Gemma models (Gemma-3-4B, Gemma-3-27B)
- **Ollama**: Local models running on your machine
  - Examples: Llama 3.2, Mistral, Qwen 2.5, Phi-3, Gemma 2
  - No API key required, fully private

**Key Features:**
- Flexible provider configuration via YAML
- Consistent system prompts across models
- Temperature and token controls
- Mix local and cloud models in the same experiment

### Evaluation

#### Framework
- **RAGAS**: For faithfulness and answer relevancy metrics
- **BertScore**: For semantic similarity measurement
- **LLM-as-a-Judge**: Using Llama-4-Scout or other LLMs via Groq API

#### Judges
- **LLMJudge**: Standard RAG evaluation (faithfulness, relevancy, correctness)
- **NegativeJudge**: Evaluates rejection behavior with similarity thresholds
- **CounterfactualJudge**: Compares answers against true vs. fake answers and counterfactual recognition messages

#### Metrics

| Metric | What it Measures | Used In |
|--------|------------------|---------|
| **Faithfulness** | Answer derived only from provided chunks | Perfect Context, Noise Robustness |
| **Answer Relevance** | Answer addresses the user's query | Perfect Context, Noise Robustness |
| **Answer Correctness** | Semantic similarity to ground truth | Perfect Context, Noise Robustness |
| **True/Fake Similarity** | Semantic similarity to correct vs. incorrect answer | Counterfactual Robustness |
| **Rejection Accuracy** | Ability to identify when answer isn't in context | Negative Rejection |

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Gatmatz/rag-study.git
cd rag-study

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set up your API keys in a `.env` file:
```bash
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
CEREBRAS_API_KEY=your_cerebras_key_here
GOOGLE_API_KEY=your_google_key_here
HF_TOKEN=your_huggingface_token_here
```

### Running Experiments

Each experiment can be run individually:

```bash
cd experiments

# Run perfect context baseline
python perfect_context.py

# Run noise robustness test
python noise_robustness.py

# Run counterfactual robustness test
python counterfactual_robustness.py

# Run negative rejection test
python negative_rejection.py
```

Or use the pooling manager to run multiple experiments sequentially:

```bash
cd experiments
python experiment_pooling.py
```

### Configuring Experiments

Edit the YAML files in `configs/` to customize:
- Number of questions to evaluate
- Models to test
- Dataset source and paths
- Similarity thresholds
- Noise ratios
- Random seeds for reproducibility

Example configuration (`configs/perfect_context.yaml`):
```yaml
number_of_questions: 100
output_file: "perfect_context_evaluation"
random_seed: 42
dataset_source: "json"
file_path: "../data/en_fact.json"
models:
  - name: "openai/gpt-oss-20b"
    generator: "openai_compatible"
    provider: "groq"
```

## 📈 Results & Analysis

Results are saved in JSON format in the `output/` directory, organized by experiment type:
- `output/perfect_context/` - Baseline performance metrics
- `output/noise_robustness/` - Performance vs. noise ratio
- `output/counterfactual_robustness/` - True vs. fake answer similarity
- `output/negative_rejection/` - Rejection accuracy metrics

### Analysis Scripts

The `analysis/` directory contains plotting scripts.

### Key Findings

The experiments evaluate:
1. **Baseline Performance**: How well models perform with perfect context
2. **Noise Tolerance**: Performance degradation curves with increasing noise
3. **Counterfactual Resistance**: Ability to stick to correct information
4. **Rejection Capability**: Recognition of insufficient context

## 📝 Research Methodology

1. **Controlled Context:**
   - Fixed retriever outputs for each scenario
   - Identical questions across all experiments
   - Reproducible with random seeds

2. **Experimental Variables:**
   - Context quality (perfect, noisy, counterfactual, negative)
   - Model size and family
   - Noise ratios (for noise robustness)
   - Similarity thresholds (for evaluation)

3. **Evaluation:**
   - Automated metrics via RAGAS and custom judges
   - LLM-as-judge for complex assessments
   - BertScore for semantic similarity
   - Statistical comparison across models

## 🤝 Contributing

This is a research project. Contributions, suggestions, and discussions are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.