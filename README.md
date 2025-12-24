# RAG Study:  Model Size vs.  Context Utilization

A research project investigating how generator model size influences Retrieval-Augmented Generation (RAG) system performance, focusing on the trade-off between context utilization and model capacity.

## 🎯 Core Research Question

**How does the model size of the Generator influence the performance of RAG systems?**

We are testing **Context Utilization vs. Model Capacity** by: 
- Freezing the Retriever component
- Simulating retrieval by providing specific top-k chunks for each query
- Comparing small models (7-8B parameters) vs. large models (70-72B parameters)

## 📊 Hypotheses

### 1. Lost in the Middle
**Hypothesis:** Smaller models struggle to identify and use relevant information when it appears in the middle of the context window, while larger models maintain consistent performance regardless of context position.

### 2. Noise Tolerance Threshold
**Hypothesis:** Larger models can maintain performance with more irrelevant context (noise), while smaller models degrade faster. 

### 3. Context vs.  Priors
**Hypothesis:** Smaller models rely more heavily on pre-training knowledge rather than the provided context, while larger models better adhere to the given chunks.

## 🗂️ Project Structure

```
rag-study/
├── src/
│   ├── retriever/          # Retriever implementation
│   ├── generator/          # Model wrapper implementations
│   ├── evaluation/         # Evaluation metrics and judges
│   └── utils/              # Utility functions
├── configs/                # Experiment configurations
│   ├── baseline.yaml
│   └── lost_in_middle.yaml
├── experiments/            # Experiment scripts
├── tests/                  # Unit tests
│   └── test_retriever.py
├── . gitignore
├── requirements. txt
├── LICENSE
└── README.md
```

## 📚 Datasets

### Primary Dataset
- **Natural Questions (NQ-Open)** - For "Lost in the Middle" hypothesis
- Manually constructed prompts with noise injection

### Additional Datasets (TBD)
- **[RGB Dataset](https://github.com/chen700564/RGB)** - Academic-based dataset for hypotheses 2 & 4
- Other QA datasets as needed

## 🔧 Components

### Retriever
- **Fixed top-k chunks** for every query to ensure consistency
- Optional: Implement actual retriever for comparison if time permits

### Generator Models
Comparing open-source model families:
- **Llama 3**:  8B vs.  70B
- **Qwen 2**:  7B vs. 72B
- Other open-source variants as needed

**Key Controls:**
- Same system prompt for all models
- Context window sized for smallest model (no truncation bias)

### Evaluation

#### Framework
- **LLM-as-a-Judge**: Using frontier models (GPT-4o, Claude 3.5 Sonnet, or Llama 3.1 405B)
- **Tools**:  RAGAS and DeepEval

#### Metrics

| Metric | What it Measures | Why it Matters |
|--------|------------------|----------------|
| **Faithfulness** | Is the answer derived only from provided chunks? | Small models often hallucinate or rely on pre-training memory |
| **Answer Relevance** | Does the answer address the user's query? | Small models get distracted by context and forget the question |
| **Context Adherence** | Did the model use specific details from chunks? | Larger models better pick up subtle details |
| **Coherence/Fluency** | Is the text grammatically correct and readable? | 70B models typically excel over 7B models |

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Gatmatz/rag-study.git
cd rag-study

# Install dependencies
pip install -r requirements. txt
```

### Running Experiments

```bash
# Run baseline experiment
python experiments/run_experiment.py --config configs/baseline.yaml

# Run "Lost in the Middle" experiment
python experiments/run_experiment.py --config configs/lost_in_middle.yaml
```

## 📈 Expected Outcomes

1. **Quantitative Evidence**: Performance metrics showing how model size affects RAG performance
2. **Context Position Analysis**: U-shaped vs. flat performance curves
3. **Noise Tolerance**:  Degradation curves for different model sizes
4. **Context vs. Priors**: Evidence of when models rely on context vs. pre-training

## 📝 Research Methodology

1. **Control Variables:**
   - Fixed retriever outputs
   - Identical prompts across models
   - Same context window constraints

2. **Experimental Variables:**
   - Model size (7-8B vs.  70-72B)
   - Context position (for Lost in the Middle)
   - Noise ratio (for Noise Tolerance)

3. **Evaluation:**
   - Automated metrics via LLM-as-judge
   - Statistical significance testing
   - Qualitative analysis of failure cases

## 🤝 Contributing

This is a research project. Contributions, suggestions, and discussions are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [RGB Dataset](https://github.com/chen700564/RGB)
- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [DeepEval](https://github.com/confident-ai/deepeval)
- Natural Questions Dataset