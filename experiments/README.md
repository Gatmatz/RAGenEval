# Experiments

This directory contains experimental scripts for evaluating and testing various aspects of the RAG (Retrieval-Augmented Generation) system.

## Scripts

### `perfect_context.py`
Baseline experiment that evaluates RAG performance with ideal, relevant context only.

**Purpose:**
- Establish performance baseline with optimal conditions
- Measure faithfulness, answer relevancy, and answer correctness
- Provide reference point for other experiments

**Retriever:** `PerfectContext` - Returns only relevant, accurate context chunks  
**Judge:** `LLMJudge` - Evaluates using RAGAS metrics  
**Configuration:** `configs/perfect_context.yaml`

**Output:** JSON file with:
- Individual question results with RAGAS metrics
- Aggregate metrics (mean faithfulness, relevancy, correctness)
- Generated answers for qualitative analysis

### `noise_robustness.py`
Evaluates the robustness of the RAG system when dealing with noisy or irrelevant retrieved documents.

**Purpose:**
- Measure performance degradation with increasing noise
- Test model's ability to filter relevant from irrelevant information
- Identify noise tolerance thresholds

**Retriever:** `NoiseRobustness` - Injects configurable ratio of irrelevant context  
**Judge:** `LLMJudge` - Evaluates using RAGAS metrics  
**Configuration:** `configs/noise_robustness.yaml`  
**Key Parameter:** `noise_ratio` (0.0-1.0)

**Output:** Performance metrics at specified noise level

### `counterfactual_robustness.py`
Tests the system's resistance to factually similar but incorrect information.

**Purpose:**
- Evaluate if models can distinguish between correct and counterfactual information
- Measure tendency to be misled by plausible but wrong facts
- Compare semantic similarity to true vs. fake answers

**Example:** Testing if a model can correctly answer "Who acquired Instagram?" (Facebook) when given context stating "Apple acquired Instagram"

**Retriever:** `CounterfactualRobustness` - Provides factually similar but incorrect context  
**Judge:** `CounterfactualJudge` - Compares similarity to true and fake answers  
**Configuration:** `configs/counterfactual_robustness.yaml`  
**Key Parameter:** `similarity_threshold` for evaluation

**Output:** JSON file with:
- True answer similarity scores
- Fake answer similarity scores
- Accuracy metrics (correct vs. misled responses)

### `negative_rejection.py`
Tests the system's ability to recognize when relevant information is not available in the retrieved context.

**Purpose:**
- Evaluate model's capability to say "I don't know"
- Measure false positive rate (answering when shouldn't)
- Test adherence to provided context

**Retriever:** `NegativeRejection` - Provides only irrelevant/negative context  
**Judge:** `NegativeJudge` - Evaluates rejection behavior  
**Configuration:** `configs/negative_rejection.yaml`  
**Key Parameter:** `similarity_threshold` (answers below threshold = successful rejection)

**Output:** JSON file with:
- Individual rejection decisions
- Similarity scores to ground truth
- Rejection accuracy rate

### `experiment_pooling.py`
A pooling manager that executes multiple experiment scripts sequentially with automatic retry logic and completion tracking.

**Features:**
- Reads script names and configuration from `configs/script_pooling.yaml`
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

## Running Experiments

Each script can be run independently from the `experiments/` directory:

```bash
cd experiments

# Run individual experiments
python perfect_context.py
python noise_robustness.py
python counterfactual_robustness.py
python negative_rejection.py

# Or run all experiments sequentially
python experiment_pooling.py
```

## Experiment Workflow

Each experiment follows this general workflow:

1. **Load Configuration**: Read settings from YAML file
2. **Initialize Dataset**: Load data from JSON or HuggingFace
3. **Initialize Components**:
   - Create appropriate Retriever for the scenario
   - Set up Generator(s) for each model to test
   - Configure Judge for evaluation
4. **Generate Evaluation Data**: Create question/context pairs using QA_Selector
5. **Run Evaluation**: For each model:
   - Generate answers using the Generator
   - Evaluate answers using the Judge
   - Save results to JSON file

## Output Structure

Results are saved to `output/{experiment_name}/` with the filename format:
```
{output_file_base}_{model_name}.json
```

Example: `output/perfect_context/perfect_context_evaluation_openai_gpt-oss-20b.json`

Each output file contains:
- `questions`: List of evaluated questions
- `answers`: Generated answers
- `contexts`: Context provided to the model
- `ground_truths`: Correct answers (when applicable)
- `metrics`: Per-question evaluation metrics
- `aggregate_metrics`: Overall performance statistics

## Prerequisites

Before running experiments:

1. **API Keys**: Set up required API keys in `.env` file:
   ```
   GROQ_API_KEY=your_key
   OPENROUTER_API_KEY=your_key
   CEREBRAS_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   HF_TOKEN=your_token
   ```

2. **Dataset**: Ensure dataset files are in the `data/` directory

3. **Configuration**: Review and adjust YAML configurations as needed

## Troubleshooting

- **Import Errors**: Run from project root or ensure `src/` is in Python path
- **API Errors**: Check API keys and rate limits
- **Memory Issues**: Reduce `number_of_questions` in config or use smaller models
- **Timeout Errors**: Increase timeout in `RunConfig` (LLMJudge initialization)

