import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read JSON files
output_dir = Path("../output")
retriever_scenario = "perfect_context"

# Find all JSON files matching the pattern
json_files = list(output_dir.glob(f"{retriever_scenario}/{retriever_scenario}_evaluation_*.json"))

# Extract model names and data
models_data = {}
for json_file in json_files:
    # Extract model name from filename
    model_name = json_file.stem.replace(f"{retriever_scenario}_evaluation_", "")

    with open(json_file, 'r') as f:
        models_data[model_name] = json.load(f)

# Extract aggregate metrics
metrics = ['faithfulness', 'answer_relevancy', 'answer_correctness']
model_names = list(models_data.keys())
n_models = len(model_names)

# Prepare data for plotting
metrics_values = {metric: [] for metric in metrics}
for model_name in model_names:
    for metric in metrics:
        metrics_values[metric].append(models_data[model_name]['aggregate_metrics'][metric])

# Create barplot
x = np.arange(len(metrics))
width = 0.8 / n_models  # Adjust width based on number of models

fig, ax = plt.subplots(figsize=(12, 6))

bars = []
for i, model_name in enumerate(model_names):
    values = [metrics_values[metric][i] for metric in metrics]
    offset = (i - n_models / 2 + 0.5) * width
    bar = ax.bar(x + offset, values, width, label=model_name)
    bars.append(bar)

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title(f'Model Performance Comparison on {retriever_scenario.replace("_", " ").title()}')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'plots/{retriever_scenario}_metrics_comparison.png', dpi=300)
plt.show()
