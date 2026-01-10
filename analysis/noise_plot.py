import json
import matplotlib.pyplot as plt
from pathlib import Path

# Read JSON files
output_dir = Path("../output/noise_robustness")

# Find all JSON files
json_files = list(output_dir.glob("noise_robustness_evaluation_*.json"))

# Organize data by model and noise ratio
models_data = {}
for json_file in json_files:
    # Extract model name and noise ratio from filename
    # Format: noise_robustness_evaluation_{model}_noise_{ratio}.json
    filename = json_file.stem
    parts = filename.replace("noise_robustness_evaluation_", "").split("_noise_")
    model_name = parts[0]
    noise_ratio = float(parts[1])

    if model_name not in models_data:
        models_data[model_name] = {}

    with open(json_file, 'r') as f:
        data = json.load(f)
        models_data[model_name][noise_ratio] = data['aggregate_metrics']

# Metrics to plot
metrics = ['faithfulness', 'answer_relevancy', 'answer_correctness']
metric_labels = {
    'faithfulness': 'Faithfulness',
    'answer_relevancy': 'Answer Relevancy',
    'answer_correctness': 'Answer Correctness'
}

# Group models by family
def get_model_family(model_name):
    """Extract model family from model name (e.g., 'llama' from 'llama-3.1-8b-instant')"""
    return model_name.split('-')[0]

model_families = {}
for model_name in models_data.keys():
    family = get_model_family(model_name)
    if family not in model_families:
        model_families[family] = []
    model_families[family].append(model_name)

# Sort families and models within each family
sorted_families = sorted(model_families.keys())
for family in sorted_families:
    model_families[family] = sorted(model_families[family])

# Colors for each metric
metric_colors = {
    'faithfulness': '#2E86AB',
    'answer_relevancy': '#A23B72',
    'answer_correctness': '#F18F01'
}

# Markers for each metric
metric_markers = {
    'faithfulness': 'o',
    'answer_relevancy': 's',
    'answer_correctness': '^'
}

# Line widths for different model versions within a family (bold and light)
model_line_widths = {
    0: 3.5,      # bold (thicker line)
    1: 1.5,      # light (thinner line)
    2: 3.5,      # bold
    3: 1.5       # light
}

# Create figure with subplots - one for each metric
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Get all noise ratios
all_noise_ratios = sorted(set(ratio for model_data in models_data.values() for ratio in model_data.keys()))

# Plot each metric
for metric_idx, metric in enumerate(metrics):
    ax = axes[metric_idx]

    # Plot each model family
    for family in sorted_families:
        models_in_family = model_families[family]

        for model_idx, model_name in enumerate(models_in_family):
            noise_ratios = sorted(models_data[model_name].keys())
            values = [models_data[model_name][ratio][metric] for ratio in noise_ratios]

            # Create label with model version info
            # Extract size info (e.g., "8b" or "70b")
            size = None
            for part in model_name.split('-'):
                if 'b' in part and any(c.isdigit() for c in part):
                    size = part
                    break

            label = f"{family.upper()} ({size})" if size else model_name

            # Use metric color but different line width for different models in same family
            ax.plot(noise_ratios, values,
                    marker=metric_markers[metric],
                    linestyle='-',
                    linewidth=model_line_widths.get(model_idx, 2.5),
                    markersize=8,
                    label=label,
                    color=metric_colors[metric],
                    alpha=1.0)  # Full opacity for clarity

    # Customize subplot
    ax.set_xlabel('Noise Ratio', fontsize=13, fontweight='bold')
    ax.set_title(metric_labels[metric], fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(all_noise_ratios)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)

    # Only add y-label to the leftmost plot
    if metric_idx == 0:
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')

    # Add legend
    ax.legend(loc='best', framealpha=0.95, fontsize=10)

# Overall title
fig.suptitle('Impact of Noise Ratio on RAG Metrics by Model Family',
             fontsize=17, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('plots/noise_robustness_by_family.png', dpi=300, bbox_inches='tight')
print("Plot saved to: plots/noise_robustness_by_family.png")
plt.show()



