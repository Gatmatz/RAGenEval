import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Read Perfect Context data (from multiple seeds)
perfect_context_dir = Path("../output/perfect_context")
seed_dirs = [d for d in perfect_context_dir.iterdir() if d.is_dir() and d.name.startswith("seed")]
seed_dirs = sorted(seed_dirs)

print(f"Found {len(seed_dirs)} seeds for Perfect Context: {[d.name for d in seed_dirs]}")

# Collect Perfect Context data from all seeds
# Structure: pc_models_data[model_name][seed_name] = aggregate_metrics
pc_models_data = defaultdict(dict)

for seed_dir in seed_dirs:
    seed_name = seed_dir.name
    json_files = list(seed_dir.glob("perfect_context_evaluation_*.json"))

    for json_file in json_files:
        # Extract model name from filename
        model_name = json_file.stem.replace("perfect_context_evaluation_", "")

        with open(json_file, 'r') as f:
            data = json.load(f)
            pc_models_data[model_name][seed_name] = data['aggregate_metrics']

# Read Custom Retriever data
custom_retriever_dir = Path("../output/custom_retriever")
json_files = list(custom_retriever_dir.glob("custom_retriever_evaluation_*.json"))

cr_models_data = {}
for json_file in json_files:
    model_name = json_file.stem.replace("custom_retriever_evaluation_", "")

    with open(json_file, 'r') as f:
        data = json.load(f)
        cr_models_data[model_name] = data['aggregate_metrics']

# Get all model names (should be the same for both)
model_names = sorted(pc_models_data.keys())
print(f"\nFound models: {model_names}")

# Metrics to analyze
metrics = ['faithfulness', 'answer_relevancy', 'answer_correctness']
metric_display_names = {
    'faithfulness': 'Faithfulness',
    'answer_relevancy': 'Answer Relevancy',
    'answer_correctness': 'Answer Correctness'
}

# Calculate mean and std for Perfect Context across seeds
pc_aggregated_results = {}
for model_name in model_names:
    pc_aggregated_results[model_name] = {}
    for metric in metrics:
        values = [pc_models_data[model_name][seed_name][metric]
                 for seed_name in pc_models_data[model_name].keys()]
        pc_aggregated_results[model_name][metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

# Print summary
print("\n=== Perfect Context Results (Mean ± Std) ===")
for model_name in model_names:
    print(f"\n{model_name}:")
    for metric in metrics:
        mean = pc_aggregated_results[model_name][metric]['mean']
        std = pc_aggregated_results[model_name][metric]['std']
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")

print("\n=== Custom Retriever Results ===")
for model_name in model_names:
    if model_name in cr_models_data:
        print(f"\n{model_name}:")
        for metric in metrics:
            value = cr_models_data[model_name][metric]
            print(f"  {metric}: {value:.4f}")

# Group models by family for better visualization
model_families = {}
for model in model_names:
    if 'gemma' in model:
        family = 'Gemma'
    elif 'qwen' in model.lower():
        family = 'Qwen'
    elif 'gpt-oss' in model:
        family = 'GPT-OSS'
    else:
        family = 'Other'

    if family not in model_families:
        model_families[family] = []
    model_families[family].append(model)

# Sort families and models within families
sorted_families = sorted(model_families.keys())
sorted_model_names = []
for family in sorted_families:
    sorted_models = sorted(model_families[family])
    sorted_model_names.extend(sorted_models)

# Create comparison plot (vertical layout)
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Assign different colors to each family
family_colors = {
    'Gemma': '#FF6B6B',
    'GPT-OSS': '#4ECDC4',
    'Qwen': '#45B7D1',
    'Other': '#FFA07A'
}

# Create color list based on model family
colors = []
for model in sorted_model_names:
    for family, models in model_families.items():
        if model in models:
            colors.append(family_colors[family])
            break

# Simplify model names for display
def simplify_model_name(name):
    """Simplify model names for better readability"""
    name = name.replace('openai_gpt-oss-', 'GPT-OSS-')
    name = name.replace('gemma-3-', 'Gemma-')
    name = name.replace('qwen-3-', 'Qwen-')
    name = name.replace('qwen3:', 'Qwen-')
    name = name.replace('-it', '')
    return name

simplified_names = [simplify_model_name(name) for name in sorted_model_names]

# Width of bars and positions
bar_width = 0.35
x_pos = np.arange(len(sorted_model_names))

for idx, metric in enumerate(metrics):
    ax = axes[idx]  # Now accessing vertically stacked subplots

    # Get Perfect Context data
    pc_means = [pc_aggregated_results[model][metric]['mean'] for model in sorted_model_names]
    pc_stds = [pc_aggregated_results[model][metric]['std'] for model in sorted_model_names]

    # Get Custom Retriever data
    cr_values = [cr_models_data[model][metric] if model in cr_models_data else 0
                 for model in sorted_model_names]

    # Create grouped horizontal bars (Perfect Context on top, Simple Retriever on bottom)
    bars1 = ax.barh(x_pos - bar_width/2, pc_means, bar_width,
                    label='Perfect Context', color=colors, alpha=0.8,
                    edgecolor='black', linewidth=0.5)

    bars2 = ax.barh(x_pos + bar_width/2, cr_values, bar_width,
                    label='Simple Retriever', color=colors, alpha=0.4,
                    edgecolor='black', linewidth=0.5, hatch='//')

    # Customize subplot
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(metric_display_names[metric], fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(simplified_names, fontsize=10)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()  # Invert so first model is at top

    # Add value labels on bars
    for bar in bars1:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8, rotation=0)

    for bar in bars2:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8, rotation=0)

plt.tight_layout()

# Save plot
output_path = Path("plots/perfect_vs_custom_retriever_comparison.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"\n✓ Plot saved to {output_path}")


plt.show()

