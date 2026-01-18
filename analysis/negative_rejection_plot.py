import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Read JSON files
output_dir = Path("../output/negative_rejection")

# Find all seed directories
seed_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("bert_score_seed")]
seed_dirs = sorted(seed_dirs)

print(f"Found {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")

# Collect data from all seeds
# Structure: models_data[model_name][seed_name] = aggregate_metrics
models_data = defaultdict(dict)

for seed_dir in seed_dirs:
    seed_name = seed_dir.name
    json_files = list(seed_dir.glob("negative_rejection_evaluation_*.json"))

    for json_file in json_files:
        # Extract model name from filename
        model_name = json_file.stem.replace("negative_rejection_evaluation_", "")

        with open(json_file, 'r') as f:
            data = json.load(f)
            models_data[model_name][seed_name] = data['aggregate_metrics']

# Get all model names and metrics
model_names = sorted(models_data.keys())
print(f"\nFound models: {model_names}")

# Aggregate metrics to analyze
metrics = ['accuracy']

# Calculate mean and std for each metric across seeds
aggregated_results = {}
for model_name in model_names:
    aggregated_results[model_name] = {}
    for metric in metrics:
        values = [models_data[model_name][seed_name][metric]
                 for seed_name in models_data[model_name].keys()]
        aggregated_results[model_name][metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

# Print summary
print("\n=== Aggregated Results ===")
for model_name in model_names:
    print(f"\n{model_name}:")
    for metric in metrics:
        mean = aggregated_results[model_name][metric]['mean']
        std = aggregated_results[model_name][metric]['std']
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")

# Group models by family
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
family_boundaries = [0]
for family in sorted_families:
    sorted_models = sorted(model_families[family])
    sorted_model_names.extend(sorted_models)
    family_boundaries.append(len(sorted_model_names))

print(f"\nModel families: {sorted_families}")
for family in sorted_families:
    print(f"  {family}: {model_families[family]}")

# Create grouped bar chart organized by model family
fig, ax = plt.subplots(figsize=(14, 8))

# Assign different colors to each family
family_colors = {
    'Gemma': '#FF6B6B',
    'GPT-OSS': '#4ECDC4',
    'Qwen': '#45B7D1',
    'Other': '#FFA07A'
}

# Create color list based on model family
bar_colors = []
for model in sorted_model_names:
    for family, models in model_families.items():
        if model in models:
            bar_colors.append(family_colors[family])
            break

x = np.arange(len(sorted_model_names))
width = 0.5

for i, metric in enumerate(metrics):
    means = [aggregated_results[model][metric]['mean'] for model in sorted_model_names]
    stds = [aggregated_results[model][metric]['std'] for model in sorted_model_names]

    bars = ax.bar(x, means, width,
                  yerr=stds, capsize=4, color=bar_colors,
                  alpha=0.8, edgecolor='black', linewidth=1)

# Add vertical lines to separate model families
for boundary in family_boundaries[1:-1]:
    ax.axvline(x=boundary - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)

# Add family labels as shaded regions
for i, family in enumerate(sorted_families):
    start = family_boundaries[i]
    end = family_boundaries[i + 1]
    mid = (start + end - 1) / 2

    # Add subtle background shading for each family
    ax.axvspan(start - 0.5, end - 0.5, alpha=0.05, color=family_colors[family])

    # Add family label below the x-axis
    ax.text(mid, -0.28, family, ha='center', va='top',
            fontsize=13, fontweight='bold', transform=ax.get_xaxis_transform(),
            color=family_colors[family])

ax.set_xlabel('Model', fontsize=14, fontweight='bold', labelpad=50)
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Negative Rejection: Accuracy Comparison by Model Family',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(sorted_model_names, rotation=45, ha='right', fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots/negative_rejection_by_family.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: plots/negative_rejection_by_family.png")
plt.show()

print("\n✓ Plot generated successfully!")

