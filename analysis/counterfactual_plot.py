import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Read JSON files
output_dir = Path("../output/counterfactual_robustness")

# Find all run directories
run_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("bert_scores_run")]
run_dirs = sorted(run_dirs)

print(f"Found {len(run_dirs)} runs: {[d.name for d in run_dirs]}")

# Collect data from all runs
# Structure: models_data[model_name][run_name] = custom_metrics
models_data = defaultdict(dict)

for run_dir in run_dirs:
    run_name = run_dir.name
    json_files = list(run_dir.glob("counterfactual_robustness_evaluation_*.json"))

    for json_file in json_files:
        # Extract model name from filename
        model_name = json_file.stem.replace("counterfactual_robustness_evaluation_", "")

        with open(json_file, 'r') as f:
            data = json.load(f)

            # Keep accuracy as is from aggregate_metrics
            accuracy = data['aggregate_metrics']['accuracy']

            # Calculate BERTScore metrics only for cases where detection is false
            false_detections = [r for r in data['individual_results'] if not r['is_correct_detection']]

            if len(false_detections) > 0:
                avg_true_bert = np.mean([r['true_answer_bert_score'] for r in false_detections])
                avg_fake_bert = np.mean([r['fake_answer_bert_score'] for r in false_detections])
            else:
                # If no false detections, use NaN or 0
                avg_true_bert = 0.0
                avg_fake_bert = 0.0

            models_data[model_name][run_name] = {
                'accuracy': accuracy,
                'average_true_answer_bert_score': avg_true_bert,
                'average_fake_answer_bert_score': avg_fake_bert
            }

# Get all model names and metrics
model_names = sorted(models_data.keys())
print(f"\nFound models: {model_names}")

# Aggregate metrics to analyze
metrics = ['accuracy', 'average_true_answer_bert_score', 'average_fake_answer_bert_score']

# Calculate mean and std for each metric across runs
aggregated_results = {}
for model_name in model_names:
    aggregated_results[model_name] = {}
    for metric in metrics:
        values = [models_data[model_name][run_name][metric]
                 for run_name in models_data[model_name].keys()]
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
    elif 'qwen' in model:
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
fig, ax = plt.subplots(figsize=(16, 8))

# Assign different colors to each family
family_colors = {
    'Gemma': ['#FF6B6B', '#E85555', '#D14444'],
    'GPT-OSS': ['#4ECDC4', '#3DB8AF', '#2CA39A'],
    'Qwen': ['#45B7D1', '#34A6C0', '#2395AF'],
    'Other': ['#FFA07A', '#FF8F69', '#FF7E58']
}

# Metric colors (keep for legend)
metric_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

x = np.arange(len(sorted_model_names))
width = 0.22
offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2

for i, metric in enumerate(metrics):
    means = [aggregated_results[model][metric]['mean'] for model in sorted_model_names]
    stds = [aggregated_results[model][metric]['std'] for model in sorted_model_names]

    bars = ax.bar(x + offsets[i] * width, means, width,
                  label=metric.replace('_', ' ').title(),
                  yerr=stds, capsize=3, color=metric_colors[i],
                  alpha=0.8, edgecolor='black', linewidth=1)

# Add vertical lines to separate model families
for boundary in family_boundaries[1:-1]:
    ax.axvline(x=boundary - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)

# Add family labels as shaded regions
family_main_colors = {
    'Gemma': '#FF6B6B',
    'GPT-OSS': '#4ECDC4',
    'Qwen': '#45B7D1',
    'Other': '#FFA07A'
}

for i, family in enumerate(sorted_families):
    start = family_boundaries[i]
    end = family_boundaries[i + 1]
    mid = (start + end - 1) / 2

    # Add subtle background shading for each family
    ax.axvspan(start - 0.5, end - 0.5, alpha=0.05, color=family_main_colors[family])

    # Add family label below the x-axis
    ax.text(mid, -0.28, family, ha='center', va='top',
            fontsize=13, fontweight='bold', transform=ax.get_xaxis_transform(),
            color=family_main_colors[family])

ax.set_xlabel('Model', fontsize=14, fontweight='bold', labelpad=50)
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Counterfactual Robustness: Metrics Comparison by Model Family\n(BERTScores calculated only for false detections)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(sorted_model_names, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots/counterfactual_robustness_by_family.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: plots/counterfactual_robustness_by_family.png")
plt.show()

print("\n✓ Plot generated successfully!")

