import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Read JSON files from no_instruction folder
output_dir = Path("../output/counterfactual_robustness/no_instruction")

print(f"Reading data from: {output_dir}")

# Collect data from no_instruction folder
models_data = {}

json_files = list(output_dir.glob("counterfactual_robustness_evaluation_*.json"))

for json_file in json_files:
    # Extract model name from filename
    model_name = json_file.stem.replace("counterfactual_robustness_evaluation_", "")

    with open(json_file, 'r') as f:
        data = json.load(f)

        # Calculate BERTScore metrics for all cases
        all_results = data['individual_results']

        if len(all_results) > 0:
            avg_true_bert = np.mean([r['true_answer_bert_score'] for r in all_results])
            avg_fake_bert = np.mean([r['fake_answer_bert_score'] for r in all_results])
        else:
            avg_true_bert = 0.0
            avg_fake_bert = 0.0

        models_data[model_name] = {
            'average_true_answer_bert_score': avg_true_bert,
            'average_fake_answer_bert_score': avg_fake_bert,
            'total_count': len(all_results)
        }

# Get all model names
model_names = sorted(models_data.keys())
print(f"\nFound models: {model_names}")

# Print summary
print("\n=== No Instruction Results (All Cases) ===")
for model_name in model_names:
    avg_true = models_data[model_name]['average_true_answer_bert_score']
    avg_fake = models_data[model_name]['average_fake_answer_bert_score']
    count = models_data[model_name]['total_count']
    print(f"\n{model_name}:")
    print(f"  Total cases: {count}")
    print(f"  Average True Answer BERTScore: {avg_true:.4f}")
    print(f"  Average Fake Answer BERTScore: {avg_fake:.4f}")

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

# Metric colors
metric_colors = ['#4ECDC4', '#45B7D1']  # Teal shades for true and fake
metrics = ['average_true_answer_bert_score', 'average_fake_answer_bert_score']
metric_labels = ['Average True Answer BERTScore', 'Average Fake Answer BERTScore']

x = np.arange(len(sorted_model_names))
width = 0.35
offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2

for i, metric in enumerate(metrics):
    means = [models_data[model][metric] for model in sorted_model_names]

    bars = ax.bar(x + offsets[i] * width, means, width,
                  label=metric_labels[i],
                  color=metric_colors[i],
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
ax.set_ylabel('BERTScore', fontsize=14, fontweight='bold')
ax.set_title('Counterfactual Robustness (No Instruction): BERTScore Comparison by Model Family',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(sorted_model_names, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots/counterfactual_no_instruction_by_family.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: plots/counterfactual_no_instruction_by_family.png")
plt.show()

print("\n✓ Plot generated successfully!")

