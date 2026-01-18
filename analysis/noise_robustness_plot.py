import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Read JSON files from noise_robustness folder
output_dir = Path("../output/noise_robustness")

print(f"Reading data from: {output_dir}")

# Collect data from all noise ratio folders
# Structure: models_data[model_name][noise_ratio] = metrics
models_data = defaultdict(dict)

# Get all noise ratio directories
noise_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('noise_ratio')]
noise_dirs = sorted(noise_dirs)

print(f"Found noise ratio folders: {[d.name for d in noise_dirs]}")

for noise_dir in noise_dirs:
    # Extract noise ratio from directory name (e.g., "noise_ratio_05" -> 0.5)
    dir_name = noise_dir.name
    if "05" in dir_name:
        noise_ratio = 0.5
    elif "08" in dir_name or "_08" in dir_name:
        noise_ratio = 0.8
    else:
        # Try to extract from directory name
        parts = dir_name.replace("noise_ratio_", "").replace("noise_ratio__", "")
        try:
            noise_ratio = float(parts) / 100 if len(parts) == 2 else float(parts)
        except:
            continue

    json_files = list(noise_dir.glob("noise_robustness_evaluation_*.json"))

    for json_file in json_files:
        # Extract model name from filename
        # Format: noise_robustness_evaluation_noise_0.5_model-name.json
        filename = json_file.stem
        parts = filename.split("_noise_")
        if len(parts) == 2:
            model_name = parts[1].split("_", 1)[1]  # Remove the noise value part
        else:
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)
            models_data[model_name][noise_ratio] = data['aggregate_metrics']

# Get all model names and noise ratios
model_names = sorted(models_data.keys())
noise_ratios = sorted(set(ratio for model in models_data.values() for ratio in model.keys()))

print(f"\nFound models: {model_names}")
print(f"Found noise ratios: {noise_ratios}")

# Print summary
print("\n=== Noise Robustness Results ===")
for model_name in model_names:
    print(f"\n{model_name}:")
    for noise_ratio in noise_ratios:
        if noise_ratio in models_data[model_name]:
            metrics = models_data[model_name][noise_ratio]
            print(f"  Noise {noise_ratio:.1f}:")
            print(f"    Faithfulness: {metrics['faithfulness']:.4f}")
            print(f"    Answer Relevancy: {metrics['answer_relevancy']:.4f}")
            print(f"    Answer Correctness: {metrics['answer_correctness']:.4f}")

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
for family in sorted_families:
    sorted_models = sorted(model_families[family])
    sorted_model_names.extend(sorted_models)

print(f"\nModel families: {sorted_families}")

# Create a comprehensive visualization with subplots for each metric
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
metrics = ['faithfulness', 'answer_relevancy', 'answer_correctness']
metric_titles = ['Faithfulness', 'Answer Relevancy', 'Answer Correctness']

# Color scheme for noise ratios
noise_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(noise_ratios)))

# Family colors for background shading
family_colors = {
    'Gemma': '#FF6B6B',
    'GPT-OSS': '#4ECDC4',
    'Qwen': '#45B7D1',
    'Other': '#FFA07A'
}

x = np.arange(len(sorted_model_names))
bar_width = 0.35

for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes[metric_idx]

    # Plot bars for each noise ratio
    for noise_idx, noise_ratio in enumerate(noise_ratios):
        values = []
        for model in sorted_model_names:
            if noise_ratio in models_data[model]:
                values.append(models_data[model][noise_ratio][metric])
            else:
                values.append(0.0)

        offset = (noise_idx - (len(noise_ratios) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                     label=f'Noise {noise_ratio:.1f}',
                     color=noise_colors[noise_idx],
                     alpha=0.8, edgecolor='black', linewidth=1)

    # Add family background shading
    current_x = 0
    for family in sorted_families:
        family_models = sorted(model_families[family])
        family_size = len(family_models)
        if family_size > 0:
            ax.axvspan(current_x - 0.5, current_x + family_size - 0.5,
                      alpha=0.05, color=family_colors[family])
            current_x += family_size

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if metric_idx == 2:  # Only show legend on the last subplot
        ax.legend(fontsize=10, loc='lower right', framealpha=0.9)

# Add overall title
fig.suptitle('Noise Robustness: Performance Comparison Across Models and Noise Ratios',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('plots/noise_robustness_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: plots/noise_robustness_comparison.png")

# Create a second plot: Line plot showing metric trends by noise ratio
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))

# Color scheme for models by family
model_color_map = {}
family_color_palette = {
    'Gemma': ['#FF6B6B', '#FF8E8E'],
    'GPT-OSS': ['#4ECDC4', '#70D8CE'],
    'Qwen': ['#45B7D1', '#6AC5DD'],
    'Other': ['#FFA07A', '#FFB894']
}

for family in sorted_families:
    family_models = sorted(model_families[family])
    colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 0.9, len(family_models)))
    for i, model in enumerate(family_models):
        # Use family-specific colors if available
        if family in family_color_palette and i < len(family_color_palette[family]):
            model_color_map[model] = family_color_palette[family][i]
        else:
            model_color_map[model] = colors[i]

for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes2[metric_idx]

    # Plot line for each model
    for model in sorted_model_names:
        noise_vals = []
        metric_vals = []
        for noise_ratio in noise_ratios:
            if noise_ratio in models_data[model]:
                noise_vals.append(noise_ratio)
                metric_vals.append(models_data[model][noise_ratio][metric])

        if noise_vals:
            ax.plot(noise_vals, metric_vals, marker='o', linewidth=2.5,
                   markersize=8, label=model, color=model_color_map[model],
                   alpha=0.8)

    ax.set_xlabel('Noise Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')

    if metric_idx == 2:  # Only show legend on the last subplot
        ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=1)

# Add overall title
fig2.suptitle('Noise Robustness: Metric Trends Across Noise Ratios',
              fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('plots/noise_robustness_trends.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to: plots/noise_robustness_trends.png")

plt.show()

print("\n✓ Plots generated successfully!")

