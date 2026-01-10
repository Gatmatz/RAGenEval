import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for beautiful plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Define paths
output_dir = Path(__file__).parent.parent / "output" / "negative_rejection"
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)

# Load data from JSON files
models_data = {}

for json_file in output_dir.glob("negative_rejection_evaluation_*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
        # Extract model name from filename
        model_name = json_file.stem.replace("negative_rejection_evaluation_", "")
        # Format model name for better readability
        model_name = model_name.replace("-", " ").title()
        models_data[model_name] = data['aggregate_metrics']['accuracy']

# Sort models by name alphabetically
models_data = dict(sorted(models_data.items(), key=lambda x: x[0]))

# Extract data for plotting
models = list(models_data.keys())
accuracies = list(models_data.values())

# Create the barplot
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars with a beautiful color palette
colors = sns.color_palette("husl", len(models))
bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on top of each bar
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1%}',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

# Customize the plot
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Negative Rejection Accuracy Comparison Across Models',
             fontsize=16, fontweight='bold', pad=20)

# Set y-axis to percentage format
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add a subtle grid
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Add a reference line at 100% accuracy
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Accuracy')
ax.legend(loc='upper right', fontsize=11)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_path = plots_dir / "negative_rejection_accuracy_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Display the plot
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("NEGATIVE REJECTION ACCURACY SUMMARY")
print("="*60)
for model, accuracy in models_data.items():
    print(f"{model:30s}: {accuracy:6.2%}")
print("="*60)

