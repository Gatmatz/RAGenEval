"""
Robustness Tests Plot for RAG Study
Creates a focused visualization of the three robustness experiments
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style for publication quality
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# Define model families and colors
def get_model_family(model_name):
    if 'gemma' in model_name.lower():
        return 'Gemma'
    elif 'qwen' in model_name.lower():
        return 'Qwen'
    elif 'gpt-oss' in model_name.lower():
        return 'GPT-OSS'
    else:
        return 'Other'

FAMILY_COLORS = {
    'Gemma': '#FF6B6B',
    'GPT-OSS': '#4ECDC4',
    'Qwen': '#45B7D1',
    'Other': '#FFA07A'
}

# Simplified model names
def simplify_model_name(model_name):
    """Create shorter, cleaner model names"""
    replacements = {
        'gemma-3-27b-it': 'Gemma-27B',
        'gemma-3-4b-it': 'Gemma-4B',
        'qwen-3-32b': 'Qwen-32B',
        'qwen3:0.6b': 'Qwen-0.6B',
        'openai_gpt-oss-120b': 'GPT-120B',
        'openai_gpt-oss-20b': 'GPT-20B'
    }
    return replacements.get(model_name, model_name)

# ============================================================================
# Data Loading Functions
# ============================================================================
def load_perfect_context_data():
    output_dir = Path("../output/perfect_context")
    seed_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("seed")]

    models_data = defaultdict(dict)
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        json_files = list(seed_dir.glob("perfect_context_evaluation_*.json"))

        for json_file in json_files:
            model_name = json_file.stem.replace("perfect_context_evaluation_", "")
            with open(json_file, 'r') as f:
                data = json.load(f)
                models_data[model_name][seed_name] = data['aggregate_metrics']

    results = {}
    metrics = ['faithfulness', 'answer_relevancy', 'answer_correctness']
    for model_name in models_data.keys():
        results[model_name] = {}
        for metric in metrics:
            values = [models_data[model_name][seed][metric]
                     for seed in models_data[model_name].keys()]
            results[model_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    return results

def load_noise_robustness_data():
    output_dir = Path("../output/noise_robustness")
    models_data = defaultdict(dict)

    noise_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('noise_ratio')]

    for noise_dir in noise_dirs:
        dir_name = noise_dir.name
        if "05" in dir_name:
            noise_ratio = 0.5
        elif "08" in dir_name or "_08" in dir_name:
            noise_ratio = 0.8
        else:
            continue

        json_files = list(noise_dir.glob("noise_robustness_evaluation_*.json"))

        for json_file in json_files:
            filename = json_file.stem
            parts = filename.split("_noise_")
            if len(parts) == 2:
                model_name = parts[1].split("_", 1)[1]
            else:
                continue

            with open(json_file, 'r') as f:
                data = json.load(f)
                models_data[model_name][noise_ratio] = data['aggregate_metrics']

    return models_data

def load_counterfactual_data():
    output_dir = Path("../output/counterfactual_robustness")
    run_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("bert_scores_run")]

    models_data = defaultdict(dict)
    for run_dir in run_dirs:
        run_name = run_dir.name
        json_files = list(run_dir.glob("counterfactual_robustness_evaluation_*.json"))

        for json_file in json_files:
            model_name = json_file.stem.replace("counterfactual_robustness_evaluation_", "")

            with open(json_file, 'r') as f:
                data = json.load(f)
                accuracy = data['aggregate_metrics']['accuracy']
                models_data[model_name][run_name] = {'accuracy': accuracy}

    results = {}
    for model_name in models_data.keys():
        values = [models_data[model_name][run]['accuracy']
                 for run in models_data[model_name].keys()]
        results[model_name] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    return results

def load_negative_rejection_data():
    output_dir = Path("../output/negative_rejection")
    seed_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("bert_score_seed")]

    models_data = defaultdict(dict)
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        json_files = list(seed_dir.glob("negative_rejection_evaluation_*.json"))

        for json_file in json_files:
            model_name = json_file.stem.replace("negative_rejection_evaluation_", "")
            with open(json_file, 'r') as f:
                data = json.load(f)
                models_data[model_name][seed_name] = data['aggregate_metrics']

    results = {}
    for model_name in models_data.keys():
        values = [models_data[model_name][seed]['accuracy']
                 for seed in models_data[model_name].keys()]
        results[model_name] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    return results

# ============================================================================
# Create Robustness Tests Visualization
# ============================================================================
def create_robustness_plot():
    # Load all data
    print("Loading data from all experiments...")
    perfect_context = load_perfect_context_data()
    noise_robustness = load_noise_robustness_data()
    counterfactual = load_counterfactual_data()
    negative_rejection = load_negative_rejection_data()

    # Get common models across all experiments
    all_models = set(perfect_context.keys()) & set(noise_robustness.keys()) & \
                 set(counterfactual.keys()) & set(negative_rejection.keys())
    all_models = sorted(list(all_models))

    print(f"Found {len(all_models)} models with data across all experiments")

    # Sort models by family and parameter size
    def get_model_size(model):
        """Extract parameter size for sorting"""
        if '120b' in model.lower():
            return 120
        elif '32b' in model.lower():
            return 32
        elif '27b' in model.lower():
            return 27
        elif '20b' in model.lower():
            return 20
        elif '4b' in model.lower():
            return 4
        elif '0.6b' in model.lower():
            return 0.6
        return 0

    # Group and sort by family then size
    model_families = {}
    for model in all_models:
        family = get_model_family(model)
        if family not in model_families:
            model_families[family] = []
        model_families[family].append(model)

    sorted_families = sorted(model_families.keys())
    sorted_models = []
    for family in sorted_families:
        sorted_models.extend(sorted(model_families[family], key=get_model_size))

    # Create simplified names
    simple_names = [simplify_model_name(m) for m in sorted_models]
    colors = [FAMILY_COLORS[get_model_family(model)] for model in sorted_models]

    # ========================================================================
    # Create the plot: 3x1 grid with robustness tests
    # ========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(8, 14))

    x = np.arange(len(sorted_models))
    bar_width = 0.6

    # Plot 1: Noise Robustness
    # ----------------------------------------------
    ax = axes[0]
    width = 0.2

    # Baseline (no noise)
    baseline = [perfect_context[m]['answer_correctness']['mean'] for m in sorted_models]
    ax.bar(x - width, baseline, width, label='No Noise',
           color='lightgray', alpha=0.7, edgecolor='black', linewidth=0.8)

    # 50% noise
    noise_50 = []
    for m in sorted_models:
        if 0.5 in noise_robustness[m]:
            noise_50.append(noise_robustness[m][0.5]['answer_correctness'])
        else:
            noise_50.append(0)
    ax.bar(x, noise_50, width, label='50% Noise',
           alpha=0.7, edgecolor='black', linewidth=0.8)
    for bar, model in zip(ax.containers[-1], sorted_models):
        bar.set_color(FAMILY_COLORS[get_model_family(model)])

    # 80% noise
    noise_80 = []
    for m in sorted_models:
        if 0.8 in noise_robustness[m]:
            noise_80.append(noise_robustness[m][0.8]['answer_correctness'])
        else:
            noise_80.append(0)
    ax.bar(x + width, noise_80, width, label='80% Noise',
           alpha=0.9, edgecolor='black', linewidth=0.8)
    for bar, model in zip(ax.containers[-1], sorted_models):
        bar.set_color(FAMILY_COLORS[get_model_family(model)])

    ax.set_title('Noise Robustness\nAnswer Correctness', fontweight='bold', pad=10)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(simple_names, rotation=45, ha='right')
    ax.set_ylim([0.7, 1.0])
    ax.legend(loc='lower left', frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Counterfactual Robustness
    # ----------------------------------------------
    ax = axes[1]
    means = [counterfactual[m]['mean'] for m in sorted_models]
    stds = [counterfactual[m]['std'] for m in sorted_models]
    bars = ax.bar(x, means, bar_width, yerr=stds, color=colors, alpha=0.85,
                 capsize=4, edgecolor='black', linewidth=1.2)
    ax.set_title('Counterfactual Detection\nAccuracy', fontweight='bold', pad=10)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(simple_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Removed the random line here

    # Plot 3: Negative Rejection
    # ----------------------------------------------
    ax = axes[2]
    means = [negative_rejection[m]['mean'] for m in sorted_models]
    stds = [negative_rejection[m]['std'] for m in sorted_models]
    bars = ax.bar(x, means, bar_width, yerr=stds, color=colors, alpha=0.85,
                 capsize=4, edgecolor='black', linewidth=1.2)
    ax.set_title('Negative Rejection\nAccuracy', fontweight='bold', pad=10)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(simple_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add family legend at the bottom
    legend_elements = [mpatches.Patch(facecolor=FAMILY_COLORS[family],
                                     edgecolor='black',
                                     label=f'{family} Family',
                                     alpha=0.85)
                      for family in sorted_families]

    fig.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.01), ncol=len(sorted_families),
              frameon=True, fancybox=True, fontsize=12, title='Model Families',
              title_fontsize=13)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 1])

    # Save high-res version for poster
    output_path_large = Path("../analysis/plots/robustness_tests_large.png")
    plt.savefig(output_path_large, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"High-res version saved to: {output_path_large}")

    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print("ROBUSTNESS TESTS SUMMARY")
    print("="*80)

    for model in sorted_models:
        simple_name = simplify_model_name(model)
        family = get_model_family(model)
        print(f"\n{simple_name} ({family}):")

        # Noise robustness
        print(f"  Noise Robustness (Answer Correctness):")
        print(f"    No Noise:  {perfect_context[model]['answer_correctness']['mean']:.3f}")
        if 0.5 in noise_robustness[model]:
            print(f"    50% Noise: {noise_robustness[model][0.5]['answer_correctness']:.3f}")
        if 0.8 in noise_robustness[model]:
            print(f"    80% Noise: {noise_robustness[model][0.8]['answer_correctness']:.3f}")

        # Counterfactual detection
        print(f"  Counterfactual Detection: {counterfactual[model]['mean']:.3f} ± {counterfactual[model]['std']:.3f}")

        # Negative rejection
        print(f"  Negative Rejection:       {negative_rejection[model]['mean']:.3f} ± {negative_rejection[model]['std']:.3f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    print("="*80)
    print("Creating Robustness Tests Plot")
    print("="*80)
    print("\nThis creates a visualization of the three robustness experiments:")
    print("  1. Noise Robustness")
    print("  2. Counterfactual Detection")
    print("  3. Negative Rejection")
    print()

    create_robustness_plot()

    print("\n✓ Plot generation complete!")
    print("\nFile created:")
    print("  • robustness_tests_large.png (600 DPI, high-quality for posters)")

