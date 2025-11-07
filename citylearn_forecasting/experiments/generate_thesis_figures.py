"""
Script per generare tutte le figure necessarie per la tesi
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str):
    """Load experiment results"""
    results_file = os.path.join(results_dir, 'all_results.json')

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def plot_metrics_comparison(results: dict, output_dir: str):
    """
    Figure 1: Bar plot comparing all metrics
    """
    metrics_to_plot = ['mae', 'rmse', 'mape', 'r2']
    model_names = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        values = [results[model]['metrics'][metric] for model in model_names]
        colors = sns.color_palette("husl", len(model_names))

        bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure1_metrics_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print("✓ Figure 1: Metrics comparison saved")


def plot_accuracy_vs_complexity(results: dict, output_dir: str):
    """
    Figure 2: Scatter plot - Accuracy vs Model Complexity
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    mae_values = []
    params = []

    for model_name, model_results in results.items():
        models.append(model_name)
        mae_values.append(model_results['metrics']['mae'])
        params.append(model_results['num_parameters'] / 1000)  # in thousands

    colors = sns.color_palette("husl", len(models))

    # Scatter plot
    for i, model in enumerate(models):
        ax.scatter(params[i], mae_values[i], s=200, c=[colors[i]],
                  alpha=0.7, edgecolors='black', linewidth=2, label=model)

    ax.set_xlabel('Number of Parameters (thousands)', fontweight='bold', fontsize=12)
    ax.set_ylabel('MAE (Mean Absolute Error)', fontweight='bold', fontsize=12)
    ax.set_title('Model Complexity vs Accuracy Trade-off', fontweight='bold', fontsize=14)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Annotate points
    for i, model in enumerate(models):
        ax.annotate(model, (params[i], mae_values[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure2_accuracy_vs_complexity.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print("✓ Figure 2: Accuracy vs Complexity saved")


def plot_training_time_comparison(results: dict, output_dir: str):
    """
    Figure 3: Bar plot - Training Time Comparison
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    training_times = [results[m]['training_time'] / 60 for m in models]  # in minutes

    colors = sns.color_palette("husl", len(models))
    bars = ax.barh(models, training_times, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.2f} min',
               ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Training Time (minutes)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Model', fontweight='bold', fontsize=12)
    ax.set_title('Training Time Comparison', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure3_training_time.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print("✓ Figure 3: Training time comparison saved")


def plot_radar_chart(results: dict, output_dir: str):
    """
    Figure 4: Radar chart for multi-metric comparison
    """
    from math import pi

    # Normalize metrics to 0-1 scale
    models = list(results.keys())
    metrics = ['mae', 'rmse', 'mape']

    # Get min/max for normalization
    min_max = {}
    for metric in metrics:
        values = [results[m]['metrics'][metric] for m in models]
        min_max[metric] = (min(values), max(values))

    # Normalize (lower is better, so invert)
    def normalize(value, metric):
        min_val, max_val = min_max[metric]
        if max_val == min_val:
            return 1.0
        # Invert: lower error = higher score
        return 1 - ((value - min_val) / (max_val - min_val))

    # Create radar chart
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = sns.color_palette("husl", len(models))

    for idx, model in enumerate(models):
        values = [normalize(results[model]['metrics'][m], m) for m in metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model,
               color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Model Comparison (Normalized)',
                fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure4_radar_chart.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print("✓ Figure 4: Radar chart saved")


def plot_pareto_frontier(results: dict, output_dir: str):
    """
    Figure 5: Pareto frontier - Accuracy vs Speed
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    mae_values = []
    training_times = []

    for model_name, model_results in results.items():
        models.append(model_name)
        mae_values.append(model_results['metrics']['mae'])
        training_times.append(model_results['training_time'] / 60)  # minutes

    colors = sns.color_palette("husl", len(models))

    # Scatter plot
    for i, model in enumerate(models):
        ax.scatter(training_times[i], mae_values[i], s=200, c=[colors[i]],
                  alpha=0.7, edgecolors='black', linewidth=2, label=model)

    ax.set_xlabel('Training Time (minutes)', fontweight='bold', fontsize=12)
    ax.set_ylabel('MAE (Mean Absolute Error)', fontweight='bold', fontsize=12)
    ax.set_title('Pareto Frontier: Accuracy vs Training Time', fontweight='bold', fontsize=14)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Annotate
    for i, model in enumerate(models):
        ax.annotate(model, (training_times[i], mae_values[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure5_pareto_frontier.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print("✓ Figure 5: Pareto frontier saved")


def plot_model_summary_table(results: dict, output_dir: str):
    """
    Figure 6: Summary table as image
    """
    data = []
    for model, res in results.items():
        data.append([
            model,
            f"{res['metrics']['mae']:.4f}",
            f"{res['metrics']['rmse']:.4f}",
            f"{res['metrics']['r2']:.4f}",
            f"{res['num_parameters']:,}",
            f"{res['training_time']/60:.1f}"
        ])

    df = pd.DataFrame(data, columns=['Model', 'MAE', 'RMSE', 'R²', 'Parameters', 'Time (min)'])

    fig, ax = plt.subplots(figsize=(12, len(data) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15, 0.18, 0.17])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Model Comparison Summary', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure6_summary_table.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print("✓ Figure 6: Summary table saved")


def generate_all_figures(results_dir: str, output_dir: str = None):
    """
    Generate all thesis figures
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'figures')

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING THESIS FIGURES")
    print("="*70 + "\n")

    # Load results
    print("Loading results...")
    results = load_results(results_dir)
    print(f"✓ Loaded results for {len(results)} models\n")

    # Generate figures
    print("Generating figures...\n")

    plot_metrics_comparison(results, output_dir)
    plot_accuracy_vs_complexity(results, output_dir)
    plot_training_time_comparison(results, output_dir)
    plot_radar_chart(results, output_dir)
    plot_pareto_frontier(results, output_dir)
    plot_model_summary_table(results, output_dir)

    print("\n" + "="*70)
    print("ALL FIGURES GENERATED!")
    print("="*70)
    print(f"\nFigures saved in: {output_dir}/")
    print("\nGenerated figures:")
    print("  1. figure1_metrics_comparison.png")
    print("  2. figure2_accuracy_vs_complexity.png")
    print("  3. figure3_training_time.png")
    print("  4. figure4_radar_chart.png")
    print("  5. figure5_pareto_frontier.png")
    print("  6. figure6_summary_table.png")
    print("\n" + "="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate thesis figures')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory with experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for figures')

    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
