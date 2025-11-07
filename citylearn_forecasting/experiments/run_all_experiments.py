"""
Script per eseguire tutti gli esperimenti per la tesi

Questo script:
1. Train tutti i 5 modelli
2. Evalua su test set
3. Salva metriche e grafici
4. Crea tabelle comparative
5. Genera material per la tesi
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train import train_model
from experiments.evaluate import evaluate_model


def run_full_experiment(
    data_path: str,
    output_dir: str = 'thesis_results',
    models_to_test: list = None
):
    """
    Esegue esperimenti completi per tutti i modelli

    Args:
        data_path: Path al dataset CityLearn
        output_dir: Directory per salvare risultati
        models_to_test: Lista di modelli da testare (default: tutti)
    """

    if models_to_test is None:
        models_to_test = [
            'lstm',
            'lstm_attention',
            'lstm_encoder_decoder',
            'lstm_autoencoder',
            'timesnet'
        ]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    results = {}

    print("="*70)
    print("RUNNING COMPLETE EXPERIMENTS FOR THESIS")
    print("="*70)
    print(f"\nModels to test: {models_to_test}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print("\n" + "="*70 + "\n")

    # Train and evaluate each model
    for i, model_type in enumerate(models_to_test, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(models_to_test)}: {model_type.upper()}")
        print(f"{'='*70}\n")

        try:
            # Train
            print(f"[1/3] Training {model_type}...")
            start_time = time.time()

            model, history = train_model(
                model_type=model_type,
                data_path=data_path,
                save_dir=checkpoints_dir
            )

            training_time = time.time() - start_time

            # Evaluate
            print(f"\n[2/3] Evaluating {model_type}...")
            model_path = os.path.join(checkpoints_dir, f'{model_type}_best.pth')

            metrics, predictions, targets = evaluate_model(
                model_path=model_path,
                model_type=model_type,
                data_path=data_path,
                output_dir=os.path.join(output_dir, model_type)
            )

            # Store results
            results[model_type] = {
                'metrics': metrics,
                'training_time': training_time,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'history': {
                    'train_losses': history['train_losses'],
                    'val_losses': history['val_losses']
                }
            }

            print(f"\n[3/3] {model_type} completed successfully!")
            print(f"Training time: {training_time/60:.2f} minutes")
            print(f"Test MAE: {metrics['mae']:.4f}")
            print(f"Test RMSE: {metrics['rmse']:.4f}")

        except Exception as e:
            print(f"\n❌ ERROR with {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = {'error': str(e)}

        print(f"\n{'='*70}\n")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")

    # Save as JSON
    results_file = os.path.join(output_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for model, res in results.items():
            if 'error' not in res:
                json_results[model] = {
                    'metrics': {k: float(v) for k, v in res['metrics'].items()},
                    'training_time': float(res['training_time']),
                    'num_parameters': int(res['num_parameters'])
                }
            else:
                json_results[model] = res

        json.dump(json_results, f, indent=2)

    print(f"✓ Results saved to {results_file}")

    # Create comparison table
    create_comparison_table(results, output_dir)

    # Create summary report
    create_summary_report(results, output_dir, data_path)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("Check:")
    print(f"  - {output_dir}/comparison_table.csv")
    print(f"  - {output_dir}/summary_report.txt")
    print(f"  - {output_dir}/all_results.json")
    print("\n" + "="*70 + "\n")

    return results


def create_comparison_table(results: dict, output_dir: str):
    """Create comparison table for thesis"""

    data = []

    for model_name, model_results in results.items():
        if 'error' in model_results:
            continue

        metrics = model_results['metrics']
        row = {
            'Model': model_name,
            'MAE': f"{metrics['mae']:.4f}",
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAPE (%)': f"{metrics['mape']:.2f}",
            'R²': f"{metrics['r2']:.4f}",
            'Training Time (min)': f"{model_results['training_time']/60:.2f}",
            'Parameters': f"{model_results['num_parameters']:,}"
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Save as CSV
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)

    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'comparison_table.tex')
    latex_table = df.to_latex(index=False, caption='Model Comparison on CityLearn Dataset',
                               label='tab:model_comparison')
    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"✓ Comparison table saved to {csv_path}")
    print(f"✓ LaTeX table saved to {latex_path}")

    # Print to console
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")


def create_summary_report(results: dict, output_dir: str, data_path: str):
    """Create summary report for thesis"""

    report_path = os.path.join(output_dir, 'summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("THESIS EXPERIMENTS - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Dataset: {data_path}\n")
        f.write(f"Number of models tested: {len([r for r in results.values() if 'error' not in r])}\n\n")

        # Best model per metric
        f.write("="*70 + "\n")
        f.write("BEST MODELS PER METRIC\n")
        f.write("="*70 + "\n\n")

        # Get best for each metric
        metrics_to_check = ['mae', 'rmse', 'mape', 'r2']
        best_models = {}

        for metric in metrics_to_check:
            valid_results = {k: v for k, v in results.items() if 'error' not in v}

            if metric == 'r2':
                # Higher is better
                best_model = max(valid_results.items(),
                               key=lambda x: x[1]['metrics'][metric])
            else:
                # Lower is better
                best_model = min(valid_results.items(),
                               key=lambda x: x[1]['metrics'][metric])

            best_models[metric] = best_model
            f.write(f"{metric.upper()}: {best_model[0]} ({best_model[1]['metrics'][metric]:.4f})\n")

        # Fastest training
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        fastest = min(valid_results.items(), key=lambda x: x[1]['training_time'])
        f.write(f"\nFastest Training: {fastest[0]} ({fastest[1]['training_time']/60:.2f} min)\n")

        # Smallest model
        smallest = min(valid_results.items(), key=lambda x: x[1]['num_parameters'])
        f.write(f"Smallest Model: {smallest[0]} ({smallest[1]['num_parameters']:,} params)\n")

        # Detailed results
        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*70 + "\n\n")

        for model_name, model_results in results.items():
            if 'error' in model_results:
                f.write(f"{model_name.upper()}: ERROR - {model_results['error']}\n\n")
                continue

            f.write(f"{model_name.upper()}\n")
            f.write("-"*70 + "\n")
            f.write(f"Parameters: {model_results['num_parameters']:,}\n")
            f.write(f"Training Time: {model_results['training_time']/60:.2f} minutes\n")
            f.write(f"\nMetrics:\n")
            for metric, value in model_results['metrics'].items():
                f.write(f"  {metric.upper()}: {value:.6f}\n")
            f.write("\n")

        # Recommendations
        f.write("="*70 + "\n")
        f.write("RECOMMENDATIONS FOR THESIS\n")
        f.write("="*70 + "\n\n")

        f.write("Best Overall Model (accuracy): ")
        best_overall = min(valid_results.items(), key=lambda x: x[1]['metrics']['mae'])
        f.write(f"{best_overall[0]}\n\n")

        f.write("Best for Real-Time Applications (speed): ")
        f.write(f"{fastest[0]}\n\n")

        f.write("Best Trade-off (accuracy/complexity): ")
        # Calculate score: MAE * (params / 100000)
        scored = [(k, v['metrics']['mae'] * (v['num_parameters'] / 100000))
                  for k, v in valid_results.items()]
        best_tradeoff = min(scored, key=lambda x: x[1])
        f.write(f"{best_tradeoff[0]}\n\n")

        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")

    print(f"✓ Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run all experiments for thesis')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CityLearn data file')
    parser.add_argument('--output', type=str, default='thesis_results',
                       help='Output directory for results')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['lstm', 'lstm_attention', 'lstm_encoder_decoder',
                               'lstm_autoencoder', 'timesnet'],
                       help='Models to test (default: all)')

    args = parser.parse_args()

    run_full_experiment(
        data_path=args.data,
        output_dir=args.output,
        models_to_test=args.models
    )


if __name__ == '__main__':
    main()
