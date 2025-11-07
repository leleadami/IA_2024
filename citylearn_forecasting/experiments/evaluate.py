"""
Evaluation script for time series forecasting models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    LSTMForecaster,
    LSTMAutoencoder,
    LSTMAttention,
    LSTMEncoderDecoder,
    TimesNet
)
from data import TimeSeriesDataset
from utils import calculate_metrics, horizon_metrics, plot_predictions, plot_residuals
from experiments.train import load_config, create_model


def evaluate_model(
    model_path: str,
    model_type: str,
    config_path: str = 'configs/model_configs.yaml',
    data_path: str = None,
    output_dir: str = 'results'
):
    """
    Evaluate a trained forecasting model

    Args:
        model_path: Path to trained model checkpoint
        model_type: Type of model
        config_path: Path to configuration file
        data_path: Path to test data
        output_dir: Directory to save results
    """
    # Load configuration
    config = load_config(config_path)
    model_config = config['models'][model_type]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading test data...")
    if data_path is None:
        # Generate synthetic data
        print("No data path provided. Generating synthetic test data...")
        t = np.linspace(0, 50, 5000)
        data = np.sin(t) + 0.5 * np.sin(3 * t) + np.random.normal(0, 0.1, len(t))
        data = data.reshape(-1, 1)
    else:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.values
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            raise ValueError("Unsupported data format")

    # Create dataset
    dataset = TimeSeriesDataset(
        data=data,
        seq_len=model_config.get('seq_len', 96),
        pred_len=model_config.get('pred_len', 24),
        stride=1,
        scale=True
    )

    # Create data loader
    test_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print(f"Test size: {len(dataset)}")

    # Get input size
    sample_x, _ = dataset[0]
    input_size = sample_x.shape[-1]

    # Create model
    print(f"Loading {model_type} model...")
    model = create_model(model_type, model_config, input_size)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")

    # Make predictions
    print("\nMaking predictions...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            # Forward pass
            predictions = model(batch_x)

            # Handle different output shapes
            if len(predictions.shape) == 3 and len(batch_y.shape) == 2:
                predictions = predictions.squeeze(-1)
            elif len(predictions.shape) == 2 and len(batch_y.shape) == 1:
                predictions = predictions.squeeze(-1)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.numpy())

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Inverse transform if needed
    if hasattr(dataset, 'inverse_transform'):
        predictions = dataset.inverse_transform(predictions, target=True)
        targets = dataset.inverse_transform(targets, target=True)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(targets, predictions)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.6f}")
    print("="*50)

    # Calculate horizon-specific metrics if multi-step
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        print("\nCalculating horizon-specific metrics...")
        h_metrics = horizon_metrics(targets, predictions)

        print("\nMetrics per forecast horizon:")
        for step in range(len(h_metrics['mae_per_step'])):
            print(f"Step {step+1}: MAE={h_metrics['mae_per_step'][step]:.4f}, "
                  f"RMSE={h_metrics['rmse_per_step'][step]:.4f}, "
                  f"MAPE={h_metrics['mape_per_step'][step]:.4f}%")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to file
    metrics_file = os.path.join(output_dir, f'{model_type}_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name.upper()}: {value:.6f}\n")
        f.write("="*50 + "\n")

        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            f.write("\nMetrics per forecast horizon:\n")
            for step in range(len(h_metrics['mae_per_step'])):
                f.write(f"Step {step+1}: MAE={h_metrics['mae_per_step'][step]:.4f}, "
                       f"RMSE={h_metrics['rmse_per_step'][step]:.4f}, "
                       f"MAPE={h_metrics['mape_per_step'][step]:.4f}%\n")

    print(f"\nMetrics saved to: {metrics_file}")

    # Save predictions
    predictions_file = os.path.join(output_dir, f'{model_type}_predictions.npz')
    np.savez(
        predictions_file,
        predictions=predictions,
        targets=targets
    )
    print(f"Predictions saved to: {predictions_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Plot predictions vs ground truth (first 500 points)
    plot_predictions(
        y_true=targets.flatten()[:500],
        y_pred=predictions.flatten()[:500],
        title=f'{model_type.upper()} - Predictions vs Ground Truth',
        save_path=os.path.join(output_dir, f'{model_type}_predictions_plot.png')
    )

    # Plot residuals
    plot_residuals(
        y_true=targets,
        y_pred=predictions,
        save_path=os.path.join(output_dir, f'{model_type}_residuals.png')
    )

    print(f"\nVisualization saved to: {output_dir}")

    print("\nEvaluation completed!")

    return metrics, predictions, targets


def compare_models(
    model_paths: dict,
    config_path: str = 'configs/model_configs.yaml',
    data_path: str = None,
    output_dir: str = 'results'
):
    """
    Compare multiple models

    Args:
        model_paths: Dictionary mapping model types to checkpoint paths
        config_path: Path to configuration file
        data_path: Path to test data
        output_dir: Directory to save results
    """
    print("="*50)
    print("MODEL COMPARISON")
    print("="*50)

    all_metrics = {}
    all_predictions = {}

    for model_type, model_path in model_paths.items():
        print(f"\nEvaluating {model_type}...")
        metrics, predictions, targets = evaluate_model(
            model_path=model_path,
            model_type=model_type,
            config_path=config_path,
            data_path=data_path,
            output_dir=output_dir
        )

        all_metrics[model_type] = metrics
        all_predictions[model_type] = predictions

    # Create comparison table
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)

    comparison_df = pd.DataFrame(all_metrics).T
    print(comparison_df)

    # Save comparison
    comparison_file = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_file)
    print(f"\nComparison saved to: {comparison_file}")

    # Plot all models together
    from utils import plot_multiple_forecasts

    plot_multiple_forecasts(
        y_true=targets.flatten()[:500],
        forecasts={name: pred.flatten()[:500] for name, pred in all_predictions.items()},
        title='Model Comparison',
        save_path=os.path.join(output_dir, 'models_comparison_plot.png')
    )

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate time series forecasting model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['lstm', 'lstm_autoencoder', 'lstm_attention',
                               'lstm_encoder_decoder', 'timesnet'],
                       help='Model type')
    parser.add_argument('--config', type=str, default='configs/model_configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to test data file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        model_type=args.model_type,
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
