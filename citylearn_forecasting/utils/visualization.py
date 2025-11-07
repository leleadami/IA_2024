"""
Visualization utilities for time series forecasting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Optional, List, Dict


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Predictions vs Ground Truth',
    figsize: tuple = (15, 6),
    save_path: Optional[str] = None,
    show_metrics: bool = True
):
    """
    Plot predictions against ground truth

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (if provided)
        show_metrics: Whether to show metrics on plot
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    plt.figure(figsize=figsize)

    # If multi-dimensional, flatten
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

    plt.plot(y_true, label='Ground Truth', linewidth=2, alpha=0.8)
    plt.plot(y_pred, label='Predictions', linewidth=2, alpha=0.8)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add metrics if requested
    if show_metrics:
        from .metrics import calculate_metrics
        metrics = calculate_metrics(y_true, y_pred)
        metrics_text = f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}"
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_history(
    history: Dict,
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot training history (loss and metrics over epochs)

    Args:
        history: Dictionary with training history
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot losses
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    if 'val_losses' in history and len(history['val_losses']) > 0:
        axes[0].plot(history['val_losses'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metrics (MAE)
    if 'train_metrics' in history and len(history['train_metrics']) > 0:
        train_mae = [m['mae'] for m in history['train_metrics']]
        axes[1].plot(train_mae, label='Train MAE', linewidth=2)

    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        val_mae = [m['mae'] for m in history['val_metrics']]
        axes[1].plot(val_mae, label='Val MAE', linewidth=2)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    input_labels: Optional[List[str]] = None,
    output_labels: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot attention weights as heatmap

    Args:
        attention_weights: Attention weights matrix
        input_labels: Labels for input sequence
        output_labels: Labels for output sequence
        figsize: Figure size
        save_path: Path to save figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=figsize)

    sns.heatmap(
        attention_weights,
        cmap='YlOrRd',
        cbar=True,
        xticklabels=input_labels if input_labels else 'auto',
        yticklabels=output_labels if output_labels else 'auto',
        fmt='.2f',
        linewidths=0.5
    )

    plt.xlabel('Input Sequence', fontsize=12)
    plt.ylabel('Output Sequence', fontsize=12)
    plt.title('Attention Weights', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_forecast_horizon_metrics(
    metrics_per_step: Dict[str, np.ndarray],
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot how metrics change across forecast horizon

    Args:
        metrics_per_step: Dictionary with metrics per forecast step
        figsize: Figure size
        save_path: Path to save figure
    """
    n_metrics = len(metrics_per_step)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics_per_step.items()):
        ax.plot(range(1, len(values) + 1), values, marker='o', linewidth=2)
        ax.set_xlabel('Forecast Horizon', fontsize=12)
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(f'{metric_name.upper()} per Forecast Step', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot residuals analysis

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Path to save figure
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals over time
    axes[0, 0].plot(residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Scatter plot: Predicted vs True
    axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Predicted vs True Values')
    axes[1, 0].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_multiple_forecasts(
    y_true: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    title: str = 'Model Comparison',
    figsize: tuple = (15, 6),
    save_path: Optional[str] = None
):
    """
    Plot multiple forecasts from different models

    Args:
        y_true: Ground truth values
        forecasts: Dictionary mapping model names to predictions
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Flatten if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()

    plt.figure(figsize=figsize)

    plt.plot(y_true, label='Ground Truth', linewidth=2.5, alpha=0.8, color='black')

    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))

    for (model_name, predictions), color in zip(forecasts.items(), colors):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        plt.plot(predictions, label=model_name, linewidth=2, alpha=0.7, color=color)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    title: str = 'Feature Importance',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot feature importance

    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_scores = importance_scores[indices]

    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_features)), sorted_scores, alpha=0.8)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_time_series_decomposition(
    data: np.ndarray,
    period: int = 24,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot time series decomposition (trend, seasonal, residual)

    Args:
        data: Time series data
        period: Period for seasonal decomposition
        figsize: Figure size
        save_path: Path to save figure
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if len(data.shape) > 1:
        data = data.flatten()

    # Perform decomposition
    decomposition = seasonal_decompose(data, model='additive', period=period)

    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Original
    axes[0].plot(data, linewidth=2)
    axes[0].set_ylabel('Original')
    axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(decomposition.trend, linewidth=2, color='orange')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(decomposition.seasonal, linewidth=2, color='green')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].plot(decomposition.resid, linewidth=2, color='red')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time Steps')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
