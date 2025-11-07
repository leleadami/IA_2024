"""
Evaluation metrics for time series forecasting
"""

import numpy as np
import torch
from typing import Dict, Union


def MAE(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Absolute Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Squared Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MSE score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.mean((y_true - y_pred) ** 2)


def RMSE(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Root Mean Squared Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE score
    """
    return np.sqrt(MSE(y_true, y_pred))


def MAPE(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    epsilon: float = 1e-8
) -> float:
    """
    Mean Absolute Percentage Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE score (in percentage)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return 0.0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def SMAPE(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Symmetric Mean Absolute Percentage Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        SMAPE score (in percentage)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0

    return np.mean(diff) * 100


def R2(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    R-squared (coefficient of determination)

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        R2 score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def WAPE(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Weighted Absolute Percentage Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        WAPE score (in percentage)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Calculate all metrics at once

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with all metric scores
    """
    metrics = {
        'mae': MAE(y_true, y_pred),
        'mse': MSE(y_true, y_pred),
        'rmse': RMSE(y_true, y_pred),
        'mape': MAPE(y_true, y_pred),
        'smape': SMAPE(y_true, y_pred),
        'r2': R2(y_true, y_pred),
        'wape': WAPE(y_true, y_pred)
    }

    return metrics


def rolling_window_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    window_size: int = 24
) -> Dict[str, np.ndarray]:
    """
    Calculate metrics over rolling windows

    Useful for understanding how forecast quality degrades over time

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        window_size: Size of rolling window

    Returns:
        Dictionary with metric arrays over time
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_windows = len(y_true) - window_size + 1

    mae_rolling = np.zeros(n_windows)
    rmse_rolling = np.zeros(n_windows)
    mape_rolling = np.zeros(n_windows)

    for i in range(n_windows):
        window_true = y_true[i:i + window_size]
        window_pred = y_pred[i:i + window_size]

        mae_rolling[i] = MAE(window_true, window_pred)
        rmse_rolling[i] = RMSE(window_true, window_pred)
        mape_rolling[i] = MAPE(window_true, window_pred)

    return {
        'mae_rolling': mae_rolling,
        'rmse_rolling': rmse_rolling,
        'mape_rolling': mape_rolling
    }


def horizon_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, np.ndarray]:
    """
    Calculate metrics for each forecast horizon step

    Useful for multi-step forecasting to see how accuracy changes
    with forecast horizon

    Args:
        y_true: Ground truth values (batch_size, forecast_horizon)
        y_pred: Predicted values (batch_size, forecast_horizon)

    Returns:
        Dictionary with metric arrays per horizon step
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    forecast_horizon = y_true.shape[1]

    mae_per_step = np.zeros(forecast_horizon)
    rmse_per_step = np.zeros(forecast_horizon)
    mape_per_step = np.zeros(forecast_horizon)

    for i in range(forecast_horizon):
        mae_per_step[i] = MAE(y_true[:, i], y_pred[:, i])
        rmse_per_step[i] = RMSE(y_true[:, i], y_pred[:, i])
        mape_per_step[i] = MAPE(y_true[:, i], y_pred[:, i])

    return {
        'mae_per_step': mae_per_step,
        'rmse_per_step': rmse_per_step,
        'mape_per_step': mape_per_step
    }


def directional_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions)

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Directional accuracy (0-100)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate differences
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    # Check if directions match
    correct_direction = np.sign(true_diff) == np.sign(pred_diff)

    return np.mean(correct_direction) * 100


class MetricsTracker:
    """
    Track metrics during training and evaluation

    Args:
        metrics (list): List of metric names to track
    """

    def __init__(self, metrics: list = ['mae', 'rmse', 'r2']):
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}

    def update(self, y_true, y_pred):
        """Update metrics with new predictions"""
        all_metrics = calculate_metrics(y_true, y_pred)

        for metric in self.metrics:
            if metric in all_metrics:
                self.history[metric].append(all_metrics[metric])

    def get_average(self, metric: str) -> float:
        """Get average value of a metric"""
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0.0
        return np.mean(self.history[metric])

    def get_last(self, metric: str) -> float:
        """Get last value of a metric"""
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0.0
        return self.history[metric][-1]

    def reset(self):
        """Reset all metrics"""
        self.history = {metric: [] for metric in self.metrics}

    def summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        return {metric: self.get_average(metric) for metric in self.metrics}
