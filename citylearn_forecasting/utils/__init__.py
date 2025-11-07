"""
Utilities package for CityLearn Forecasting
"""

from .metrics import calculate_metrics, MAE, MSE, RMSE, MAPE, R2
from .trainer import Trainer
from .visualization import plot_predictions, plot_training_history, plot_attention_weights
from .advanced_training import (
    AdvancedTrainer,
    get_advanced_scheduler,
    init_weights,
    EMA,
    create_optimizer,
    BEST_PRACTICES
)

__all__ = [
    'calculate_metrics',
    'MAE', 'MSE', 'RMSE', 'MAPE', 'R2',
    'Trainer',
    'AdvancedTrainer',
    'get_advanced_scheduler',
    'init_weights',
    'EMA',
    'create_optimizer',
    'BEST_PRACTICES',
    'plot_predictions',
    'plot_training_history',
    'plot_attention_weights'
]
