"""
Training script for time series forecasting models
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
from utils import Trainer, plot_training_history


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type: str, config: dict, input_size: int) -> nn.Module:
    """
    Create model based on type and configuration

    Args:
        model_type: Type of model to create
        config: Model configuration
        input_size: Number of input features

    Returns:
        PyTorch model
    """
    if model_type == 'lstm':
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            output_size=config.get('output_size', 1),
            dropout=config.get('dropout', 0.2),
            bidirectional=config.get('bidirectional', False)
        )

    elif model_type == 'lstm_autoencoder':
        model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            output_size=config.get('output_size', input_size),
            dropout=config.get('dropout', 0.2),
            forecast_horizon=config.get('pred_len', 24)
        )

    elif model_type == 'lstm_attention':
        model = LSTMAttention(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            output_size=config.get('output_size', 1),
            dropout=config.get('dropout', 0.2),
            attention_type=config.get('attention_type', 'bahdanau')
        )

    elif model_type == 'lstm_encoder_decoder':
        model = LSTMEncoderDecoder(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 128),
            output_size=config.get('output_size', 1),
            num_layers=config.get('num_layers', 2),
            forecast_horizon=config.get('pred_len', 24),
            dropout=config.get('dropout', 0.2),
            teacher_forcing_ratio=config.get('teacher_forcing_ratio', 0.5)
        )

    elif model_type == 'timesnet':
        model = TimesNet(
            input_size=input_size,
            seq_len=config.get('seq_len', 96),
            pred_len=config.get('pred_len', 24),
            d_model=config.get('d_model', 64),
            d_ff=config.get('d_ff', 128),
            num_layers=config.get('num_layers', 2),
            top_k=config.get('top_k', 5),
            num_kernels=config.get('num_kernels', 6),
            dropout=config.get('dropout', 0.1)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_model(
    model_type: str = 'lstm',
    config_path: str = 'configs/model_configs.yaml',
    data_path: str = None,
    save_dir: str = 'checkpoints'
):
    """
    Train a forecasting model

    Args:
        model_type: Type of model to train
        config_path: Path to configuration file
        data_path: Path to data file
        save_dir: Directory to save checkpoints
    """
    # Load configuration
    config = load_config(config_path)
    model_config = config['models'][model_type]
    train_config = config['training']

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    if data_path is None:
        # Generate synthetic data for demonstration
        print("No data path provided. Generating synthetic data for demonstration...")
        t = np.linspace(0, 100, 10000)
        data = np.sin(t) + 0.5 * np.sin(3 * t) + np.random.normal(0, 0.1, len(t))
        data = data.reshape(-1, 1)
    else:
        # Load from file
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.values
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            raise ValueError("Unsupported data format. Use .csv or .npy")

    # Create dataset
    dataset = TimeSeriesDataset(
        data=data,
        seq_len=model_config.get('seq_len', 96),
        pred_len=model_config.get('pred_len', 24),
        stride=train_config.get('stride', 1),
        scale=True
    )

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config.get('num_workers', 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config.get('num_workers', 0)
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Get input size
    sample_x, _ = dataset[0]
    input_size = sample_x.shape[-1]

    # Create model
    print(f"Creating {model_type} model...")
    model = create_model(model_type, model_config, input_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config.get('weight_decay', 0.0)
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Loss function
    criterion = nn.MSELoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        grad_clip=train_config.get('grad_clip', None)
    )

    # Train
    print("\nStarting training...")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_best.pth')

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config['epochs'],
        verbose=True,
        save_best=True,
        save_path=save_path,
        early_stopping=train_config.get('early_stopping', 20)
    )

    print("\nTraining completed!")
    print(f"Best model saved to: {save_path}")

    # Plot training history
    plot_training_history(history, save_path=os.path.join(save_dir, f'{model_type}_history.png'))

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train time series forecasting model')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'lstm_autoencoder', 'lstm_attention',
                               'lstm_encoder_decoder', 'timesnet'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default='configs/model_configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        config_path=args.config,
        data_path=args.data,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
