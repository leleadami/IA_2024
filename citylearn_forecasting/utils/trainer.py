"""
Training utilities for time series forecasting models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Callable
import os

from .metrics import calculate_metrics


class Trainer:
    """
    Trainer class for time series forecasting models

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on ('cpu' or 'cuda')
        scheduler: Optional learning rate scheduler
        grad_clip: Gradient clipping value (None for no clipping)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip

        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            verbose: Whether to show progress bar

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        iterator = tqdm(train_loader, desc='Training') if verbose else train_loader

        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)

            # Handle different output shapes
            if len(predictions.shape) == 3 and len(batch_y.shape) == 2:
                # Model outputs (batch, seq, features), target is (batch, seq)
                predictions = predictions.squeeze(-1)
            elif len(predictions.shape) == 2 and len(batch_y.shape) == 1:
                # Model outputs (batch, features), target is (batch,)
                predictions = predictions.squeeze(-1)

            # Calculate loss
            loss = self.criterion(predictions, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(batch_y.detach().cpu())

            if verbose:
                iterator.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        self.train_losses.append(avg_loss)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        self.train_metrics.append(metrics)

        return metrics

    def validate(
        self,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Validate the model

        Args:
            val_loader: Validation data loader
            verbose: Whether to show progress bar

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        iterator = tqdm(val_loader, desc='Validation') if verbose else val_loader

        with torch.no_grad():
            for batch_x, batch_y in iterator:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                predictions = self.model(batch_x)

                # Handle different output shapes
                if len(predictions.shape) == 3 and len(batch_y.shape) == 2:
                    predictions = predictions.squeeze(-1)
                elif len(predictions.shape) == 2 and len(batch_y.shape) == 1:
                    predictions = predictions.squeeze(-1)

                # Calculate loss
                loss = self.criterion(predictions, batch_y)

                # Track metrics
                epoch_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())

                if verbose:
                    iterator.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = epoch_loss / len(val_loader)
        self.val_losses.append(avg_loss)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        self.val_metrics.append(metrics)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
        save_best: bool = True,
        save_path: str = 'best_model.pth',
        early_stopping: Optional[int] = None
    ):
        """
        Train the model for multiple epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            verbose: Whether to show progress
            save_best: Whether to save best model
            save_path: Path to save best model
            early_stopping: Number of epochs without improvement before stopping

        Returns:
            Training history
        """
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, verbose=verbose)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, verbose=verbose)

                if verbose:
                    print(f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val MAE: {val_metrics['mae']:.4f}, "
                          f"Val RMSE: {val_metrics['rmse']:.4f}")

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

                # Save best model
                if save_best and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_path)
                    epochs_without_improvement = 0
                    if verbose:
                        print(f"Saved best model with val_loss: {best_val_loss:.4f}")
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    if verbose:
                        print(f"\nEarly stopping after {epoch + 1} epochs")
                    break

            else:
                if verbose:
                    print(f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train MAE: {train_metrics['mae']:.4f}, "
                          f"Train RMSE: {train_metrics['rmse']:.4f}")

                if self.scheduler is not None:
                    self.scheduler.step()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions on a dataset

        Args:
            data_loader: Data loader

        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)
                all_predictions.append(predictions.cpu().numpy())

        return np.concatenate(all_predictions, axis=0)

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint

        Args:
            path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class EarlyStopping:
    """
    Early stopping utility

    Args:
        patience (int): Number of epochs to wait before stopping
        min_delta (float): Minimum change to qualify as improvement
        mode (str): 'min' or 'max' (whether lower or higher is better)
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False
