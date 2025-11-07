"""
Advanced optimizations and best practices for CityLearn Forecasting models

This module provides enhanced training utilities to prevent overfitting and improve model performance:
- Advanced learning rate schedulers
- Mixed precision training
- Gradient accumulation
- Better weight initialization
- Additional regularization techniques
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional, Dict
import numpy as np


class AdvancedTrainer:
    """
    Enhanced trainer with advanced optimization techniques

    Features:
    - Mixed precision training (faster, less memory)
    - Gradient accumulation (simulate larger batches)
    - Advanced LR schedulers (OneCycleLR, CosineAnnealing)
    - Stochastic Weight Averaging (SWA)
    - Better regularization

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on
        use_amp: Use automatic mixed precision
        accumulation_steps: Number of steps to accumulate gradients
        grad_clip: Gradient clipping value
        swa_start: Epoch to start Stochastic Weight Averaging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        use_amp: bool = True,
        accumulation_steps: int = 1,
        grad_clip: Optional[float] = 1.0,
        swa_start: Optional[int] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.accumulation_steps = accumulation_steps
        self.grad_clip = grad_clip
        self.swa_start = swa_start

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # SWA model
        self.swa_model = None
        if swa_start is not None:
            self.swa_model = torch.optim.swa_utils.AveragedModel(model)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with advanced techniques"""
        self.model.train()
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(batch_x)

                    # Handle shape mismatch
                    if len(predictions.shape) == 3 and len(batch_y.shape) == 2:
                        predictions = predictions.squeeze(-1)

                    loss = self.criterion(predictions, batch_y)
                    loss = loss / self.accumulation_steps

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (i + 1) % self.accumulation_steps == 0:
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            else:
                # Standard training
                predictions = self.model(batch_x)

                if len(predictions.shape) == 3 and len(batch_y.shape) == 2:
                    predictions = predictions.squeeze(-1)

                loss = self.criterion(predictions, batch_y)
                loss = loss / self.accumulation_steps
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.accumulation_steps

        # Update SWA model if applicable
        if self.swa_model is not None and epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)

        avg_loss = epoch_loss / len(train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss


def get_advanced_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine_warmup',
    num_epochs: int = 100,
    steps_per_epoch: int = 100,
    **kwargs
):
    """
    Get advanced learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
            - 'cosine_warmup': Cosine annealing with warmup
            - 'onecycle': One Cycle LR policy
            - 'reduce_plateau': Reduce on plateau
            - 'cosine': Cosine annealing
        num_epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch
        **kwargs: Additional scheduler parameters

    Returns:
        Learning rate scheduler
    """

    if scheduler_type == 'cosine_warmup':
        # Cosine annealing with warmup
        warmup_epochs = kwargs.get('warmup_epochs', 5)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == 'onecycle':
        # One Cycle LR
        max_lr = kwargs.get('max_lr', 0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )

    elif scheduler_type == 'cosine':
        # Simple cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get('eta_min', 1e-6)
        )

    elif scheduler_type == 'reduce_plateau':
        # Reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def init_weights(model: nn.Module, init_type: str = 'xavier'):
    """
    Initialize model weights with better strategies

    Args:
        model: PyTorch model
        init_type: Initialization type
            - 'xavier': Xavier/Glorot initialization
            - 'kaiming': Kaiming/He initialization
            - 'orthogonal': Orthogonal initialization
    """

    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(param)

        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


class EMA:
    """
    Exponential Moving Average of model parameters

    Helps stabilize training and can improve generalization

    Args:
        model: PyTorch model
        decay: EMA decay rate
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for regression (experimental)

    Adds small noise to targets to prevent overconfident predictions

    Args:
        smoothing: Smoothing factor (0-1)
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Add small random noise to targets
        noise = torch.randn_like(target) * self.smoothing * target.std()
        smoothed_target = target + noise
        return self.mse(pred, smoothed_target)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 0.001,
    weight_decay: float = 0.01,
    **kwargs
):
    """
    Create optimizer with best practices

    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
            - 'adamw': AdamW (Adam with decoupled weight decay)
            - 'adam': Standard Adam
            - 'sgd': SGD with momentum
            - 'radam': Rectified Adam
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer parameters

    Returns:
        PyTorch optimizer
    """

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )

    elif optimizer_type == 'radam':
        # RAdam (requires torch >= 1.13)
        try:
            optimizer = torch.optim.RAdam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        except AttributeError:
            print("RAdam not available, using AdamW instead")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


# Best practices summary
BEST_PRACTICES = {
    'dropout': {
        'lstm': 0.2,  # Good for LSTM layers
        'attention': 0.1,  # Lower for attention mechanisms
        'timesnet': 0.1,  # Lower for complex architectures
    },
    'learning_rate': {
        'initial': 0.001,  # Standard starting point
        'min': 1e-6,  # Minimum LR for schedulers
        'warmup_epochs': 5,  # Warmup epochs
    },
    'regularization': {
        'weight_decay': 0.01,  # L2 regularization (AdamW)
        'grad_clip': 1.0,  # Gradient clipping
        'early_stopping': 20,  # Patience for early stopping
    },
    'batch_size': {
        'small_data': 32,  # < 10k samples
        'medium_data': 64,  # 10k-100k samples
        'large_data': 128,  # > 100k samples
    },
    'data_split': {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15,
    }
}
