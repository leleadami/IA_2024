"""
TimesNet: Temporal 2D-Variation Modeling for Time Series Forecasting

Based on the paper: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
https://arxiv.org/abs/2210.02186

TimesNet transforms 1D time series into 2D tensors based on multiple periods,
then applies 2D convolutions to capture complex temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Inception_Block_V1(nn.Module):
    """
    Inception block for multi-scale feature extraction in 2D space

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_kernels (int): Number of different kernel sizes to use
    """

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i
                )
            )
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class FFT_for_Period(nn.Module):
    """
    Fast Fourier Transform to identify dominant periods in time series

    Args:
        k (int): Number of top periods to extract
    """

    def __init__(self, k=5):
        super(FFT_for_Period, self).__init__()
        self.k = k

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, channels)

        Returns:
            period_list: List of dominant periods
            period_weight: Weights for each period
        """
        # Apply FFT along the time dimension
        xf = torch.fft.rfft(x, dim=1)

        # Get frequency amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0  # Remove DC component

        # Find top k frequencies
        _, top_list = torch.topk(frequency_list, self.k)
        top_list = top_list.detach().cpu().numpy()

        # Convert frequencies to periods
        period_list = []
        period_weight = []

        for idx in top_list:
            if idx == 0:
                period_list.append(x.shape[1])
            else:
                period_list.append(x.shape[1] // idx)
            period_weight.append(frequency_list[idx])

        # Normalize weights
        period_weight = torch.stack(period_weight)
        period_weight = period_weight / period_weight.sum()

        return period_list, period_weight


class TimesBlock(nn.Module):
    """
    TimesBlock: Core building block that transforms 1D time series to 2D
    and applies Inception blocks for feature extraction

    Args:
        seq_len (int): Length of input sequence
        pred_len (int): Length of prediction
        k (int): Number of top periods to use
        d_model (int): Dimension of the model
        d_ff (int): Dimension of feed-forward network
        num_kernels (int): Number of kernels in Inception block
    """

    def __init__(self, seq_len, pred_len, k, d_model, d_ff, num_kernels=6):
        super(TimesBlock, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = k

        # Period detection using FFT
        self.period_detector = FFT_for_Period(k)

        # Parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            res: Output tensor (batch_size, seq_len, d_model)
        """
        B, T, N = x.shape

        # Detect dominant periods
        period_list, period_weight = self.period_detector(x)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # Padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([B, (length - self.seq_len), N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x

            # Reshape to 2D: (batch, period, num_periods, channels)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D convolution
            out = self.conv(out)

            # Reshape back to 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])

        # Aggregate results from different periods
        res = torch.stack(res, dim=-1)

        # Weighted sum based on period importance
        period_weight = period_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(x.device)
        res = torch.sum(res * period_weight, dim=-1)

        # Residual connection
        res = res + x

        return res


class TimesNet(nn.Module):
    """
    TimesNet: Complete model for time series forecasting

    Args:
        input_size (int): Number of input features
        seq_len (int): Length of input sequence
        pred_len (int): Length of prediction (forecast horizon)
        d_model (int): Dimension of the model
        d_ff (int): Dimension of feed-forward network
        num_layers (int): Number of TimesBlocks
        top_k (int): Number of top periods to use
        num_kernels (int): Number of kernels in Inception block
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 96,
        pred_len: int = 24,
        d_model: int = 64,
        d_ff: int = 128,
        num_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1
    ):
        super(TimesNet, self).__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.num_layers = num_layers

        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)

        # TimesBlocks
        self.layers = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Projection for forecasting
        self.projection = nn.Linear(d_model, input_size)

        # Optional: Direct forecasting head
        self.forecast_head = nn.Linear(seq_len, pred_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            forecast: Predictions (batch_size, pred_len, input_size)
        """
        # Input embedding
        x = self.embedding(x)
        x = self.dropout(x)

        # Apply TimesBlocks
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Forecast
        # Transpose to apply forecast head on time dimension
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.forecast_head(x)  # (batch, d_model, pred_len)
        x = x.permute(0, 2, 1)  # (batch, pred_len, d_model)

        # Project back to input size
        forecast = self.projection(x)

        return forecast


class TimesNetClassification(nn.Module):
    """
    TimesNet adapted for time series classification tasks

    Args:
        input_size (int): Number of input features
        seq_len (int): Length of input sequence
        num_classes (int): Number of classes
        d_model (int): Dimension of the model
        d_ff (int): Dimension of feed-forward network
        num_layers (int): Number of TimesBlocks
        top_k (int): Number of top periods
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 96,
        num_classes: int = 2,
        d_model: int = 64,
        d_ff: int = 128,
        num_layers: int = 2,
        top_k: int = 5,
        dropout: float = 0.1
    ):
        super(TimesNetClassification, self).__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model

        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)

        # TimesBlocks
        self.layers = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, d_model, d_ff)
            for _ in range(num_layers)
        ])

        # Global pooling and classification head
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Embedding
        x = self.embedding(x)
        x = self.dropout(x)

        # TimesBlocks
        for layer in self.layers:
            x = layer(x)

        # Flatten and classify
        x = self.layer_norm(x)
        x = x.reshape(x.shape[0], -1)
        logits = self.classifier(x)

        return logits


class TimesNetAnomalyDetection(nn.Module):
    """
    TimesNet adapted for anomaly detection in time series

    Uses reconstruction error as anomaly score

    Args:
        input_size (int): Number of input features
        seq_len (int): Length of input sequence
        d_model (int): Dimension of the model
        d_ff (int): Dimension of feed-forward network
        num_layers (int): Number of TimesBlocks
        top_k (int): Number of top periods
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 96,
        d_model: int = 64,
        d_ff: int = 128,
        num_layers: int = 2,
        top_k: int = 5,
        dropout: float = 0.1
    ):
        super(TimesNetAnomalyDetection, self).__init__()

        # Encoder
        self.embedding = nn.Linear(input_size, d_model)

        self.layers = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, d_model, d_ff)
            for _ in range(num_layers)
        ])

        # Decoder
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            reconstruction: Reconstructed input (batch_size, seq_len, input_size)
        """
        # Encode
        enc = self.embedding(x)
        enc = self.dropout(enc)

        for layer in self.layers:
            enc = layer(enc)

        # Decode
        enc = self.layer_norm(enc)
        reconstruction = self.projection(enc)

        return reconstruction

    def detect_anomaly(self, x, threshold=None):
        """
        Detect anomalies based on reconstruction error

        Args:
            x: Input tensor
            threshold: Anomaly threshold (if None, returns scores only)

        Returns:
            anomaly_scores: Reconstruction error for each sample
            is_anomaly (optional): Boolean mask if threshold is provided
        """
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)

            # Calculate reconstruction error (MSE per sample)
            anomaly_scores = torch.mean(
                (x - reconstruction) ** 2,
                dim=[1, 2]
            )

            if threshold is not None:
                is_anomaly = anomaly_scores > threshold
                return anomaly_scores, is_anomaly

            return anomaly_scores
