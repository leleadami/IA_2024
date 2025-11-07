"""
LSTM Autoencoder for time series forecasting and anomaly detection
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder component

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            outputs: All LSTM outputs (batch_size, sequence_length, hidden_size)
            hidden: Hidden state tuple (h_n, c_n)
        """
        outputs, hidden = self.lstm(x)
        return outputs, hidden


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder component

    Args:
        hidden_size (int): Number of hidden units in LSTM (should match encoder)
        output_size (int): Number of output features
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LSTMDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, output_size)
            hidden: Hidden state from encoder

        Returns:
            outputs: Reconstructed sequence (batch_size, sequence_length, output_size)
        """
        lstm_out, hidden = self.lstm(x, hidden)
        outputs = self.fc(lstm_out)
        return outputs, hidden


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time series forecasting and reconstruction.

    The autoencoder learns compressed representations of input sequences,
    useful for:
    - Anomaly detection
    - Denoising
    - Feature extraction for forecasting

    Args:
        input_size (int): Number of input features
        hidden_size (int): Size of the latent representation
        num_layers (int): Number of LSTM layers in encoder and decoder
        output_size (int): Number of output features (usually same as input_size)
        dropout (float): Dropout rate
        forecast_horizon (int): Number of steps to forecast (default: 0, reconstruction only)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = None,
        dropout: float = 0.2,
        forecast_horizon: int = 0
    ):
        super(LSTMAutoencoder, self).__init__()

        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon

        # Encoder
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Decoder
        self.decoder = LSTMDecoder(
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Optional: Additional layer for forecasting
        if forecast_horizon > 0:
            self.forecast_fc = nn.Linear(hidden_size, forecast_horizon * output_size)

    def forward(self, x, return_latent=False):
        """
        Forward pass for reconstruction

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_latent: If True, also return the latent representation

        Returns:
            reconstruction: Reconstructed sequence
            latent (optional): Latent representation if return_latent=True
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        encoder_outputs, hidden = self.encoder(x)

        # Use the last encoder output as the latent representation
        latent = encoder_outputs[:, -1, :]

        # Prepare decoder input: start with zeros
        decoder_input = torch.zeros(
            (batch_size, seq_len, self.output_size),
            device=x.device
        )

        # Decode
        reconstruction, _ = self.decoder(decoder_input, hidden)

        if return_latent:
            return reconstruction, latent

        return reconstruction

    def encode(self, x):
        """
        Encode input sequence to latent representation

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Latent representation (batch_size, hidden_size)
        """
        encoder_outputs, _ = self.encoder(x)
        latent = encoder_outputs[:, -1, :]
        return latent

    def decode(self, latent, sequence_length):
        """
        Decode latent representation back to sequence

        Args:
            latent: Latent representation (batch_size, hidden_size)
            sequence_length: Length of sequence to generate

        Returns:
            Decoded sequence (batch_size, sequence_length, output_size)
        """
        batch_size = latent.shape[0]

        # Prepare hidden state from latent
        h_0 = latent.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)

        # Prepare decoder input
        decoder_input = torch.zeros(
            (batch_size, sequence_length, self.output_size),
            device=latent.device
        )

        # Decode
        outputs, _ = self.decoder(decoder_input, hidden)

        return outputs

    def forecast(self, x):
        """
        Forecast future values using the learned latent representation

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Forecasted values (batch_size, forecast_horizon, output_size)
        """
        if self.forecast_horizon == 0:
            raise ValueError("forecast_horizon must be > 0 for forecasting")

        # Encode to latent representation
        latent = self.encode(x)

        # Generate forecast from latent representation
        forecast = self.forecast_fc(latent)

        # Reshape to (batch_size, forecast_horizon, output_size)
        batch_size = x.shape[0]
        forecast = forecast.view(batch_size, self.forecast_horizon, self.output_size)

        return forecast


class ConvolutionalLSTMAutoencoder(nn.Module):
    """
    Enhanced LSTM Autoencoder with convolutional layers for better feature extraction.

    Args:
        input_size (int): Number of input features
        conv_channels (list): List of channel sizes for convolutional layers
        hidden_size (int): Size of LSTM hidden state
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        conv_channels: list = [64, 32],
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(ConvolutionalLSTMAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Encoder with 1D convolutions
        encoder_layers = []
        in_channels = input_size

        for out_channels in conv_channels:
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv_encoder = nn.Sequential(*encoder_layers)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # LSTM decoder
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=conv_channels[-1],
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder with transposed convolutions
        decoder_layers = []
        channels_reversed = list(reversed(conv_channels))

        for i in range(len(channels_reversed) - 1):
            decoder_layers.extend([
                nn.ConvTranspose1d(
                    channels_reversed[i],
                    channels_reversed[i + 1],
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(),
                nn.BatchNorm1d(channels_reversed[i + 1]),
                nn.Dropout(dropout)
            ])

        # Final layer to reconstruct input
        decoder_layers.append(
            nn.ConvTranspose1d(
                channels_reversed[-1],
                input_size,
                kernel_size=3,
                padding=1
            )
        )

        self.conv_decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Reconstructed sequence (batch_size, sequence_length, input_size)
        """
        batch_size, seq_len, _ = x.shape

        # Transpose for Conv1d: (batch, channels, sequence)
        x_transposed = x.transpose(1, 2)

        # Convolutional encoding
        conv_encoded = self.conv_encoder(x_transposed)

        # Transpose back for LSTM: (batch, sequence, channels)
        conv_encoded = conv_encoded.transpose(1, 2)

        # LSTM encoding
        _, (hidden, cell) = self.lstm_encoder(conv_encoded)

        # Prepare decoder input
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # LSTM decoding
        lstm_decoded, _ = self.lstm_decoder(decoder_input)

        # Transpose for Conv1d
        lstm_decoded = lstm_decoded.transpose(1, 2)

        # Convolutional decoding
        reconstruction = self.conv_decoder(lstm_decoded)

        # Transpose back to (batch, sequence, features)
        reconstruction = reconstruction.transpose(1, 2)

        return reconstruction
