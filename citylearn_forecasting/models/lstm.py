"""
LSTM Forecaster for time series prediction
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Basic LSTM model for time series forecasting.

    Architecture:
    - Input Layer
    - LSTM Layers (stacked)
    - Dropout
    - Fully Connected Output Layer

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM
        num_layers (int): Number of stacked LSTM layers
        output_size (int): Number of output features (forecast horizon)
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional LSTM
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTMForecaster, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate the size for the linear layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(lstm_output_size, output_size)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the output from the last time step
        # Shape: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        last_output = self.dropout(last_output)

        # Pass through fully connected layer
        # Shape: (batch_size, output_size)
        output = self.fc(last_output)

        return output

    def predict_sequence(self, x, future_steps):
        """
        Predict multiple steps into the future using recursive prediction.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            future_steps: Number of future steps to predict

        Returns:
            Predictions tensor of shape (batch_size, future_steps, output_size)
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            current_input = x.clone()

            for _ in range(future_steps):
                # Predict next step
                pred = self.forward(current_input)
                predictions.append(pred)

                # Update input sequence for next prediction
                # Remove oldest timestep and append prediction
                if self.input_size == self.output_size:
                    # If input and output dimensions match, use prediction directly
                    new_input = pred.unsqueeze(1)
                else:
                    # Otherwise, pad or project the prediction
                    new_input = torch.zeros(
                        (pred.shape[0], 1, self.input_size),
                        device=pred.device
                    )
                    new_input[:, :, :self.output_size] = pred.unsqueeze(1)

                current_input = torch.cat([current_input[:, 1:, :], new_input], dim=1)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)

        return predictions


class LSTMMultiHorizon(nn.Module):
    """
    LSTM model for multi-horizon forecasting (predicting multiple steps at once).

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM
        num_layers (int): Number of stacked LSTM layers
        forecast_horizon (int): Number of future steps to predict
        output_size (int): Number of features to predict at each step
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(LSTMMultiHorizon, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        # Output layer that produces all future steps at once
        self.fc = nn.Linear(hidden_size, forecast_horizon * output_size)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, forecast_horizon, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        last_output = self.dropout(last_output)

        # Generate all predictions
        output = self.fc(last_output)

        # Reshape to (batch_size, forecast_horizon, output_size)
        batch_size = x.shape[0]
        output = output.view(batch_size, self.forecast_horizon, self.output_size)

        return output
