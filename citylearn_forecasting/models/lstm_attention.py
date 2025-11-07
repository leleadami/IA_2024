"""
LSTM with Attention Mechanism for time series forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism

    Args:
        hidden_size (int): Size of the hidden state
    """

    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()

        self.hidden_size = hidden_size

        # Attention weights
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: Encoder outputs (batch_size, seq_len, hidden_size)
            decoder_hidden: Current decoder hidden state (batch_size, hidden_size)

        Returns:
            context: Context vector (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.shape

        # Expand decoder hidden to match encoder outputs shape
        # (batch_size, seq_len, hidden_size)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention scores
        # (batch_size, seq_len, hidden_size)
        score = torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(decoder_hidden_expanded)
        )

        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        attention_scores = self.v(score).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate context vector as weighted sum of encoder outputs
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_size)
        # -> (batch_size, 1, hidden_size) -> (batch_size, hidden_size)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention Mechanism

    Args:
        hidden_size (int): Size of the hidden state
        attention_type (str): Type of attention ('dot', 'general', 'concat')
    """

    def __init__(self, hidden_size: int, attention_type: str = 'general'):
        super(LuongAttention, self).__init__()

        self.hidden_size = hidden_size
        self.attention_type = attention_type

        if attention_type == 'general':
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention_type == 'concat':
            self.W_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: Encoder outputs (batch_size, seq_len, hidden_size)
            decoder_hidden: Current decoder hidden state (batch_size, hidden_size)

        Returns:
            context: Context vector (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_size = encoder_outputs.shape

        if self.attention_type == 'dot':
            # Dot product attention
            # (batch_size, hidden_size, 1)
            decoder_hidden_expanded = decoder_hidden.unsqueeze(2)
            # (batch_size, seq_len, hidden_size) @ (batch_size, hidden_size, 1)
            # -> (batch_size, seq_len, 1) -> (batch_size, seq_len)
            attention_scores = torch.bmm(
                encoder_outputs,
                decoder_hidden_expanded
            ).squeeze(2)

        elif self.attention_type == 'general':
            # General attention
            # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
            decoder_hidden_transformed = self.W_a(decoder_hidden).unsqueeze(1)
            # (batch_size, seq_len, hidden_size) @ (batch_size, hidden_size, 1)
            # -> (batch_size, seq_len)
            attention_scores = torch.bmm(
                encoder_outputs,
                decoder_hidden_transformed.transpose(1, 2)
            ).squeeze(2)

        elif self.attention_type == 'concat':
            # Concat attention
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            # Concatenate encoder outputs and decoder hidden
            combined = torch.cat([encoder_outputs, decoder_hidden_expanded], dim=2)
            # Apply transformations
            attention_scores = self.v(torch.tanh(self.W_a(combined))).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights


class LSTMAttention(nn.Module):
    """
    LSTM with Attention mechanism for time series forecasting.

    This model uses attention to focus on relevant parts of the input sequence
    when making predictions.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM
        num_layers (int): Number of stacked LSTM layers
        output_size (int): Number of output features
        dropout (float): Dropout rate
        attention_type (str): Type of attention ('bahdanau', 'luong')
        luong_attention_variant (str): Variant for Luong attention
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        attention_type: str = 'bahdanau',
        luong_attention_variant: str = 'general'
    ):
        super(LSTMAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.attention_type = attention_type

        # Encoder LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size)
        elif attention_type == 'luong':
            self.attention = LuongAttention(hidden_size, luong_attention_variant)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layers
        # Combine context and LSTM output
        self.fc_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_attention=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_attention: If True, also return attention weights

        Returns:
            output: Predictions (batch_size, output_size)
            attention_weights (optional): Attention weights if return_attention=True
        """
        # Encoder: pass through LSTM
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        encoder_outputs, (hidden, cell) = self.lstm(x)

        # Use last hidden state as query for attention
        # hidden: (num_layers, batch_size, hidden_size)
        # Take the last layer's hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Apply attention
        context, attention_weights = self.attention(encoder_outputs, last_hidden)

        # Combine context vector with last hidden state
        combined = torch.cat([context, last_hidden], dim=1)
        combined = self.dropout(combined)

        # Pass through fully connected layers
        combined = torch.relu(self.fc_combine(combined))
        combined = self.dropout(combined)

        output = self.fc_out(combined)

        if return_attention:
            return output, attention_weights

        return output


class LSTMAttentionMultiHorizon(nn.Module):
    """
    LSTM with Attention for multi-horizon forecasting.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units
        num_layers (int): Number of LSTM layers
        forecast_horizon (int): Number of future steps to predict
        output_size (int): Number of features to predict at each step
        dropout (float): Dropout rate
        attention_type (str): Type of attention mechanism
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        output_size: int = 1,
        dropout: float = 0.2,
        attention_type: str = 'bahdanau'
    ):
        super(LSTMAttentionMultiHorizon, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        # Encoder LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size)
        else:
            self.attention = LuongAttention(hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Output layer for all forecast steps
        self.fc_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, forecast_horizon * output_size)

    def forward(self, x, return_attention=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_attention: If True, also return attention weights

        Returns:
            output: Predictions (batch_size, forecast_horizon, output_size)
            attention_weights (optional): Attention weights if return_attention=True
        """
        batch_size = x.shape[0]

        # Encoder
        encoder_outputs, (hidden, cell) = self.lstm(x)

        # Attention
        last_hidden = hidden[-1]
        context, attention_weights = self.attention(encoder_outputs, last_hidden)

        # Combine and predict
        combined = torch.cat([context, last_hidden], dim=1)
        combined = self.dropout(combined)

        combined = torch.relu(self.fc_combine(combined))
        combined = self.dropout(combined)

        output = self.fc_out(combined)

        # Reshape to (batch_size, forecast_horizon, output_size)
        output = output.view(batch_size, self.forecast_horizon, self.output_size)

        if return_attention:
            return output, attention_weights

        return output


class SelfAttentionLSTM(nn.Module):
    """
    LSTM with Self-Attention mechanism.

    This model applies self-attention over the LSTM outputs to capture
    dependencies between different time steps.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units
        num_layers (int): Number of LSTM layers
        output_size (int): Number of output features
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super(SelfAttentionLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            output: Predictions (batch_size, output_size)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Self-attention with residual connection
        attended, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        attended = self.layer_norm1(lstm_out + attended)

        # Feed-forward with residual connection
        ffn_out = self.ffn(attended)
        output_sequence = self.layer_norm2(attended + ffn_out)

        # Take the last time step
        last_output = output_sequence[:, -1, :]

        # Final prediction
        output = self.fc_out(last_output)

        return output
