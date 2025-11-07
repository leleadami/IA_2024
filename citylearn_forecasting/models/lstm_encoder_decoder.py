"""
LSTM Encoder-Decoder (Seq2Seq) for time series forecasting
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder component of the Seq2Seq model

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
        super(Encoder, self).__init__()

        self.input_size = input_size
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
            outputs: LSTM outputs (batch_size, sequence_length, hidden_size)
            hidden: Tuple of (h_n, c_n) hidden and cell states
        """
        outputs, hidden = self.lstm(x)
        return outputs, hidden


class Decoder(nn.Module):
    """
    Decoder component of the Seq2Seq model

    Args:
        input_size (int): Number of input features (typically output_size from prediction)
        hidden_size (int): Number of hidden units in LSTM (should match encoder)
        output_size (int): Number of output features
        num_layers (int): Number of LSTM layers (should match encoder)
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, input_size)
            hidden: Hidden state from encoder or previous decoder step

        Returns:
            output: Prediction (batch_size, 1, output_size)
            hidden: Updated hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Apply dropout and fully connected layer
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)

        return output, hidden


class LSTMEncoderDecoder(nn.Module):
    """
    LSTM Encoder-Decoder (Seq2Seq) model for time series forecasting.

    This architecture is particularly effective for multi-step ahead forecasting
    where the output sequence length can differ from the input sequence length.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM
        output_size (int): Number of output features
        num_layers (int): Number of LSTM layers
        forecast_horizon (int): Number of future steps to predict
        dropout (float): Dropout rate
        teacher_forcing_ratio (float): Probability of using teacher forcing during training
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        dropout: float = 0.2,
        teacher_forcing_ratio: float = 0.5
    ):
        super(LSTMEncoderDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Decoder
        self.decoder = Decoder(
            input_size=output_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x, target=None):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_size)
            target: Target tensor for teacher forcing (batch_size, forecast_horizon, output_size)
                   If None, uses model's own predictions

        Returns:
            outputs: Predictions (batch_size, forecast_horizon, output_size)
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode input sequence
        encoder_outputs, hidden = self.encoder(x)

        # Initialize decoder input with zeros or last encoder output
        decoder_input = torch.zeros(
            (batch_size, 1, self.output_size),
            device=device
        )

        # Store decoder outputs
        outputs = []

        # Decode for forecast_horizon steps
        for t in range(self.forecast_horizon):
            # Decoder step
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(decoder_output)

            # Teacher forcing: use actual target as next input
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                # Use model's own prediction as next input
                decoder_input = decoder_output

        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)

        return outputs

    def predict(self, x):
        """
        Make predictions without teacher forcing

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_size)

        Returns:
            outputs: Predictions (batch_size, forecast_horizon, output_size)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, target=None)


class LSTMEncoderDecoderWithAttention(nn.Module):
    """
    LSTM Encoder-Decoder with Attention mechanism for improved forecasting.

    Combines the sequence-to-sequence architecture with attention to allow
    the decoder to focus on relevant parts of the input sequence.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM
        output_size (int): Number of output features
        num_layers (int): Number of LSTM layers
        forecast_horizon (int): Number of future steps to predict
        dropout (float): Dropout rate
        teacher_forcing_ratio (float): Probability of using teacher forcing
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        dropout: float = 0.2,
        teacher_forcing_ratio: float = 0.5
    ):
        super(LSTMEncoderDecoderWithAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=output_size + hidden_size,  # Input + context from attention
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def attention_mechanism(self, decoder_hidden, encoder_outputs):
        """
        Compute attention weights and context vector

        Args:
            decoder_hidden: Current decoder hidden state (batch_size, hidden_size)
            encoder_outputs: All encoder outputs (batch_size, seq_len, hidden_size)

        Returns:
            context: Context vector (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.shape

        # Expand decoder hidden to match encoder outputs
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate and compute attention scores
        combined = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
        attention_scores = self.attention(combined).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Compute context vector as weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights

    def forward(self, x, target=None, return_attention=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_size)
            target: Target tensor for teacher forcing
            return_attention: If True, return attention weights

        Returns:
            outputs: Predictions (batch_size, forecast_horizon, output_size)
            attention_history (optional): List of attention weights if return_attention=True
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode
        encoder_outputs, hidden = self.encoder(x)

        # Initialize decoder input
        decoder_input = torch.zeros((batch_size, 1, self.output_size), device=device)

        outputs = []
        attention_history = [] if return_attention else None

        # Decode
        for t in range(self.forecast_horizon):
            # Get decoder hidden state
            decoder_hidden = hidden[0][-1]  # Last layer's hidden state

            # Compute attention and context
            context, attention_weights = self.attention_mechanism(
                decoder_hidden,
                encoder_outputs
            )

            if return_attention:
                attention_history.append(attention_weights)

            # Concatenate decoder input with context
            decoder_input_with_context = torch.cat(
                [decoder_input, context.unsqueeze(1)],
                dim=2
            )

            # Decoder step
            decoder_output, hidden = self.decoder_lstm(
                decoder_input_with_context,
                hidden
            )

            # Generate output
            decoder_output = self.dropout(decoder_output)
            output = self.fc_out(decoder_output)
            outputs.append(output)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = output

        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)

        if return_attention:
            return outputs, attention_history

        return outputs

    def predict(self, x):
        """
        Make predictions without teacher forcing

        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_size)

        Returns:
            outputs: Predictions (batch_size, forecast_horizon, output_size)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, target=None)


class BidirectionalLSTMEncoderDecoder(nn.Module):
    """
    Bidirectional LSTM Encoder-Decoder for improved context understanding.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units
        output_size (int): Number of output features
        num_layers (int): Number of LSTM layers
        forecast_horizon (int): Number of future steps
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        dropout: float = 0.2
    ):
        super(BidirectionalLSTMEncoderDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Projection layer to convert bidirectional hidden state to decoder hidden state
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, input_seq_len, input_size)

        Returns:
            outputs: Predictions (batch_size, forecast_horizon, output_size)
        """
        batch_size = x.shape[0]
        device = x.device

        # Bidirectional encoding
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Project bidirectional hidden states to decoder hidden states
        # hidden shape: (num_layers * 2, batch_size, hidden_size)
        # Reshape to (num_layers, batch_size, hidden_size * 2)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(self.num_layers, batch_size, self.hidden_size * 2)
        hidden = self.hidden_projection(hidden)

        # Same for cell state
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = cell.transpose(1, 2).contiguous()
        cell = cell.view(self.num_layers, batch_size, self.hidden_size * 2)
        cell = self.cell_projection(cell)

        # Decode
        decoder_input = torch.zeros(
            (batch_size, 1, self.output_size),
            device=device
        )

        outputs = []
        for _ in range(self.forecast_horizon):
            decoder_output, (hidden, cell) = self.decoder(
                decoder_input,
                (hidden, cell)
            )

            decoder_output = self.dropout(decoder_output)
            output = self.fc_out(decoder_output)
            outputs.append(output)

            decoder_input = output

        outputs = torch.cat(outputs, dim=1)
        return outputs
