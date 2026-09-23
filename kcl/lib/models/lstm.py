import torch
import torch.nn as nn

class TextLSTM_Deep(nn.Module):
    """
    4-layer LSTM text classifier (embedding, stacked LSTM, linear classifier).

    Architecture:
    - 4-layer LSTM (vs 2-layer in standard)
    - hidden_dim=512 (vs 256 in standard)
    - ~4.5M parameters (vs 1.5M in standard)

    Expected behavior:
    - Converges in 15-20 epochs (vs 5 epochs for standard)
    - Final accuracy: ~90-91%
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 128,
        hidden_dim: int = 512,      # Increased from 256
        num_classes: int = 4,
        num_layers: int = 4,        # Increased from 2
        dropout: float = 0.3,       # Reduced from 0.5 to prevent underfitting
        bidirectional: bool = False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Deeper LSTM stack
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] - token indices
        Returns:
            logits: [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Use last layer's hidden state
        if self.bidirectional:
            h_n_forward = h_n[-2, :, :]
            h_n_backward = h_n[-1, :, :]
            h_n_final = torch.cat([h_n_forward, h_n_backward], dim=1)
        else:
            h_n_final = h_n[-1, :, :]

        # Classifier
        logits = self.fc(h_n_final)

        return logits
