"""PyTorch LSTM regressor for UI-PRMD posture-quality scoring (PoC)."""

import torch
import torch.nn as nn


class LSTMScorer(nn.Module):
    def __init__(
        self,
        num_features: int = 117,
        hidden_1: int = 64,
        hidden_2: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=num_features, hidden_size=hidden_1, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_1, hidden_size=hidden_2, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, (h_n, _) = self.lstm2(out)
        last = h_n[-1]
        last = torch.relu(self.fc1(last))
        return torch.sigmoid(self.fc2(last))
