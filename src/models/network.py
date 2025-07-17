import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x is [batch_size, seq_len, embedding_dim] due to batch_first=True
        # We need to add positional encoding appropriate for this shape.
        # Currently, self.pe is [max_len, 1, d_model]. We need to broadcast.
        # This requires `x = x + self.pe[:x.size(1), :, :]` if x was [seq_len, batch_size, features]
        # For batch_first=True, it's `x = x + self.pe[:x.size(1)].transpose(0,1)` or reshape pe.
        # A simpler way is to make pe directly compatible with batch_first:
        # pe = torch.zeros(1, max_len, d_model)
        # pe[0, :, 0::2] = ...
        # pe[0, :, 1::2] = ...
        # Then x = x + self.pe[:, :x.size(1), :]
        
        # Adapting for batch_first=True, assuming self.pe is [max_len, 1, d_model]
        # We need self.pe to be [1, max_len, d_model] for broadcasting, or slice and unsqueeze.
        # For simplicity in this adaptation, we'll assume `pe` is prepared for `batch_first=True`
        # if the original TransformerEncoder is set with `batch_first=True`.
        # However, the current `PositionalEncoding` is designed for `seq_len, batch_size, features`.
        # Let's adjust `forward` to handle `batch_first=True` inputs.
        
        # If input x is [batch_size, seq_len, embedding_dim]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0) # Broadcast pe across the batch dimension
        return self.dropout(x)

class TransformerSiameseNet(nn.Module):
    """Siamese Network using Transformer Encoder for audio cognate detection."""
    def __init__(self, n_mels=80, d_model=240, nhead=8, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1, embedding_dim=128):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        # Initial CNN layers to process spectrogram and match d_model
        # Input: [batch_size, 1, n_mels, n_frames]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)), # Reduces n_mels dimension
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)), # Reduces n_mels dimension further
            # Output height after two MaxPool2d((2,2)) is n_mels // 4
            # We want the final feature dimension after flattening to be d_model
            # So, output channels * (n_mels // 4) should ideally be d_model.
            # Thus, output channels = d_model / (n_mels // 4)
            nn.Conv2d(64, d_model // (n_mels // 4), kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(d_model // (n_mels // 4))
        )
        
        # The actual feature_dim_after_cnn depends on the output of the last CNN layer.
        # This will be `channels * height` where height = n_mels // 4.
        # The number of channels is `d_model // (n_mels // 4)`.
        # So, `(d_model // (n_mels // 4)) * (n_mels // 4)` which simplifies to `d_model`.
        self.feature_dim_after_cnn = d_model 

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # batch_first=True means input/output of TransformerEncoderLayer/TransformerEncoder is [batch_size, seq_len, features]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Final linear layer to produce the embedding
        self.fc = nn.Linear(d_model, embedding_dim)

    def _forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single branch of the Siamese network."""
        # x shape: [batch_size, 1, n_mels, n_frames]
        x = self.cnn(x)
        # x shape: [batch_size, channels, height, width(time)]
        
        # Reshape for Transformer: [batch_size, n_frames, features]
        b, c, h, w = x.size()
        # Flatten channels and height into the feature dimension for the Transformer
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h) 
        # x shape: [batch_size, n_frames, self.feature_dim_after_cnn] which should be d_model

        # Add positional encoding
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x) # Input: [batch_size, n_frames, d_model]
        # x shape: [batch_size, n_frames, d_model]

        # Aggregate sequence: Use mean pooling over the time dimension
        x = x.mean(dim=1) # [batch_size, d_model]

        # Final embedding layer
        embedding = self.fc(x) # [batch_size, embedding_dim]
        return embedding

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes two inputs through the Siamese branches."""
        # Ensure input has channel dimension: [batch_size, n_mels, n_frames] -> [batch_size, 1, n_mels, n_frames]
        if x1.dim() == 3:
            x1 = x1.unsqueeze(1)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(1)

        emb1 = self._forward_one(x1)
        emb2 = self._forward_one(x2)
        return emb1, emb2