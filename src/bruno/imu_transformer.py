import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer

class IMUTransformerEncoder(nn.Module):

    def __init__(
        self,
        input_shape: tuple = (6, 60),
        transformer_dim: int = 64,
        encode_position: bool = True,
        nhead: int = 8,
        dim_feedforward: int = 128,
        transformer_dropout: float = 0.1,
        transformer_activation: str = "gelu",
        num_encoder_layers: int = 6,
        permute: bool = False,
        return_cls_token: bool = False
    ):
        """
        input_shape: (tuple) shape of the input data
        transformer_dim: (int) dimension of the transformer
        encode_position: (bool) whether to encode position or not
        nhead: (int) number of attention heads
        dim_feedforward: (int) dimension of the feedforward network
        transformer_dropout: (float) dropout rate for the transformer
        transformer_activation: (str) activation function for the transformer
        num_encoder_layers: (int) number of transformer encoder layers
        num_classes: (int) number of output classes
        permute: bool, optional. If `True` the input data will be permuted before passing through the model, by default False.
        only_last: bool, optional. If `True` returns only last embedding of sample sequence
        """
        super().__init__()

        self.input_shape = input_shape
        self.transformer_dim = transformer_dim
        self.permute = permute
        self.return_cls_token = return_cls_token

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_shape[0], self.transformer_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1),
            nn.GELU(),
        )

        self.encode_position = encode_position
        encoder_layer = TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(self.transformer_dim),
        )
        if return_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros((1, self.transformer_dim)), requires_grad=True
            )

        extra_token = 1 if return_cls_token else 0
        if self.encode_position:
            self.position_embed = nn.Parameter(
                torch.randn(input_shape[1] + extra_token, 1, self.transformer_dim)
            )

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """Forward

        Parameters
        ----------
        x : _type_
            A tensor of shape (B, C, S) with B = batch size, C = channels, S = sequence length

        """
        if self.permute:
            x = x.permute(0, 2, 1)
        # Embed in a high dimensional space and reshape to Transformer's expected shape
        x = self.input_proj(x)
        # print(f"proj.shape: {x.shape}")
        x = x.permute(2, 0, 1)

        ## Prepend class token
        if self.return_cls_token:
            cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([cls_token, x])

        # Add the position embedding
        if self.encode_position:
            x += self.position_embed

        # Transformer Encoder pass
        if not self.return_cls_token:
            causal_mask = Transformer.generate_square_subsequent_mask(self.input_shape[1], device=x.device)
            target = self.transformer_encoder(x, mask=causal_mask, is_causal=True)
            target = target.permute(1,0,2)
        else:
            target = self.transformer_encoder(x)[0]

        return target