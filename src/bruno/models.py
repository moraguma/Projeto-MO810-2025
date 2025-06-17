import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
from minerva.models.ssl.cpc import CPC
from minerva.models.nets.cpc_networks import Convolutional1DEncoder, HARCPCAutoregressive, Genc_Gar
from minerva.models.ssl.tfc import TFC_Model
from minerva.models.nets.tfc import TFC_Backbone

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


def tfc_model_cnn(input_shape, batch_size):
    """
        input_shape (tuple): Input data shape. Must be in the form (channels, seq_length)
        batch_size (int): Batch size for trainning
    """
    return TFC_Model(
        input_channels=input_shape[0],
        TS_length=input_shape[1],
        single_encoding_size=128,
        pred_head=False,
        batch_size=batch_size,
    )
    
def tfc_model_imu(input_shape, batch_size):
    """
        input_shape (tuple): Input data shape. Must be in the form (channels, seq_length)
        batch_size (int): Batch size for trainning
    """
    time_enc = IMUTransformerEncoder(input_shape=input_shape, transformer_dim=128, nhead=8, num_encoder_layers=8, dim_feedforward=128, return_cls_token=True)
    freq_enc = IMUTransformerEncoder(input_shape=input_shape, transformer_dim=128, nhead=8, num_encoder_layers=8, dim_feedforward=128, return_cls_token=True)

    return TFC_Model(
        input_channels=input_shape[0],
        TS_length=input_shape[1],
        single_encoding_size=128,
        pred_head=False,
        batch_size=batch_size,
        time_encoder=time_enc,
        frequency_encoder=freq_enc,
        learning_rate=1e-4,
    )
    
def cpc_model_cnn(input_shape, batch_size):
    g_enc = Convolutional1DEncoder(input_size=input_shape[0], kernel_size=3, padding=1)
    g_ar = HARCPCAutoregressive(input_size=128, hidden_size=256, batch_first=True, bidirectional=False)

    return CPC(
        g_enc=g_enc,
        g_ar=g_ar,
        prediction_head_in_channels=256,
        prediction_head_out_channels=128,
        num_steps_prediction=28,
        batch_size=batch_size,
        minimum_steps=10,
    )

def cpc_model_imu_transformer(input_shape, batch_size):
    g_enc = IMUTransformerEncoder(input_shape=input_shape, transformer_dim=64, nhead=8, num_encoder_layers=8, dim_feedforward=128)
    g_ar = HARCPCAutoregressive(input_size=64, hidden_size=256, batch_first=True, bidirectional=False)

    return CPC(
        g_enc=g_enc,
        g_ar=g_ar,
        prediction_head_in_channels=256,
        prediction_head_out_channels=64,
        num_steps_prediction=28,
        batch_size=batch_size,
        minimum_steps=10,
        learning_rate=1e-4,
    )
    
def tfc_cnn_backbone(input_shape, batch_size, ckpt_file=None):
    if ckpt_file != None:
        pretext_model = TFC_Model.load_from_checkpoint(
            ckpt_file,
            input_channels=input_shape[0],
            TS_length=input_shape[1],
            single_encoding_size=128,
            pred_head=False,
            batch_size=batch_size,
        )
    else:
        pretext_model = TFC_Model(
            input_channels=input_shape[0],
            TS_length=input_shape[1],
            single_encoding_size=128,
            pred_head=False,
            batch_size=batch_size,
        )

    return pretext_model.backbone
    
def tfc_imu_backbone(input_shape, batch_size, ckpt_file=None):
    time_enc = IMUTransformerEncoder(input_shape=input_shape, transformer_dim=128, nhead=8, num_encoder_layers=8, dim_feedforward=128, return_cls_token=True)
    freq_enc = IMUTransformerEncoder(input_shape=input_shape, transformer_dim=128, nhead=8, num_encoder_layers=8, dim_feedforward=128, return_cls_token=True)

    if ckpt_file != None:
        pretext_model = TFC_Model.load_from_checkpoint(
            ckpt_file,
            input_channels=input_shape[0],
            TS_length=input_shape[1],
            single_encoding_size=128,
            pred_head=False,
            batch_size=batch_size,
            time_encoder=time_enc,
            frequency_encoder=freq_enc,
        )

    return TFC_Backbone(
            input_channels=input_shape[0],
            TS_length=input_shape[1],
            single_encoding_size=128,
            time_encoder=time_enc.to('cpu'), # Send to CPU for projector input size calculation
            frequency_encoder=freq_enc.to('cpu'), # Send to CPU for projector input size calculation
            )

def cpc_imu_backbone(input_shape, batch_size, ckpt_file=None):
    g_enc = IMUTransformerEncoder(input_shape=input_shape, transformer_dim=64, nhead=8, num_encoder_layers=8, dim_feedforward=128)
    g_ar = HARCPCAutoregressive(input_size=64, hidden_size=256, batch_first=True, bidirectional=False)

    if ckpt_file != None:
        pretext_model = CPC.load_from_checkpoint(
            ckpt_file,
            g_enc=g_enc,
            g_ar=g_ar,
            prediction_head_in_channels=256,
            prediction_head_out_channels=64,
            num_steps_prediction=28,
            batch_size=batch_size,
            minimum_steps=10,
        )
        
    return Genc_Gar(g_enc=g_enc, g_ar=g_ar)
    
def cpc_cnn_backbone(input_shape, batch_size, ckpt_file=None):
    g_enc = Convolutional1DEncoder(input_size=input_shape[0], kernel_size=3, padding=1)
    g_ar = HARCPCAutoregressive(input_size=128, hidden_size=256, batch_first=True, bidirectional=False)
    
    if ckpt_file != None:
        pretext_model = CPC.load_from_checkpoint(
            ckpt_file,
            g_enc=g_enc,
            g_ar=g_ar,
            prediction_head_in_channels=256,
            prediction_head_out_channels=128,
            num_steps_prediction=28,
            batch_size=batch_size,
            minimum_steps=10,
        )
        
    return Genc_Gar(g_enc=g_enc, g_ar=g_ar)