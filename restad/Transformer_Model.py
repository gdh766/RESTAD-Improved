#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

from RBF_Layer import RBFLayer


class TriangularCausalMask():
    """
    This class creates a 2D causal mask for sequences to ensure causality 
    in transformers. The mask ensures that a position in the sequence 
    cannot attend to future positions, thereby ensuring the order of 
    the sequence is respected.
    """
    
    def __init__(self, L, device="cpu"):
        """
        Initialize the TriangularCausalMask.
        
        Args:
        L (int): Length of each sequence.
        device (str, optional): The computing device where the mask will reside. 
                                Defaults to "cpu".
        """
        
        # Define the shape of the mask tensor
        mask_shape = [L, L]
        
        with torch.no_grad():
            # Create an upper triangular matrix of shape [L, L]
            # with the diagonal and above set to True (indicating masked positions)
            # and the rest set to False.
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            
    @property
    def mask(self):
        """
        Property to get the mask tensor.
        
        Returns:
        torch.Tensor: The mask tensor.
        """
        return self._mask
    


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        
        # Padding depending on PyTorch version
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        
        # 1D convolutional layer to project each token of the input to a dense vector
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        
        # Initialization for the convolution layer for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Convert tokens to dense vectors using the convolutional layer
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        
        # Compute the positional encodings once in log space.
        # This encoding will remain fixed across all batches and epochs.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 'position' corresponds to the position in the sequence.
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        # 'div_term' scales down the positional encoding, especially for larger dimensions.
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Generate sinusoidal positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store the positional encodings in the buffer so it's not treated as a model parameter
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Return positional encoding for each position of the input 'x'
        return self.pe[:, :x.size(1)]



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        # Token embedding layer
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # Positional embedding layer
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Combine token and positional embeddings and apply dropout
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)



class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        
        # Generate the mask for the current batch size and sequence length
        attn_mask = TriangularCausalMask(L=x.size(1), device=x.device).mask
        # attn_mask = TriangularCausalMask(L=x.size(1), device=x.device)


        attn_output, _ = self.attention(x, x, x, attn_mask=None)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)
    
    

class Encoder(nn.Module):
    # 【毕设修改 M1】新增 use_residual_rbf / d_model 两个参数。
    #   use_residual_rbf=False 时行为与原作者代码完全一致（x = rbf_out）。
    #   use_residual_rbf=True  时改为门控残差注入：
    #       h = W(x)                        # d_model -> rbf_dim 的旁路投影
    #       x = h + sigmoid(gate) ⊙ rbf_out  # gate 可学习，初始为 0 -> 0.5
    #   动机：原做法让 RBF 输出直接顶替第二层隐表示，第三层与最终投影都建立在
    #   相似度特征上，主干的语义信息被丢弃；门控残差保留主干通路，把 RBF 作为
    #   一路"相似度旁路"叠加进来。
    def __init__(self, encoder_layers, rbf_layer=None, modified_d_model=None, norm_layer=None,
                 use_residual_rbf=False, d_model=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.rbf_layer = rbf_layer
        self.modified_d_model = modified_d_model
        self.norm = norm_layer
        self.use_residual_rbf = use_residual_rbf

        if self.use_residual_rbf and rbf_layer is not None:
            self.rbf_proj = nn.Linear(d_model, modified_d_model)
            self.rbf_gate = nn.Parameter(torch.zeros(modified_d_model))

    def forward(self, x, attn_mask=None):
        rbf_out = None
        second_layer_out = None
        
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            

            if idx == 1:  # After processing through the second layer
                second_layer_out = x  # Save the output of the second layer
                if self.rbf_layer:
                    rbf_out = self.rbf_layer(x)
                    if self.use_residual_rbf:
                        # 【毕设修改 M1】门控残差注入，替代原来的整段替换
                        x = self.rbf_proj(x) + torch.sigmoid(self.rbf_gate) * rbf_out
                    else:
                        x = rbf_out
                    
        if self.norm:
            x = self.norm(x)
        return x, rbf_out, second_layer_out

    

class Transformer(nn.Module):
    def __init__(self, cfg, activation='gelu'):
        super(Transformer, self).__init__()

        # Encoding
        self.embedding = DataEmbedding(cfg.model.input_size, cfg.model.d_model, cfg.model.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(cfg.model.d_model, cfg.model.n_heads, cfg.model.d_ff, cfg.model.dropout, activation) 
                for _ in range(cfg.model.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(cfg.model.d_model)
        )
        
        self.projection = nn.Linear(cfg.model.d_model, cfg.model.input_size)

    def forward(self, x):
        x = self.embedding(x)
        enc_out, _,  second_layer_out = self.encoder(x)

        dec_out = self.projection(enc_out)
        
        return dec_out, second_layer_out, None
    
    
    
class Transformer_RBF(nn.Module):
    def __init__(self, cfg, activation='gelu'):
        super(Transformer_RBF, self).__init__()

        # Encoding
        self.embedding = DataEmbedding(cfg.model.input_size, cfg.model.d_model, cfg.model.dropout)

        # RBF Layer
        self.rbf_layer = RBFLayer([cfg.model.rbf_dim, cfg.model.d_model])

        # Define the encoder layers, with the third layer modified to accept RBF output size
        modified_d_model = cfg.model.rbf_dim
        encoder_layers = [
            EncoderLayer(cfg.model.d_model, cfg.model.n_heads, cfg.model.d_ff, cfg.model.dropout, activation) if i != 2 else 
            EncoderLayer(modified_d_model, cfg.model.n_heads, cfg.model.d_ff, cfg.model.dropout, activation)
            for i in range(cfg.model.e_layers)
        ]

        # Encoder
        # 【毕设修改 M1】use_residual_rbf 从配置读取，默认 False 即原论文行为
        self.use_residual_rbf = bool(cfg.model.get("use_residual_rbf", False))
        self.encoder = Encoder(
            encoder_layers,
            rbf_layer=self.rbf_layer,
            modified_d_model=modified_d_model,
            norm_layer=torch.nn.LayerNorm(modified_d_model),
            use_residual_rbf=self.use_residual_rbf,
            d_model=cfg.model.d_model,
        )

        self.projection = nn.Linear(modified_d_model, cfg.model.input_size)

    def forward(self, x):
        x = self.embedding(x)
        enc_out, rbf_out, second_layer_out = self.encoder(x)
        dec_out = self.projection(enc_out)
        
        return dec_out, second_layer_out, rbf_out


    
  