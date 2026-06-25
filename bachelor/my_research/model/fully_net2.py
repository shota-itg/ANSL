# model/fully_net.py

import yaml
import math
import torch
import torch.nn as nn

from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule

# config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

nodes = config["nodes"]
N = len(nodes)
NP_2 = math.perm(N, 2)  # ノード間ペア数 P(N,2)

num_traffic = config["num_traffic"]


## ネットワークの定義
    # 全結合層のマルチモーダルニューラルネットワーク
class Modality1Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(
            in_dim=(N+1) *num_traffic, 
            out_dim=(N+1) *num_traffic *num_traffic, 
            hidden_dims=[(N+1) *num_traffic *num_traffic, (N+1) *num_traffic *num_traffic], 
            dropout=0.0, 
            activation=nn.ReLU, 
            normalization=None
            )

    def forward(self, x):
        return self.mlp(x)  # shape: [batch_size, 3]
    
class Modality2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(
            in_dim=N **2, 
            out_dim=(N+1) *num_traffic *num_traffic, 
            hidden_dims=[(N+1) *num_traffic, (N+1) *num_traffic *num_traffic], 
            dropout=0.0, 
            activation=nn.ReLU, 
            normalization=None
            )

    def forward(self, x):
        return self.mlp(x)  # shape: [batch_size, 3]
    
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_a = Modality1Encoder()
        self.encoder_b = Modality2Encoder()
        self.fusion = ConcatFusionModule(
            projection=nn.Sequential(
                nn.Linear((N+1) *num_traffic *num_traffic *2, (N+1) *num_traffic *num_traffic), 
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
        )
        self.output_layer = MLP(
            in_dim=(N+1) *num_traffic *num_traffic, 
            out_dim=(N+1) *N *num_traffic, 
            hidden_dims=[(N+1) *num_traffic *num_traffic, (N+1) *num_traffic *num_traffic, (N+1) *num_traffic *num_traffic], 
            dropout=0.0, 
            activation=nn.ReLU, 
            normalization=None
            )

    def forward(self, x_a, x_b):
        feat_a = self.encoder_a(x_a)    # shape: [batch_size, 3]
        feat_b = self.encoder_b(x_b)    # shape: [batch_size, 3]
        fused = self.fusion({'modality_a': feat_a, 'modality_b': feat_b})   # shape: [batch_size, 6]
        out = self.output_layer(fused)  #shape: [batch_size, 2]
        return out