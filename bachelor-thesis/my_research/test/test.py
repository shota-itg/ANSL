import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule
# from torchmultimodal.modules.heads.projection_head import ProjectionHead
import torch.jit

from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter

from torchview import draw_graph

from torchvista import trace_model


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='relu'):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU() if activation == 'relu' else getattr(nn, activation)())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu', dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
    

class ModalityAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = SimpleMLP(input_dim=(6+1) *15, hidden_dims=[(6+1) *15 *15, (6+1) *15 *15], activation='relu')

    def forward(self, x):
        return self.mlp(x)  # shape: [batch_size, 3]

class ModalityBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = SimpleMLP(input_dim=6, hidden_dims=[15 *15, (6+1) *15 *15], activation='relu')

    def forward(self, x):
        return self.mlp(x)  # shape: [batch_size, 3]

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_a = ModalityAEncoder()
        self.encoder_b = ModalityBEncoder()
        self.fusion = ConcatFusionModule()
        self.project = ProjectionHead(input_dim=(6+1) *15 *15 *2, output_dim=(6+1) *15 *15, activation='relu')
        self.output_layer = SimpleMLP(input_dim=(6+1) *15 *15, hidden_dims=[(6+1) *15 *15, (6+1) *6 * 15], activation='relu')

    def forward(self, x_a, x_b):
        feat_a = self.encoder_a(x_a)  # [batch_size, 3]
        feat_b = self.encoder_b(x_b)  # [batch_size, 3]
        fused = self.fusion({'modality_a': feat_a, 'modality_b': feat_b})   # [batch_size, 6]
        projected = self.project(fused)        # [batch_size, 3]
        out = self.output_layer(projected)     # [batch_size, 2]
        return out





# テスト用入力
x_a = torch.randn(32, (6+1) *15)  # モダリティA: バッチサイズ4, 入力2次元
x_b = torch.randn(32, 6)  # モダリティB: バッチサイズ4, 入力1次元


"""
model = FusionModel()
output = model(x_a, x_b)
print(output.shape)  # torch.Size([4, 2])
"""


# モデルのインスタンス
model = FusionModel()

"""
# モデルをトレース
traced_model = torch.jit.trace(model, (x_a, x_b))

# モデル構造を表示
print(traced_model.code)
"""

"""
output = model(x_a, x_b)

make_dot(output, params=dict(model.named_parameters()))
"""

"""
writer = SummaryWriter()
writer.add_graph(model, (x_a, x_b))
writer.close()
"""

"""
graph = draw_graph(model, input_data=(x_a, x_b), expand_nested=True)
graph.visual_graph  # Notebook上に描画される
"""

trace_model(model, (x_a, x_b))