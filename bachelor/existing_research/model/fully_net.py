# model/fully_net.py

import torch
import torch.nn as nn


## ネットワークの定義
class PartiallyConnectedLayerNet(nn.Module):
    """
    input_layer: (N+1) * P(N,2)
    hidden_layer: ((N+1) * P(N,2)) * P(N,2)
    output_layer: ((N+1)*N) * P(N,2)
    num_blocks: P(N,2)
    """
    
    def __init__(self, num_blocks, inputs_dim, hidden_dim, hidden_depth, outputs_dim):
        super().__init__()

        # nn.ModuleList で複数の Sequential を管理
        self.blocks = nn.ModuleList([
            build_sequential(
                num_blocks, 
                inputs_dim, 
                hidden_dim, 
                hidden_depth-1, 
                outputs_dim
            ) for _ in range(num_blocks)
        ])

    def forward(self, inputs):
        outputs = []
        for block in self.blocks:
            outputs.append(block(inputs))   # 各 Sequential に共通の入力 inputs を通す

        return torch.stack(outputs, dim=1)  # shape: [batch_size, 20] -> shape: [batch_size, num_blocks, 20]

def build_sequential(NP_2, inputs_dim, hidden_dim, num_hidden_layer, outputs_dim):
    layers = [nn.Linear(inputs_dim, int(hidden_dim /NP_2))]  # <--
    for _ in range(num_hidden_layer):
        layers.append(nn.Linear(int(hidden_dim /NP_2), int(hidden_dim /NP_2)))  # <--
        layers.append(nn.ReLU())
    layers.append(nn.Linear(int(hidden_dim /NP_2), int(outputs_dim /NP_2)))   # <--
    return nn.Sequential(*layers)