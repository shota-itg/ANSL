# model/fully_net.py

import yaml
import math
import torch
import torch.nn as nn

from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.fusions.attention_fusion import AttentionFusionModule
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule

from utils.config_loader import load_hyperparameter
from utils.hparam_map import ACTIVATION_MAP, NOMALIZATION_MAP, SCHEDULER_MAP


## ネットワークの定義
    # 全結合層のマルチモーダルニューラルネットワーク
class Modality1Encoder(nn.Module):
    def __init__(self, mdl1_in, mdl1_out, mdl1_hidden, mdl1_drop, mdl1_actv, mdl1_nrmlz):
        super().__init__()
        self.mlp = MLP(
            in_dim=mdl1_in, 
            out_dim=mdl1_out, 
            hidden_dims=mdl1_hidden, 
            dropout=mdl1_drop, 
            activation=mdl1_actv, 
            normalization=mdl1_nrmlz
        )

    def forward(self, x):
        return self.mlp(x)  # shape: [batch_size, 3]
    

class Modality2Encoder(nn.Module):
    def __init__(self, mdl2_in, mdl2_out, mdl2_hidden, mdl2_drop, mdl2_actv, mdl2_nrmlz):
        super().__init__()
        self.mlp = MLP(
            in_dim=mdl2_in, 
            out_dim=mdl2_out, 
            hidden_dims=mdl2_hidden, 
            dropout=mdl2_drop, 
            activation=mdl2_actv, 
            normalization=mdl2_nrmlz
        )

    def forward(self, x):
        return self.mlp(x)  # shape: [batch_size, 3]
    

class FusionModel(nn.Module):
    def __init__(self, exp_dir=None):
        super().__init__()

        hparam_cfg = load_hyperparameter(exp_dir)
        mdl1_in = hparam_cfg["architecture"]["modality1"]["inputs_dim"]
        mdl1_out = hparam_cfg["architecture"]["modality1"]["outputs_dim"]
        mdl1_hidden = hparam_cfg["architecture"]["modality1"]["hidden_dims"]
        mdl1_drop = hparam_cfg["architecture"]["modality1"].get(
            "dropout", 
            hparam_cfg["regularization"]["dropout"]
        )
        mdl1_actv = hparam_cfg["architecture"]["modality1"]["activation"]
        mdl1_nrmlz = hparam_cfg["architecture"]["modality1"].get(
            "normalization", 
            hparam_cfg["regularization"]["normalization"]
        )
        mdl2_in = hparam_cfg["architecture"]["modality2"]["inputs_dim"]
        mdl2_out = hparam_cfg["architecture"]["modality2"]["outputs_dim"]
        mdl2_hidden = hparam_cfg["architecture"]["modality2"]["hidden_dims"]
        mdl2_drop = hparam_cfg["architecture"]["modality2"].get(
            "dropout", 
            hparam_cfg["regularization"]["dropout"]
        )
        mdl2_actv = hparam_cfg["architecture"]["modality2"]["activation"]
        mdl2_nrmlz = hparam_cfg["architecture"]["modality2"].get(
            "normalization", 
            hparam_cfg["regularization"]["normalization"]
        )
        fsn_chnl_to_encdr_dim = hparam_cfg["architecture"]["attentionfusion"]["channel_to_encoder_dim"]
        fsn_encdg_prjctn_dim = hparam_cfg["architecture"]["attentionfusion"]["encoding_projection_dim"]
        fsn_in = hparam_cfg["architecture"]["attentionfusion"]["inputs_dim"]
        fsn_out = hparam_cfg["architecture"]["attentionfusion"]["outputs_dim"]
        fsn_hidden = hparam_cfg["architecture"]["attentionfusion"]["hidden_dims"]
        fsn_drop = hparam_cfg["architecture"]["attentionfusion"].get(
            "dropout", 
            hparam_cfg["regularization"]["dropout"]
        )
        fsn_actv = hparam_cfg["architecture"]["attentionfusion"]["activation"]
        fsn_nrmlz = hparam_cfg["architecture"]["attentionfusion"]["normalization"]

        mdl1_actv = ACTIVATION_MAP[mdl1_actv]
        mdl1_nrmlz = NOMALIZATION_MAP[mdl1_nrmlz]
        mdl2_actv = ACTIVATION_MAP[mdl2_actv]
        mdl2_nrmlz = NOMALIZATION_MAP[mdl2_nrmlz]
        fsn_actv = ACTIVATION_MAP[fsn_actv]
        fsn_nrmlz = NOMALIZATION_MAP[fsn_nrmlz]


        self.encoder_a = Modality1Encoder(
            mdl1_in=mdl1_in, 
            mdl1_out=mdl1_out, 
            mdl1_hidden=mdl1_hidden, 
            mdl1_drop=mdl1_drop, 
            mdl1_actv=mdl1_actv, 
            mdl1_nrmlz=mdl1_nrmlz
        )
        self.encoder_b = Modality2Encoder(
            mdl2_in=mdl2_in, 
            mdl2_out=mdl2_out, 
            mdl2_hidden=mdl2_hidden, 
            mdl2_drop=mdl2_drop, 
            mdl2_actv=mdl2_actv, 
            mdl2_nrmlz=mdl2_nrmlz
        )


        """ ConcatFusion
        self.fusion = ConcatFusionModule()        
        """



        """ AttentionFusion
 
        """
        self.fusion = AttentionFusionModule(
            channel_to_encoder_dim={
                "modality1": fsn_chnl_to_encdr_dim["modality1"], 
                "modality2": fsn_chnl_to_encdr_dim["modality2"]
            }, 
            encoding_projection_dim=fsn_encdg_prjctn_dim
        )       


        self.mlp = MLP(
            in_dim=fsn_in, 
            out_dim=fsn_out,  
            hidden_dims=fsn_hidden, 
            dropout=fsn_drop, 
            activation=fsn_actv, 
            normalization=fsn_nrmlz
        )

    def forward(self, m1, m2):
        out_a = self.encoder_a(m1)    # shape: [batch_size, 3]
        out_b = self.encoder_b(m2)    # shape: [batch_size, 3]
        fused = self.fusion({'modality1': out_a, 'modality2': out_b})   # shape: [batch_size, 6]
        out = self.mlp(fused)  #shape: [batch_size, 2]
        return out