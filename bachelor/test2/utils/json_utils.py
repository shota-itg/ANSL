# utils/json_utils.py

import os
import json
from datetime import datetime

from utils.config_loader import load_config, load_hyperparameter, load_runtime, save_runtime


def define_json(config, hparam_cfg, start_time=None, exp_dir=None) -> None:
    seed = config["seed"]

    topo_name = config["topology"]["name"]
    nodes = config["topology"]["nodes"]
    links = config["topology"]["links"]

    bandwidth_options = config["bandwidth_options"]

    train_lf_enabled = config["train"]["train_lf_enabled"]
    max_random_failure = config["train"]["max_random_failure"]
    num_train_data = config["train"]["num_train_data"]
    train_ratio = config["train"]["train_ratio"]
    val_ratio = config["train"]["val_ratio"]
    max_epoch = config["train"]["max_epoch"]

    test_lf_enabled = config["test"]["test_lf_enabled"]
    num_test_data = config["test"]["num_test_data"]

    result_json = config["paths"]["results"]["filename"]["result_json"]

    learning_rate = float(hparam_cfg["optimization"]["learning_rate"])
    train_batch_size = hparam_cfg["optimization"]["train_batch_size"]
    inference_batch_size = hparam_cfg["optimization"]["inference_batch_size"]
    optimizer_name = hparam_cfg["optimization"]["optimizer"]
    weight_decay = hparam_cfg["optimization"]["weight_decay"]

    num_traffic = hparam_cfg["architecture"]["num_traffic"]
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
    
    scheduler_name = hparam_cfg["schedule"]["lr_scheduler"]
    warmup_steps = hparam_cfg["schedule"]["warmup_steps"]
    StepLR_step_size = hparam_cfg["schedule"]["StepLR"]["step_size"]
    StepLR_gamma = hparam_cfg["schedule"]["StepLR"]["gamma"]
    StepLR_last_epoch = hparam_cfg["schedule"]["StepLR"]["last_epoch"]
    CosineAnnealingLR_T_max = hparam_cfg["schedule"]["CosineAnnealingLR"]["T_max"]
    CosineAnnealingLR_eta_min = hparam_cfg["schedule"]["CosineAnnealingLR"]["eta_min"]
    CosineAnnealingLR_last_epoch = hparam_cfg["schedule"]["CosineAnnealingLR"]["last_epoch"]
    ReduceLROnPlateau_mode = hparam_cfg["schedule"]["ReduceLROnPlateau"]["mode"]
    ReduceLROnPlateau_factor = hparam_cfg["schedule"]["ReduceLROnPlateau"]["factor"]
    ReduceLROnPlateau_patience = hparam_cfg["schedule"]["ReduceLROnPlateau"]["patience"]
    ReduceLROnPlateau_threshold = hparam_cfg["schedule"]["ReduceLROnPlateau"]["threshold"]
    ReduceLROnPlateau_threshold_mode = hparam_cfg["schedule"]["ReduceLROnPlateau"]["threshold_mode"]
    ReduceLROnPlateau_cooldown = hparam_cfg["schedule"]["ReduceLROnPlateau"]["cooldown"]
    ReduceLROnPlateau_min_lr = hparam_cfg["schedule"]["ReduceLROnPlateau"]["min_lr"]
    ReduceLROnPlateau_eps = hparam_cfg["schedule"]["ReduceLROnPlateau"]["eps"]

    erly_stppg_patience = hparam_cfg["early_stopping"]["patience"]
    erly_stppg_delta = hparam_cfg["early_stopping"]["delta"]
    
    rglrztn_drop = hparam_cfg["regularization"]["dropout"]
    rglrztn_nrmlz = hparam_cfg["regularization"]["normalization"]
    rglrztn_mx_grd_nrm = hparam_cfg["regularization"]["max_grad_norm"]
    rglrztn_lbl_smthg = hparam_cfg["regularization"]["label_smoothing"]

    mdl1_depth = len(mdl1_hidden)
    mdl2_depth = len(mdl2_hidden)
    fsn_depth = len(fsn_hidden)


    define_json = {
        "exp_id": f'{exp_dir}', 
        "datetime": start_time, 

        "config": {
            "seed": seed, 
            "topology": {
                "name": topo_name, 
                "nodes": nodes, 
                "links": links
            }, 
            "bandwidth_options": bandwidth_options, 
            "train": {
                "train_lf_enabled": train_lf_enabled, 
                "max_random_failure": max_random_failure, 
                "num_train_data": num_train_data, 
                "train_ratio": train_ratio, 
                "val_ratio": val_ratio, 
                "max_epoch": max_epoch
            }, 
            "test": {
                "test_lf_enabled": test_lf_enabled, 
                "num_test_data": num_test_data
            }
        }, 
        
        "hyperparameters": {
            "optimization": {
                "learning_rate": learning_rate, 
                "train_batch_size": train_batch_size, 
                "inference_batch_size": inference_batch_size, 
                "optimizer": optimizer_name, 
                "weight_decay": weight_decay
            }, 
            "architecture": {
                "num_traffic": num_traffic, 
                "modality1": {
                    "inputs_dim": mdl1_in, 
                    "outputs_dim": mdl1_out, 
                    "hidden_dims": mdl1_hidden, 
                    "depth": mdl1_depth, 
                    "dropout": mdl1_drop, 
                    "activate": mdl1_actv, 
                    "normalization": mdl1_nrmlz
                }, 
                "modality2": {
                    "inputs_dim": mdl2_in, 
                    "outputs_dim": mdl2_out, 
                    "hidden_dims": mdl2_hidden, 
                    "depth": mdl2_depth, 
                    "dropout": mdl2_drop, 
                    "activate": mdl2_actv, 
                    "normalization": mdl2_nrmlz
                }, 
                "attentionfusion": {
                    "channel_to_encoder_dim": fsn_chnl_to_encdr_dim, 
                    "encoding_prjection_dim": fsn_encdg_prjctn_dim
                }, 
                "afterfusion": {
                    "inputs_dim": fsn_in, 
                    "outputs_dim": fsn_out, 
                    "hidden_dims": fsn_hidden, 
                    "depth": fsn_depth, 
                    "dropout": fsn_drop, 
                    "activate": fsn_actv, 
                    "normalization": fsn_nrmlz
                }
            }, 
            "schedule": {
                "lr_scheduler": scheduler_name, 
                "warmup_steps": warmup_steps, 
                "StepLR": {
                    "step_size": StepLR_step_size, 
                    "gamma": StepLR_gamma, 
                    "last_epoch": StepLR_last_epoch
                }, 
                "CosineAnnealingLR": {
                    "T_max": CosineAnnealingLR_T_max, 
                    "eta_min": CosineAnnealingLR_eta_min, 
                    "last_epoch": CosineAnnealingLR_last_epoch
                }, 
                "ReduceLROnPlateau": {
                    "mode": ReduceLROnPlateau_mode, 
                    "factor": ReduceLROnPlateau_factor, 
                    "patience": ReduceLROnPlateau_patience, 
                    "threshold": ReduceLROnPlateau_threshold, 
                    "threshold_mode": ReduceLROnPlateau_threshold_mode, 
                    "cooldown": ReduceLROnPlateau_cooldown, 
                    "min_lr": ReduceLROnPlateau_min_lr, 
                    "eps": ReduceLROnPlateau_eps
                }
            }, 
            "early_stopping": {
                "patience": erly_stppg_patience, 
                "delta": erly_stppg_delta
            }, 
            "regularization": {
                "dropout": rglrztn_drop, 
                "normalization": rglrztn_nrmlz, 
                "max_grad_norm": rglrztn_mx_grd_nrm, 
                "label_smoothing": rglrztn_lbl_smthg
            }
        }, 
        
        "results": {
            "train": {
                "learning_time": None, 
                "average_epoch_time": None, 
                "final_epoch": None, 
                "loss": {
                    "final_train_loss": None, 
                    "final_val_loss": None
                }, 
                "accuracy": {
                    "final_train_demand_accuracy": None, 
                    "final_train_path_accuracy": None, 
                    "final_train_element_accuracy": None, 
                    "final_val_demand_accuracy": None, 
                    "final_val_path_accuracy": None, 
                    "final_val_element_accuracy": None, 
                }
            }, 
            "inference": {
                "inference_time": None, 
                "average_inference_time": None, 
                "num_test_datasets": None, 
                "inference_batch_size": None, 
                "accuracy": {
                    "test_demand_accuracy": None, 
                    "test_path_accuracy": None, 
                    "test_element_accuracy": None
                }
            }, 
            "routing_success_rate": {
                "routing_success_evaluation_time": None, 
                "average_routing_success_evaluation_time": None, 
                "num_test_datasets": None, 
                "inference_batch_size": None, 
                "accuracy": {
                    "routing_success_rate": None
                }, 
                "load_balance_metrics": {
                    "maximum_link_utilization": {
                        "average": None, 
                        "standard_deviation": None, 
                        "min": None, 
                        "max": None
                    }, 
                    "standard_deviation": {
                        "average": None, 
                        "standard_deviation": None
                    }, 
                    "coefficient_of_variation": {
                        "average": None, 
                        "standard_deviation": None
                    }
                }
            }, 
            "link_failure_resilience": {
                "routing_success_evaluation_time": None, 
                "average_routing_success_evaluation_time": None, 
                "num_test_datasets": None, 
                "inference_batch_size": None, 
            }, 
            "link_failure_stateaware_resilience": {
                "routing_success_evaluation_time": None, 
                "average_routing_success_evaluation_time": None, 
                "num_test_datasets": None, 
                "inference_batch_size": None, 
            }
        }
    }

    if exp_dir is not None:
        results_json_path = os.path.join(exp_dir, result_json)
    else:
        results_json_path = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            "jsons/", 
            result_json
        )

    with open(results_json_path, "w") as f:
        json.dump(define_json, f, indent=4)


if __name__=="__main__":
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config = load_config()
    hparam_cfg = load_hyperparameter()
    runtime_cfg = load_runtime()

    define_json(config, hparam_cfg, runtime_cfg, start_time, exp_dir=None)