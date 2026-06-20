# compare/compare_seed.py

import os
import json
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from utils.config_loader import load_config, load_hyperparameter


# ============================================================
# 1. YAML / JSON ロード
# ============================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# 2. 実験フォルダの読み込み
# ============================================================

def load_experiments(root):
    """
    experiments/ex_exp_c/exp_20250117_120000/
    のようなフォルダをすべて読み込む
    """
    exps = []

    for exp_name in os.listdir(root):
        exp_dir = os.path.join(root, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        if not exp_name.startswith("exp_"):
            continue

        config_path = os.path.join(exp_dir, "config.yaml")
        hparam_path = os.path.join(exp_dir, "hyperparameter.yaml")
        result_path = os.path.join(exp_dir, "result.json")

        if not (os.path.exists(config_path) and os.path.exists(hparam_path)):
            continue

        config = load_yaml(config_path)
        hparam = load_yaml(hparam_path)
        result = load_json(result_path)

        exps.append({
            "exp_dir": exp_dir,
            "config": config,
            "hparam": hparam,
            "result": result,
            "seed": config.get("seed", None)
        })

    return exps


# ============================================================
# 3. 条件キーの抽出（compare_keys に基づく）
# ============================================================

def extract_condition_key(config, hparam, compare_keys):
    """
    compare_keys に基づいて条件を抽出し、タプル化して返す
    """
    key_dict = {}

    for k in compare_keys:
        # config 側
        if k in config:
            key_dict[k] = config[k]
        # hparam 側
        elif k in hparam:
            key_dict[k] = hparam[k]

    # 辞書をソートしてタプル化（ハッシュ可能にする）
    return tuple(sorted(key_dict.items()))


# ============================================================
# 4. 条件ごとにグループ化
# ============================================================

def group_by_condition(experiments, compare_keys):
    groups = defaultdict(list)

    for exp in experiments:
        key = extract_condition_key(exp["config"], exp["hparam"], compare_keys)
        groups[key].append(exp)

    return groups


# ============================================================
# 5. 指標の集計
# ============================================================

def aggregate_metrics(exps):
    """
    exps: 同じ条件の実験（複数 seed）
    """
    metrics = {
        "train_path_acc": [],
        "val_path_acc": [],
        "test_path_acc": []
    }

    for exp in exps:
        r = exp["result"]
        if r is None:
            continue

        metrics["train_path_acc"].append(
            r["results"]["train"]["accuracy"]["final_train_path_accuracy"]
        )
        metrics["val_path_acc"].append(
            r["results"]["train"]["accuracy"]["final_val_path_accuracy"]
        )
        metrics["test_path_acc"].append(
            r["results"]["inference"]["accuracy"]["test_path_accuracy"]
        )

    summary = {}
    for k, vals in metrics.items():
        if len(vals) == 0:
            continue
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "count": len(vals)
        }

    return summary


# ============================================================
# 6. 保存処理
# ============================================================

def save_summary(summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # JSON 保存
    json_path = os.path.join(out_dir, "seed_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # CSV 保存
    rows = []
    for cond_key, data in summary.items():
        row = {"condition": str(cond_key)}
        for metric, stats in data.items():
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "seed_summary.csv"), index=False)


# ============================================================
# 7. メイン処理
# ============================================================

if __name__ == "__main__":
    # 例: ex_exp_c の seed 集計
    target_group = "ex_exp_c"
    root = os.path.join("experiments", target_group)

    # compare_keys を config.yaml から取得
    base_config = load_config()
    compare_keys = base_config["experiment"]["compare_keys"]

    # 実験読み込み
    experiments = load_experiments(root)

    # 条件ごとにグループ化
    groups = group_by_condition(experiments, compare_keys)

    # 集計
    summary = {}
    for cond_key, exps in groups.items():
        summary[cond_key] = aggregate_metrics(exps)

    # 保存
    out_dir = os.path.join("experiments_summary", target_group)
    save_summary(summary, out_dir)

    # 標準出力
    print("\n=== Seed Summary ===")
    for cond_key, stats in summary.items():
        print("\nCondition:", cond_key)
        for metric, vals in stats.items():
            print(f"  {metric}: mean={vals['mean']:.4f}, std={vals['std']:.4f}, min={vals['min']:.4f}, max={vals['max']:.4f}, count={vals['count']}")