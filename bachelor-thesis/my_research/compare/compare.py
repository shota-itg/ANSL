# compare/compare.py

import os
import json
import pandas as pd


def is_group_dir(path):
    return any(
        os.path.isdir(os.path.join(path, name)) and name.startswith("exp_")
        for name in os.listdir(path)
    )


def load_result_json(path):
    result_path = os.path.join(path, "result.json")
    if not os.path.exists(result_path):
        return None

    with open(result_path, "r") as f:
        result_json = json.load(f)

    return {
        "Experiment": os.path.basename(path), 
        "Group": os.path.basename(os.path.dirname(path)), 
        "Result": result_json
    }


def collect_all_results(root="experiments"):
    rows = []

    # root が ex_eval_A のように exp_xxx を直接含む場合
    if is_group_dir(root):
        for exp in os.listdir(root):
            exp_dir = os.path.join(root, exp)
            if os.path.isdir(exp_dir) and exp.startswith("exp_"):
                row = load_result_json(exp_dir)
                if row is not None:
                    rows.append(row)
        return pd.DataFrame(rows)

    # root が experiments の場合 → 全グループを走査
    for group in os.listdir(root):
        group_dir = os.path.join(root, group)
        if not os.path.isdir(group_dir):
            continue

        if not is_group_dir(group_dir):
            continue  # exp_xxx を含まないフォルダはスキップ

        for exp in os.listdir(group_dir):
            exp_dir = os.path.join(group_dir, exp)
            if os.path.isdir(exp_dir) and exp.startswith("exp_"):
                row = load_result_json(exp_dir)
                if row is not None:
                    rows.append(row)

    return pd.DataFrame(rows)


def safe_get(d, keys, default=None):
    """ネストした辞書から安全に値を取り出す"""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


if __name__ == "__main__":
    topo_name = input("どのトポロジ？: ")

    print("\n条件を入力してください．（空エンターでスキップ）")
    seed_input = input("シード: ")
    num_traffic_input = input("再設計の対象トラフィック数(num_traffic): ")
    train_lf_input = input("障害データを含む学習データかどうか(true or false): ")
    md1_dpth_input = input("modality1 の中間層のニューロン数: ")
    md2_dpth_input = input("modality2 の中間層のニューロン数: ")
    fsn_dpth_input = input("fusion 後の中間層のニューロン数: ")

    train_data_input = input("train データセット数: ")
    test_data_input = input("test データセット数: ")
    bs_input = input("バッチサイズ: ")


    compare_dir = os.path.join(
        "experiments", 
        topo_name, 
        f'num_traffic_{num_traffic_input}', 
        f'train_lf_{train_lf_input}', 
    ) # 比較したいexperimentsを指定

    df = collect_all_results(compare_dir)

    # 比較したい結果
    """
    df["seed"] = df["Result"].apply(
        lambda r: safe_get(r, ["config", "seed"])
    )    
    """

    df["lr"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "optimization", "learning_rate"])
    )
    df["Train lf"] = df["Result"].apply(
        lambda r: safe_get(r, ["config", "train", "train_lf_enabled"])
    )
    df["Max lf"] = df["Result"].apply(
        lambda r: safe_get(r, ["config", "train", "max_random_failure"])
    )
    df["Train Data"] = df["Result"].apply(
        lambda r: safe_get(r, ["config", "train", "num_train_data"])
    )
    df["Test Data"] = df["Result"].apply(
        lambda r: safe_get(r, ["config", "test", "num_test_data"])
    )
    df["Train BS"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "optimization", "train_batch_size"])
    )
    df["Infe BS"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "optimization", "inference_batch_size"])
    )
    df["num t"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "architecture", "num_traffic"])
    )
    df["md1 dpth"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "architecture", "modality1", "depth"])
    )
    df["md2 dpth"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "architecture", "modality2", "depth"])
    )
    df["md2 In"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "architecture", "modality2", "inputs_dim"])
    )
    df["fsn dpth"] = df["Result"].apply(
        lambda r: safe_get(r, ["hyperparameters", "architecture", "afterfusion", "depth"])
    )
    df["Train Path Acc"] = df["Result"].apply(
        lambda r: safe_get(r, ["results", "train", "accuracy", "final_train_path_accuracy"])
    )
    df["Val Path Acc"] = df["Result"].apply(
        lambda r: safe_get(r, ["results", "train", "accuracy", "final_val_path_accuracy"])
    )
    df["Test Path Acc"] = df["Result"].apply(
        lambda r: safe_get(r, ["results", "inference", "accuracy", "test_path_accuracy"])
    )
    df["Routing Success Rate"] = df["Result"].apply(
        lambda r: safe_get(r, ["results", "routing_success_rate", "accuracy", "routing_success_rate"])
    )

    """
    # ソート
    df = df.sort_values(
        by=[
            "Val Path Acc", 
            "lr", 
            "Train lf", 
            "Num lf", 
            "Train Data", 
            "Test Data", 
            "Train BS", 
            "Infe BS", 
            "Train Path Acc", 
            "Test Path Acc", 
            "Routing Success Rate"
        ], 
        ascending=[
            False, 
            True, 
            True, 
            True, 
            True, 
            True, 
            True, 
            True, 
            False, 
            False, 
            False
        ]
    )    
    """


    df_export = df[[
        # "Group", 
        # "seed", 
        "num t", 
        "md2 In", 
        "md1 dpth", 
        "md2 dpth", 
        "fsn dpth", 
        "lr", 
        "Train lf", 
        "Max lf", 
        "Train Data", 
        "Test Data", 
        "Train BS", 
        "Infe BS", 
        "Train Path Acc", 
        "Val Path Acc", 
        "Test Path Acc", 
        "Routing Success Rate", 
        "Experiment"
    ]]

    """
    print(df_export)
    print(df_export["md2 In"])
    print(df_export["md1 dpth"])
    print(df_export["md2 dpth"])
    print(df_export["fsn dpth"])
    print(df_export["Train lf"])
    print(df_export["Train Data"])
    print(df_export["Test Data"])
    print(df_export["Train BS"])
    """


    """
    if seed_input:
        df_export = df_export[df_export["seed"] == int(seed_input)]
    """
    if num_traffic_input:
        df_export = df_export[df_export["num t"] == int(num_traffic_input)]
    if md1_dpth_input:
        df_export = df_export[df_export["md1 dpth"] == int(md1_dpth_input)]
    if md2_dpth_input:
        df_export = df_export[df_export["md2 dpth"] == int(md2_dpth_input)]
    if fsn_dpth_input:
        df_export = df_export[df_export["fsn dpth"] == int(fsn_dpth_input)]
    if train_lf_input:
        if train_lf_input == "true":
            df_export = df_export[df_export["Train lf"] == bool(train_lf_input)]
        elif train_lf_input == "false":
            train_lf = ""
            df_export = df_export[df_export["Train lf"] == bool(train_lf)]
    if train_data_input:
        df_export = df_export[df_export["Train Data"] == int(train_data_input)]
    if test_data_input:
        df_export = df_export[df_export["Test Data"] == int(test_data_input)]
    if bs_input:
        df_export = df_export[df_export["Train BS"] == int(bs_input)]          
    """
  
    """


    df_export = df_export.sort_values(by="Test Path Acc", ascending=False)

    print(df_export)

    # CSV 保存
    # df.to_csv("experiments/compare_results.csv", index=False, encoding="utf-8-sig")

    # Excel 形式で保存（列幅自動調整）
    excel_path = os.path.join(
        "experiments", 
        topo_name, 
        f'num_traffic_{num_traffic_input}', 
        f'train_lf_{train_lf_input}', 
        "compare_results.xlsx"
    )

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name="results")

        worksheet = writer.sheets["results"]

        # 列幅を自動調整
        for i, col in enumerate(df_export.columns):
            max_len = max(
                df_export[col].astype(str).map(len).max(),
                len(col)
            )
            worksheet.set_column(i, i, max_len + 2)

    print(f'結果を保存: {excel_path}')