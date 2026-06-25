import os, json, csv

root = "experiments"
rows = []

for exp in os.listdir(root):
    json_path = os.path.join(root, exp, "results.json")
    if not os.path.exists(json_path):
        continue

    with open(json_path) as f:
        data = json.load(f)

    train_res = data["results"]["train"]

    rows.append([
        exp,
        train_res["loss"]["final_train_loss"],
        train_res["loss"]["final_val_loss"],
        train_res["accuracy"]["final_test_demand_accuracy"],
        train_res["accuracy"]["final_test_path_accuracy"],
        train_res["accuracy"]["final_test_element_accuracy"],
    ])

with open("experiment_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Experiment",
        "Train Loss",
        "Val Loss",
        "Val Demand Acc",
        "Val Path Acc",
        "Val Element Acc",
    ])
    writer.writerows(rows)

print("Saved experiment_summary.csv")
