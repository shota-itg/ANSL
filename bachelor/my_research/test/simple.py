import os, json

root = "experiments"
results = []

for exp in os.listdir(root):
    json_path = os.path.join(root, exp, "results.json")
    if not os.path.exists(json_path):
        continue

    with open(json_path) as f:
        data = json.load(f)

    train_res = data["results"]["train"]

    results.append({
        "exp": exp,
        "val_loss": train_res["loss"]["final_val_loss"],
        "val_demand_acc": train_res["accuracy"]["final_test_demand_accuracy"],
        "val_path_acc": train_res["accuracy"]["final_test_path_accuracy"],
        "val_elem_acc": train_res["accuracy"]["final_test_element_accuracy"],
    })

for r in sorted(results, key=lambda x: x["val_loss"]):
    print(r)
