import json

with open("results/change_detection/bce_dice_metrics.json") as f:
    hist = json.load(f)["history"]
    
with open("results/change_detection/independent_validation.json") as f:
    ind_val = json.load(f)

old_iou = "0.0000"
old_f1 = "0.0000"
old_pos = "0.0000%"

new_iou = f"{ind_val['iou']:.4f}"
new_f1 = f"{ind_val['f1']:.4f}"
new_prec = f"{ind_val['precision']:.4f}"
new_rec = f"{ind_val['recall']:.4f}"
new_pos = f"{ind_val['predicted_positive_pct']:.4f}%"

with open("datasets/PHASE_4B_REPORT.md", "r") as f:
    report = f.read()

report = report.replace("|           0.0000 |              [result] |", f"|           0.0000 |              {new_iou} |", 1)
report = report.replace("|           0.0000 |              [result] |", f"|           0.0000 |              {new_f1} |", 1)
report = report.replace("|           0.0000 |              [result] |", f"|           0.0000 |              {new_prec} |", 1)
report = report.replace("|           0.0000 |              [result] |", f"|           0.0000 |              {new_rec} |", 1)
report = report.replace("|          0.0000% |              [result] |", f"|          0.0000% |              {new_pos} |", 1)

with open("datasets/PHASE_4B_REPORT.md", "w") as f:
    f.write(report)
print("Report formatted successfully.")
