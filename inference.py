import os, json, torch, pandas as pd
from easydict import EasyDict as edict
from sklearn.metrics import classification_report, confusion_matrix
from model import primary_encoder_v2_no_pooler_for_con
from dataset_impcon import get_dataloader
from train import test   # tái dùng hàm test() bạn đã có
import numpy as np

# 1) CHỈNH đường dẫn tới thư mục đã lưu model (thư mục có model.pt, log.json)
SAVE_DIR = "/implicit-hatespeech-detection/LAHN/save/0/ViIHSD/best/Ours_0401_additional_seed/2025_10_04_14_18_39"  # <-- ĐỔI CHO ĐÚNG

# 2) Khôi phục log/param (hoặc tự set thủ công nếu không có log.json)
log_path = os.path.join(SAVE_DIR, "log.json")
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log = edict(json.load(f))
else:

    from easydict import EasyDict as edict
    log = edict()
    log.param = edict({
        "dataset": "ViIHSD",
        "model_type": "/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/phobert-base",  # hoặc đường dẫn PhoBERT local của bạn
        "hidden_size": 768,
        "eval_batch_size": 64,
        "train_batch_size": 16,
        "label_size": 3,          # 3 nhãn
        "w_aug": False,
        "w_double": False,
    })

log.param.label_size = 3

_, _, test_loader = get_dataloader(
    train_batch_size=log.param.train_batch_size,
    eval_batch_size=log.param.eval_batch_size,
    dataset=log.param.dataset,
    w_aug=False,                 
    w_double=False,
    label_list=None,
    cls_tokens=None,
    model_type=log.param.model_type
)

# 4) Tạo model và nạp trọng số
model = primary_encoder_v2_no_pooler_for_con(
    hidden_size=log.param.hidden_size,
    emotion_size=log.param.label_size,
    encoder_type=log.param.model_type
)
state = torch.load(os.path.join(SAVE_DIR, "model.pt"), map_location="cpu")
model.load_state_dict(state)

# 5) Chạy test() để lấy kết quả và lưu CSV
acc, f1, save_pred = test(test_loader, model, log)
print("Test Accuracy:", acc)
print("Test F1:", f1)

y_true = np.array(save_pred["true"], dtype=int)
y_pred = np.array(save_pred["pred_1"], dtype=int)

target_names = ["class_0", "class_1", "class_2"]
report_dict = classification_report(
    y_true, y_pred,
    labels=[0,1,2],
    target_names=target_names,
    output_dict=True,
    zero_division=0,
    digits=4
)
report_df = pd.DataFrame(report_dict).transpose()
report_path_csv = os.path.join(SAVE_DIR, "classification_report.csv")
report_df.to_csv(report_path_csv, float_format="%.4f")
print("Saved classification report ->", report_path_csv)

print(pd.DataFrame(report_dict).transpose().round(4))

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in [0,1,2]],
                        columns=[f"pred_{i}" for i in [0,1,2]])
cm_csv = os.path.join(SAVE_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_csv)
print("Saved confusion matrix ->", cm_csv)

report_json = os.path.join(SAVE_DIR, "classification_report.json")
with open(report_json, "w") as f:
    json.dump(report_dict, f, indent=2)
print("Saved JSON report ->", report_json)