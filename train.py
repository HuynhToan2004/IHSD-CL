import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
import time

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

import train_config as train_config
from dataset_impcon import get_dataloader
from util import iter_product
from sklearn.metrics import f1_score
import loss_impcon as loss
from model import primary_encoder_v2_no_pooler_for_con, weighting_network

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup 
from torch.optim import AdamW

from tqdm import tqdm
import pandas as pd
import copy

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def train(epoch, train_loader, model_main, loss_function, optimizer, lr_scheduler, log,
          model_momentum, queue_features, queue_labels):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_main.to(device)
    model_main.train()
    if log.param.momentum > 0.0 and model_momentum is not None:
        model_momentum.to(device)
        model_momentum.eval()

    total_true, total_pred_1, acc_curve_1 = [], [], []
    loss_curve = []
    train_loss_1 = 0.0
    total_epoch_acc_1 = 0.0
    steps = 0
    start_train_time = time.time()

    sim = Similarity(log.param.temperature)

    if log.param.w_aug:
        if log.param.w_double:
            train_batch_size = log.param.train_batch_size * 3
        else:
            train_batch_size = log.param.train_batch_size * 2
    else:
        train_batch_size = log.param.train_batch_size

    print("train with aug:", log.param.w_aug)
    print("train with double aug:", log.param.w_double)
    print("train with separate double aug:", log.param.w_separate)
    print("loss with sup(using label info):", log.param.w_sup)
    print("len(train_loader):", len(train_loader))
    print("train_batch_size including augmented posts/implications:", train_batch_size)
    if log.param.w_separate:
        assert log.param.w_double, "w_double should be set to True for w_separate=True option"

    # đảm bảo queue cùng device
    if queue_features is None:
        queue_features = torch.zeros((0, 768), device=device)
    else:
        queue_features = queue_features.to(device)
    if queue_labels is None:
        queue_labels = torch.zeros((0, 1), device=device, dtype=torch.long)
    else:
        queue_labels = queue_labels.to(device)

    cls_tokens = []

    for idx, batch in enumerate(train_loader):
        if "ViIHSD" in log.param.dataset or "ihc" in log.param.dataset or "sbic" in log.param.dataset or 'dynahate' in log.param.dataset or 'toxigen' in log.param.dataset:
            text_name = "text"
            label_name = "label"
        else:
            raise NotImplementedError

        text = batch[text_name]
        attn = batch[text_name + "_attn_mask"]
        label = torch.tensor(batch[label_name]).long()

        if label.size(0) != train_batch_size:
            continue  # bỏ batch lẻ

        text = text.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # -----------------------------
        # w_aug / no_aug forward
        # -----------------------------
        if log.param.w_aug:
            # tách original / augmented
            assert log.param.train_batch_size == label.shape[0] // 2
            assert label.shape[0] % 2 == 0

            original_label, augmented_label = torch.split(
                label, [log.param.train_batch_size, log.param.train_batch_size], dim=0
            )
            only_original_labels = original_label  # [B]

            original_text, augmented_text = torch.split(
                text, [log.param.train_batch_size, log.param.train_batch_size], dim=0
            )
            original_attn, augmented_attn = torch.split(
                attn, [log.param.train_batch_size, log.param.train_batch_size], dim=0
            )

            original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(
                original_text, original_attn
            )

            # tạo augmented feature tuỳ loss/aug_type
            if log.param.loss_type in ["CE", "UCL", "SCL", "SupConLoss_Original"]:
                _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn)
            elif log.param.loss_type == "ImpCon":
                _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(augmented_text, augmented_attn)
            elif log.param.aug_type == "Dropout":
                _, augmented_supcon_feature_1 = model_momentum.get_cls_features_ptrnsp(original_text, original_attn)
            elif log.param.aug_type == "Augmentation":
                _, augmented_supcon_feature_1 = model_momentum.get_cls_features_ptrnsp(augmented_text, augmented_attn)
            else:
                _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(augmented_text, augmented_attn)

            # -----------------------------
            # Momentum & MoCo queue (đa lớp)
            # -----------------------------
            if log.param.loss_type not in ["CE", "UCL", "SCL", "SupConLoss_Original", "ImpCon", "AugCon"]:
                with torch.no_grad():
                    _, original_supcon_feature_momentum = model_momentum.get_cls_features_ptrnsp(
                        original_text, original_attn
                    )
                    _, augmented_supcon_feature_momentum = model_momentum.get_cls_features_ptrnsp(
                        augmented_text, augmented_attn
                    )

                    queue_features = torch.cat((queue_features, original_supcon_feature_momentum.detach()), dim=0)
                    queue_labels = torch.cat((queue_labels, original_label.view(-1, 1)), dim=0)
                    if queue_features.shape[0] > log.param.queue_size:
                        cut = queue_features.shape[0] - log.param.queue_size
                        queue_features = queue_features[cut:, :]
                        queue_labels = queue_labels[cut:, :]

                    moco_features = queue_features  # [Q, D]
                    moco_labels_cls = queue_labels.squeeze(-1)  # [Q] (giá trị lớp 0/1/2)

            k = log.param.hard_neg_k

            # mask “cùng nhãn” đa lớp cho batch-batch
            # anchor_labels: [B, B], 1 nếu cùng lớp, 0 nếu khác
            anchor_labels = (only_original_labels.unsqueeze(1) == only_original_labels.unsqueeze(0)).long()

            # nếu dùng MoCo
            if log.param.loss_type not in ["CE", "UCL", "SCL", "SupConLoss_Original", "ImpCon"]:
                with torch.no_grad():
                    if moco_features.shape[0] > int(log.param.queue_size) * 1 / 4:
                        moco_features = moco_features.to(device)
                        moco_labels_cls = moco_labels_cls.to(device)  # [Q]
                        moco_original_labels = moco_labels_cls  # lưu lại nhãn thật của queue

                        # cosine sim anchors (B,D) với moco (Q,D) -> [B,Q]
                        moco_sim = sim(original_supcon_feature_1.unsqueeze(1), moco_features.unsqueeze(0))

                        # concat nhãn batch + queue để tạo mask tổng
                        labels_concat_moco = torch.cat((only_original_labels, moco_labels_cls), dim=0)  # [B+Q]

                        # target mask: [B, B+Q] = 1 nếu cùng lớp, 0 nếu khác
                        target = (only_original_labels.unsqueeze(1) == labels_concat_moco.unsqueeze(0)).long()

                        # tách: batch-batch (anchor_labels) & batch-queue (moco_pos_mask)
                        anchor_labels = target[:, :only_original_labels.shape[0]]          # [B,B] (đè lại bằng mask đúng)
                        moco_pos_mask = target[:, only_original_labels.shape[0]:]          # [B,Q] 1:pos, 0:neg

                        # chỉ giữ NEG cho hard-neg mining
                        cos_sim_moco_hard_neg = moco_sim * (1 - moco_pos_mask)

                        # Weighting (dựa trên logits của trọng số cho class của anchor)
                        if str(log.param.moco_weight) == "True":
                            neg_weight_logits = model_momentum(moco_features)  # [Q, C]
                            neg_weight = F.softmax(neg_weight_logits, dim=1)   # [Q, C]
                            # lấy cột theo nhãn anchor: result [Q, B] rồi transpose -> [B, Q]
                            per_anchor_cols = neg_weight[:, only_original_labels]  # [Q, B]
                            neg_weight_per_anchor = per_anchor_cols.transpose(0, 1)  # [B, Q]
                            cos_sim_moco_hard_neg = cos_sim_moco_hard_neg * neg_weight_per_anchor

                        # mask 0 thành -inf-like để topk bỏ qua
                        cos_sim_moco_hard_neg = torch.where(
                            moco_pos_mask.bool(), torch.full_like(cos_sim_moco_hard_neg, -999), cos_sim_moco_hard_neg
                        )

                        # top-k theo hàng (theo từng anchor)
                        hard_neg_idx = torch.topk(cos_sim_moco_hard_neg, k, dim=1).indices  # [B, k]

                        # gather hard-neg features: [B, k, D]
                        B, D = moco_features.size(0), moco_features.size(1)
                        hard_neg_features = moco_features[hard_neg_idx]  # advanced indexing -> [B, k, D]

            supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1], dim=0)
            assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size

            pred_1 = model_main(original_last_layer_hidden_states)

        else:
            # không augment
            assert log.param.train_batch_size == label.shape[0]
            only_original_labels = label  # [B]
            last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text, attn)
            pred_1 = model_main(last_layer_hidden_states)

        # -----------------------------
        # TÍNH LOSS
        # -----------------------------
        if log.param.w_aug and log.param.w_sup:
            if log.param.w_double:
                if log.param.w_separate:
                    raise NotImplementedError
                else:
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                             ((1 - loss_function["lambda_loss"]) * loss_function["contrastive_for_double"](supcon_feature_1, label))
            else:
                if log.param.loss_type == "CE":
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels))
                elif log.param.loss_type == "UCL":
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                             ((1 - loss_function["lambda_loss"]) * loss_function["ucl"](supcon_feature_1))
                elif log.param.loss_type == "ImpCon":
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                             ((1 - loss_function["lambda_loss"]) * loss_function["ImpCon"](supcon_feature_1))
                elif log.param.loss_type == "SCL":
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                             ((1 - loss_function["lambda_loss"]) * loss_function["SupConLoss"](supcon_feature_1, label))
                elif log.param.loss_type == "SupConLoss_Original":
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                             ((1 - loss_function["lambda_loss"]) * loss_function["SupConLoss_Original"](supcon_feature_1, label))
                elif log.param.loss_type == "Ours":
                    if 'moco_features' in locals() and moco_features.shape[0] > int(log.param.queue_size) * 1 / 4:
                        # anchor_labels là mask [B,B]; moco_pos_mask là [B,Q]
                        loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                                 ((1 - loss_function["lambda_loss"]) * loss_function["Ours"](
                                     supcon_feature_1, label,
                                     None, hard_neg_features,
                                     None, moco_pos_mask,  # dùng mask POS cho moco
                                     anchor_labels, moco_features, moco_original_labels
                                 ))
                    else:
                        loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                                 ((1 - loss_function["lambda_loss"]) * loss_function["Ours"](
                                     supcon_feature_1, label, None, None, None, None, anchor_labels
                                 ))
                else:
                    raise NotImplementedError
        elif log.param.w_aug and not log.param.w_sup:
            if log.param.w_double:
                if log.param.w_separate:
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                             ((0.5 * (1 - loss_function["lambda_loss"])) * loss_function["contrastive"](supcon_feature_1)) + \
                             ((0.5 * (1 - loss_function["lambda_loss"])) * loss_function["contrastive"](supcon_feature_2))
                else:
                    loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                             ((1 - loss_function["lambda_loss"]) * loss_function["contrastive_for_double"](supcon_feature_1))
            else:
                loss_1 = (loss_function["lambda_loss"] * loss_function["ce_loss"](pred_1, only_original_labels)) + \
                         ((1 - loss_function["lambda_loss"]) * loss_function["contrastive"](supcon_feature_1))
        else:
            loss_1 = loss_function["ce_loss"](pred_1, only_original_labels)

        loss = loss_1
        train_loss_1 += loss_1.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)

        optimizer.step()
        model_main.zero_grad()
        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print(f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss_1: {loss_1.item():.6f}, '
                  f'Time taken: {((time.time()-start_train_time)/60): .2f} min')
            start_train_time = time.time()

        # thống kê
        true_list = only_original_labels.detach().cpu().tolist()
        total_true.extend(true_list)

        preds = torch.argmax(pred_1, dim=1)
        num_corrects_1 = (preds == only_original_labels).float().sum()
        pred_list_1 = preds.detach().cpu().tolist()
        total_pred_1.extend(pred_list_1)

        acc_1 = 100.0 * (num_corrects_1 / only_original_labels.size(0))
        acc_curve_1.append(float(acc_1.item()))
        loss_curve.append(loss_1.item())
        total_epoch_acc_1 += float(acc_1.item())

        # momentum update
        if log.param.momentum > 0.:
            with torch.no_grad():
                for param, param_m in zip(model_main.parameters(), model_momentum.parameters()):
                    param_m.data = param_m.data * log.param.momentum + param.data * (1. - log.param.momentum)
                    param_m.requires_grad = False

    print(train_loss_1 / len(train_loader))
    print(total_epoch_acc_1 / len(train_loader))

    return (train_loss_1 / len(train_loader),
            total_epoch_acc_1 / len(train_loader),
            acc_curve_1, cls_tokens, loss_curve, queue_features, queue_labels)



def test(test_loader, model_main, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_main.eval()
    model_main.to(device)

    total_pred_1, total_true, total_pred_prob_1 = [], [], []
    save_pred = {"true": [], "pred_1": [], "pred_prob_1": [], "feature": []}

    total_feature = []
    total_num_corrects = 0
    total_num = 0
    print(len(test_loader))

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if "ihc" in log.param.dataset or "dynahate" in log.param.dataset \
               or "sbic" in log.param.dataset or "toxigen" in log.param.dataset or "ViIHSD" in log.param.dataset:
                text_name = "text"
                label_name = "label"
            else:
                text_name = "cause"
                label_name = "emotion"
                raise NotImplementedError

            text = batch[text_name]
            attn = batch[text_name + "_attn_mask"]
            label = torch.tensor(batch[label_name]).long()

            text = text.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text, attn)
            pred_1 = model_main(last_layer_hidden_states)
            softmaxed_tensor = F.softmax(pred_1, dim=1)

            preds = torch.argmax(pred_1, dim=1)
            num_corrects_1 = (preds == label).float().sum()

            pred_list_1 = preds.detach().cpu().tolist()
            true_list = label.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            # lưu thống kê
            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature_1.detach().cpu().tolist())
            total_pred_prob_1.extend(pred_1.detach().cpu().tolist())

    # F1 cho 3 lớp
    f1_macro = f1_score(total_true, total_pred_1, labels=[0, 1, 2], average="macro", zero_division=0)
    f1_weighted = f1_score(total_true, total_pred_1, labels=[0, 1, 2], average="weighted", zero_division=0)
    f1_score_1 = {"macro": f1_macro, "weighted": f1_weighted}

    total_acc = 100.0 * total_num_corrects / max(1, total_num)

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1
    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1

    return total_acc, f1_score_1, save_pred

def cl_train(log):
    # ==== seed & determinism ====
    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(log.param.SEED)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # ==== bảo đảm 3 lớp ====
    log.param.label_size = 3

    print("#######################start run#######################")
    print("log:", log)

    train_data, valid_data, test_data = get_dataloader(
        log.param.train_batch_size,
        log.param.eval_batch_size,
        log.param.dataset,
        w_aug=log.param.w_aug,
        w_double=log.param.w_double,
        label_list=None,
        cls_tokens=None,
        model_type=log.param.model_type
    )
    print("len(train_data):", len(train_data))

    # ==== losses ====
    losses = {
        "ucl": loss.UnSupConLoss(temperature=log.param.temperature),
        "Ours": loss.Ours(temperature=log.param.temperature),
        "ImpCon": loss.ImpCon(temperature=log.param.temperature),
        "SupConLoss": loss.SupConLoss(temperature=log.param.temperature),
        "SupConLoss_Original": loss.SupConLoss_Original(temperature=log.param.temperature),
        "ce_loss": nn.CrossEntropyLoss(),     # đa lớp
        # "bce_loss": nn.BCEWithLogitsLoss(), # không cần cho 3 lớp (có thể xoá)
        "lambda_loss": log.param.lambda_loss,
    }

    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    model_main = primary_encoder_v2_no_pooler_for_con(
        log.param.hidden_size, log.param.label_size, log.param.model_type
    )

    # momentum model (nếu dùng)
    if log.param.momentum > 0.:
        model_momentum = primary_encoder_v2_no_pooler_for_con(
            log.param.hidden_size, log.param.label_size, log.param.model_type
        )
        for param, param_m in zip(model_main.parameters(), model_momentum.parameters()):
            param_m.data.copy_(param.data)      # initialize
            param_m.requires_grad = False       # not update by gradient
    else:
        model_momentum = None

    # ==== để train() tự khởi tạo queue theo đúng device ====
    queue_features = None
    queue_labels = None

    # ==== optimizer / scheduler ====
    total_params = list(model_main.named_parameters())
    num_training_steps = int(len(train_data) * log.param.nepoch)  # len(loader) = số batch
    optimizer_grouped_parameters = [
        {'params': [p for n, p in total_params], 'weight_decay': log.param.decay}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # ==== save path ====
    if log.param.run_name != "":
        save_home = (
            "./save/"
            + str(log.param.SEED) + "/"
            + log.param.dataset + "/"
            + log.param.run_name + "/"
            + log.param.loss_type + log.param.dir_name + "/"
            + model_run_time + "/"
        )
    else:
        save_home = (
            "./save/"
            + str(log.param.SEED) + "/"
            + log.param.dataset + "/"
            + log.param.loss_type + log.param.dir_name + "/"
            + model_run_time + "/"
        )

    total_train_acc_curve_1 = []
    total_train_loss_curve = []

    best_criterion = 0.0  # theo dõi best macro-F1

    for epoch in range(1, log.param.nepoch + 1):
        # train 1 epoch
        train_loss_1, train_acc_1, train_acc_curve_1, cls_tokens, train_loss_curve, queue_features, queue_labels = \
            train(epoch, train_data, model_main, losses, optimizer, lr_scheduler, log,
                  model_momentum, queue_features, queue_labels)

        # eval
        val_acc_1, val_f1_1, val_save_pred = test(valid_data, model_main, log)
        test_acc_1, test_f1_1, test_save_pred = test(test_data, model_main, log)

        # lưu curve
        total_train_acc_curve_1.extend(train_acc_curve_1)
        total_train_loss_curve.extend(train_loss_curve)

        print('====> Epoch: {} Train loss_1: {:.4f}'.format(epoch, train_loss_1))

        # ==== ghi file JSON đúng chuẩn ====
        os.makedirs(save_home, exist_ok=True)
        with open(os.path.join(save_home, "acc_curve.json"), 'w') as fp:
            json.dump({"train_acc_curve_1": total_train_acc_curve_1}, fp, indent=4)
        with open(os.path.join(save_home, "loss_curve.json"), 'w') as fp:
            json.dump({
                "train_loss_epoch": epoch,
                "train_loss_curve": total_train_loss_curve
            }, fp, indent=4)

        # chọn best theo macro-F1
        is_best = val_f1_1["macro"] > best_criterion
        best_criterion = max(val_f1_1["macro"], best_criterion)

        print("Best model evaluated by macro f1")
        print(f'Valid Accuracy: {val_acc_1:.3f} Valid F1: {val_f1_1["macro"]:.4f}')
        print(f'Test  Accuracy: {test_acc_1:.3f} Test  F1: {test_f1_1["macro"]:.4f}')

        if is_best:
            print("======> Best epoch <======")
            log.train_loss_1 = train_loss_1
            log.stop_epoch = epoch
            log.valid_f1_score_1 = val_f1_1
            log.test_f1_score_1 = test_f1_1
            log.valid_accuracy_1 = val_acc_1
            log.test_accuracy_1 = test_acc_1
            log.train_accuracy_1 = train_acc_1

            # save log
            with open(os.path.join(save_home, "log.json"), 'w') as fp:
                json.dump(dict(log), fp, indent=4)

            # save model
            if log.param.save:
                torch.save(model_main.state_dict(), os.path.join(save_home, 'model.pt'))
                print(f"best model is saved at {os.path.join(save_home, 'model.pt')}")

##################################################################################################

if __name__ == '__main__':

    tuning_param = train_config.tuning_param

    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        log.param.label_size = 3
        
        cl_train(log)

