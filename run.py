# Ported from https://github.com/arghosh/AKT/blob/master/run.py
import math

import numpy as np
import torch
from sklearn import metrics

from utils import model_isPid_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transpose_data_model = {"akt"}


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    if mod == "avg":
        return np.average(loss) * (-1.0)
    if mod == "sum":
        return -loss.sum()
    raise AssertionError()


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred = all_pred.copy()
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, q_data, qa_data, pid_data, label):
    net.train()
    pid_flag, model_type = model_isPid_type(params.model)
    n = int(math.ceil(len(q_data) / params.batch_size))
    q_data = q_data.T
    qa_data = qa_data.T
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]
    if pid_flag:
        pid_data = pid_data.T
        pid_data = pid_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    for idx in range(n):
        optimizer.zero_grad()
        q_one_seq = q_data[:, idx * params.batch_size : (idx + 1) * params.batch_size]
        qa_one_seq = qa_data[:, idx * params.batch_size : (idx + 1) * params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx * params.batch_size : (idx + 1) * params.batch_size]

        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :])
            input_qa = np.transpose(qa_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = q_one_seq[:, :]
            input_qa = qa_one_seq[:, :]
            target = qa_one_seq[:, :]
            if pid_flag:
                input_pid = pid_one_seq[:, :]

        target = (target - 1) / params.n_question
        target_1 = np.floor(target)

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target_t = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        if pid_flag:
            loss, pred, true_ct = net(input_q, input_qa, target_t, input_pid)
        else:
            loss, pred, true_ct = net(input_q, input_qa, target_t)

        pred = pred.detach().cpu().numpy()
        loss.backward()

        if getattr(params, "maxgradnorm", -1) and params.maxgradnorm > 0.0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()

        target_flat = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target_flat >= -0.9).tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target_flat[nopadding_index]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return loss, accuracy, auc


def test(net, params, optimizer, q_data, qa_data, pid_data, label):
    pid_flag, model_type = model_isPid_type(params.model)
    net.eval()
    n = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T
    qa_data = qa_data.T
    if pid_flag:
        pid_data = pid_data.T
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []
    count = 0

    for idx in range(n):
        q_one_seq = q_data[:, idx * params.batch_size : (idx + 1) * params.batch_size]
        qa_one_seq = qa_data[:, idx * params.batch_size : (idx + 1) * params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx * params.batch_size : (idx + 1) * params.batch_size]

        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :])
            input_qa = np.transpose(qa_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = q_one_seq[:, :]
            input_qa = qa_one_seq[:, :]
            target = qa_one_seq[:, :]
            if pid_flag:
                input_pid = pid_one_seq[:, :]

        target = (target - 1) / params.n_question
        target_1 = np.floor(target)

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target_t = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        with torch.no_grad():
            if pid_flag:
                loss, pred, ct = net(input_q, input_qa, target_t, input_pid)
            else:
                loss, pred, ct = net(input_q, input_qa, target_t)

        pred = pred.cpu().numpy()
        if (idx + 1) * params.batch_size > seq_num:
            count += seq_num - idx * params.batch_size
        else:
            count += params.batch_size

        target_flat = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target_flat >= -0.9).tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target_flat[nopadding_index]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return loss, accuracy, auc
