"""
训练入口：公开 AKT (arghosh/AKT) 逻辑 + 本仓库 data/<dataset>/TrainSet|ValSet|TestSet 下 CSV。
CSV 列：skill_seq, problem_seq, correct_seq（与原先 preprocess 一致）。
"""
import argparse
import glob
import os
import os.path
import time

import numpy as np
import torch

from load_data import CSV_PID_DATA
from run import device, test, train
from utils import get_file_name_identifier, load_model, try_makedirs


def train_one_dataset(
    params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid
):
    model = load_model(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    print("\n")
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0.0
    best_epoch = 1

    for idx in range(params.max_iter):
        t0 = time.time()
        train_loss, train_accuracy, train_auc = train(
            model, params, optimizer, train_q_data, train_qa_data, train_pid, label="Train"
        )
        valid_loss, valid_accuracy, valid_auc = test(
            model, params, optimizer, valid_q_data, valid_qa_data, valid_pid, label="Valid"
        )
        t1 = time.time()

        print("epoch", idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy, "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)
        print("training time: %.2f seconds" % (t1 - t0))

        try_makedirs("model")
        try_makedirs(os.path.join("model", params.model))
        try_makedirs(os.path.join("model", params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        if valid_auc > best_valid_auc:
            path = os.path.join("model", params.model, params.save, file_name) + "_*"
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx + 1
            torch.save(
                {
                    "epoch": idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                },
                os.path.join("model", params.model, params.save, file_name) + "_" + str(idx + 1),
            )
        if idx - best_epoch >= params.early_stop:
            break

    try_makedirs("result")
    try_makedirs(os.path.join("result", params.model))
    try_makedirs(os.path.join("result", params.model, params.save))
    f_save_log = open(os.path.join("result", params.model, params.save, file_name), "w")
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch


def test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid, best_epoch):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params)

    ckpt = os.path.join("model", params.model, params.save, file_name) + "_" + str(best_epoch)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy, test_auc = test(
        model, params, None, test_q_data, test_qa_data, test_pid, label="Test"
    )
    print("\ntest_auc\t", test_auc)
    print("test_accuracy\t", test_accuracy)
    print("test_loss\t", test_loss)

    path = os.path.join("model", params.model, params.save, file_name) + "_*"
    for i in glob.glob(path):
        os.remove(i)


if __name__ == "__main__":  # 同技能tail
    parser = argparse.ArgumentParser(description="AKT on local CSV (skill / problem / correct)")
    parser.add_argument("--max_iter", type=int, default=300, help="number of iterations")
    parser.add_argument("--early_stop", type=int, default=10, help="stop if no valid AUC gain for N epochs")
    parser.add_argument("--train_set", type=int, default=1)
    parser.add_argument("--seed", type=int, default=224, help="default seed")

    parser.add_argument("--optim", type=str, default="adam", help="Default Optimizer")
    parser.add_argument("--batch_size", type=int, default=24, help="the batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--maxgradnorm", type=float, default=-1.0, help="maximum gradient norm; <=0 to disable")
    parser.add_argument("--final_fc_dim", type=int, default=512, help="hidden state dim for final fc layer")

    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model")
    parser.add_argument("--d_ff", type=int, default=1024, help="Transformer d_ff")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--n_block", type=int, default=2, help="number of blocks (AKT blocks_1 depth)")
    parser.add_argument("--n_head", type=int, default=8, help="number of heads in multihead attention")
    parser.add_argument("--kq_same", type=int, default=1)
    parser.add_argument("--l2", type=float, default=1e-5, help="l2 penalty for difficulty (pid)")
    parser.add_argument(
        "--counterfactual_blend",
        type=int,
        default=1,
        help="1: enable KT-CF-Lite regularization (sparse trigger + same-skill-tail consistency)",
    )
    parser.add_argument("--cf_slip_pred_thresh", type=float, default=0.55)
    parser.add_argument("--cf_same_skill_eps", type=float, default=0.08)
    parser.add_argument("--cf_min_same_skill_tail", type=int, default=2)
    parser.add_argument(
        "--cf_train",
        type=int,
        default=1,
        help="1: enable KT-CF-Lite regularization during training",
    )
    parser.add_argument(
        "--cf_lambda",
        type=float,
        default=0.04,
        help="weight of counterfactual consistency regularization",
    )
    parser.add_argument(
        "--cf_anti_lambda",
        type=float,
        default=0.01,
        help="weight of anti-intervention symmetry regularization",
    )
    parser.add_argument(
        "--cf_trigger_ratio",
        type=float,
        default=0.1,
        help="top uncertain sample ratio for CF trigger per batch",
    )
    parser.add_argument(
        "--cf_uncertain_margin",
        type=float,
        default=0.2,
        help="uncertainty threshold around 0.5 for sparse CF trigger",
    )
    parser.add_argument(
        "--cf_consistency_margin",
        type=float,
        default=0.005,
        help="margin in same-skill-tail consistency loss",
    )
    parser.add_argument(
        "--cf_anti_margin",
        type=float,
        default=0.005,
        help="symmetry slack for do(r=1)/do(r=0) effects",
    )
    parser.add_argument("--cf_sde_samples", type=int, default=24, help="particle count for SDE trajectories")
    parser.add_argument("--cf_sde_steps", type=int, default=6, help="Euler-Maruyama steps")
    parser.add_argument("--cf_sde_dt", type=float, default=0.25, help="SDE time step")
    parser.add_argument("--cf_sde_theta", type=float, default=1.1, help="OU drift strength")
    parser.add_argument("--cf_sde_base_sigma", type=float, default=0.08, help="base diffusion")
    parser.add_argument(
        "--cf_sde_uncertainty_scale",
        type=float,
        default=0.2,
        help="extra diffusion scale from prediction uncertainty",
    )
    parser.add_argument(
        "--cf_sde_shift_weight",
        type=float,
        default=0.35,
        help="weight for distribution spread term in shift metric",
    )

    parser.add_argument("--model", type=str, default="akt_pid", help="akt_pid recommended for our CSV")
    parser.add_argument("--dataset", type=str, default="assist2009")
    params = parser.parse_args()
    dataset = params.dataset

    if dataset in {"assist2009"}:
        params.seqlen = 200
        params.data_dir = "data/" + dataset
        params.data_name = dataset
        params.n_question = 107
        params.n_pid = 9798

    elif dataset in {"assist2017"}:
        params.seqlen = 200
        params.data_dir = "data/" + dataset
        params.data_name = dataset
        params.n_question = 97
        params.n_pid = 2521

    elif dataset in {"assist2012"}:
        params.seqlen = 200
        params.data_dir = "data/" + dataset
        params.data_name = dataset
        params.n_question = 254
        params.n_pid = 37438

    elif dataset in {"eedi"}:
        params.seqlen = 200
        params.data_dir = "data/" + dataset
        params.data_name = dataset
        params.n_question = 52
        params.n_pid = 915

    else:
        raise ValueError("Unknown dataset: %s" % dataset)

    params.save = params.data_name
    params.load = params.data_name

    dat = CSV_PID_DATA(n_question=params.n_question, seqlen=params.seqlen)
    seed_num = params.seed
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    file_name_identifier = get_file_name_identifier(params)
    d = vars(params)
    for key in d:
        print("\t", key, "\t", d[key])
    file_name = ""
    for item_ in file_name_identifier:
        file_name = file_name + item_[0] + str(item_[1])

    train_dir = os.path.join(params.data_dir, "TrainSet")
    valid_dir = os.path.join(params.data_dir, "ValSet")
    test_dir = os.path.join(params.data_dir, "TestSet")

    train_data_path = os.path.join(train_dir, params.data_name + "_train" + str(params.train_set) + ".csv")
    valid_data_path = os.path.join(valid_dir, params.data_name + "_valid" + str(params.train_set) + ".csv")
    test_data_path = os.path.join(test_dir, params.data_name + "_test" + str(params.train_set) + ".csv")

    train_q_data, train_qa_data, train_pid = dat.load_data(train_data_path)
    valid_q_data, valid_qa_data, valid_pid = dat.load_data(valid_data_path)
    test_q_data, test_qa_data, test_pid = dat.load_data(test_data_path)

    print("\n")
    print("train_q_data.shape", train_q_data.shape)
    print("train_qa_data.shape", train_qa_data.shape)
    print("train_pid.shape", train_pid.shape)
    print("valid_q_data.shape", valid_q_data.shape)
    print("valid_qa_data.shape", valid_qa_data.shape)
    print("\n")

    best_epoch = train_one_dataset(
        params,
        file_name,
        train_q_data,
        train_qa_data,
        train_pid,
        valid_q_data,
        valid_qa_data,
        valid_pid,
    )
    test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid, best_epoch)
