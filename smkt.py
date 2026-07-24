import math
from enum import IntEnum

import numpy as np
import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

import torch.nn.functional as F


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class SMKT(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        kq_same,
        dropout,
        model_type,
        final_fc_dim=512,
        n_heads=8,
        d_ff=2048,
        l2=1e-5,
        separate_qa=False,
        counterfactual_blend=True,
        cf_slip_pred_thresh=0.55,
        cf_same_skill_eps=0.08,
        cf_min_same_skill_tail=2,
        cf_train=True,
        cf_lambda=0.03,
        cf_anti_lambda=0.02,
        cf_trigger_ratio=0.1,
        cf_uncertain_margin=0.2,
        cf_consistency_margin=0.005,
        cf_anti_margin=0.005,
        cf_sde_samples=24,
        cf_sde_steps=6,
        cf_sde_dt=0.25,
        cf_sde_theta=1.1,
        cf_sde_base_sigma=0.08,
        cf_sde_uncertainty_scale=0.35,
        cf_sde_shift_weight=0.35,
        causal_enable=True,
        causal_window=3,
        causal_t_lambda=0.2,
        causal_e_lambda=0.1,
        causal_r_lambda=0.3,
        causal_g_lambda=0.1,
        causal_fusion_enable=True,
        causal_fusion_beta=0.2,
        causal_overlap_low=0.1,
        causal_overlap_high=0.9,
        causal_clip_propensity=0.02,
        causal_start_epoch=8,
        causal_warmup_epochs=8,
        causal_min_future_same_skill=2,
    ):
        super().__init__()
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        embed_l = d_model
        # 反事实（KT-CF-Lite）：
        # - 稀疏触发：只在高不确定样本上执行反事实；
        # - 同技能 tail：仅比较 q[k]==q[t], k>t；
        # - 一致性正则：训练时只加辅助损失，不直接改预测概率（避免校准恶化）。
        self.counterfactual_blend = counterfactual_blend
        self.cf_slip_pred_thresh = float(cf_slip_pred_thresh)
        self.cf_same_skill_eps = float(cf_same_skill_eps)
        self.cf_min_same_skill_tail = int(cf_min_same_skill_tail)
        self.cf_train = bool(cf_train)
        self.cf_lambda = float(cf_lambda)
        self.cf_anti_lambda = float(cf_anti_lambda)
        self.cf_trigger_ratio = float(cf_trigger_ratio)
        self.cf_uncertain_margin = float(cf_uncertain_margin)
        self.cf_consistency_margin = float(cf_consistency_margin)
        self.cf_anti_margin = float(cf_anti_margin)
        self.cf_sde_samples = int(cf_sde_samples)
        self.cf_sde_steps = int(cf_sde_steps)
        self.cf_sde_dt = float(cf_sde_dt)
        self.cf_sde_theta = float(cf_sde_theta)
        self.cf_sde_base_sigma = float(cf_sde_base_sigma)
        self.cf_sde_uncertainty_scale = float(cf_sde_uncertainty_scale)
        self.cf_sde_shift_weight = float(cf_sde_shift_weight)
        # 因果（T-learner + DML）：
        # T_t: 当前时刻是否答对；
        # Y_t: 未来窗口内同技能表现（均值）；
        # e(X): propensity，g(X): baseline outcome，tau(X)=mu1(X)-mu0(X)。
        self.causal_enable = bool(causal_enable)
        self.causal_window = int(causal_window)
        self.causal_t_lambda = float(causal_t_lambda)
        self.causal_e_lambda = float(causal_e_lambda)
        self.causal_r_lambda = float(causal_r_lambda)
        self.causal_g_lambda = float(causal_g_lambda)
        self.causal_fusion_enable = bool(causal_fusion_enable)
        self.causal_fusion_beta = float(causal_fusion_beta)
        self.causal_overlap_low = float(causal_overlap_low)
        self.causal_overlap_high = float(causal_overlap_high)
        self.causal_clip_propensity = float(causal_clip_propensity)
        self.causal_start_epoch = int(causal_start_epoch)
        self.causal_warmup_epochs = int(causal_warmup_epochs)
        self.causal_min_future_same_skill = int(causal_min_future_same_skill)
        self.current_epoch = 1
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        self.q_embed = nn.Embedding(self.n_question + 1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        self.model = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model // n_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )
        if self.causal_enable:
            causal_in_dim = d_model + embed_l
            self.mu0_head = nn.Sequential(
                nn.Linear(causal_in_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1),
            )
            self.mu1_head = nn.Sequential(
                nn.Linear(causal_in_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1),
            )
            self.e_head = nn.Sequential(
                nn.Linear(causal_in_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1),
            )
            self.g_head = nn.Sequential(
                nn.Linear(causal_in_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1),
            )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.dim() >= 1 and self.n_pid > 0 and p.size(0) == self.n_pid + 1:
                torch.nn.init.constant_(p, 0.0)

    def _embed_forward(self, q_data, qa_data, pid_data=None):
        """Encoder through Rasch + Transformer; returns q_embed_data, qa_embed_data, c_reg_loss."""
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data_proc = (qa_data - q_data) // self.n_question
            qa_embed_data = self.qa_embed(qa_data_proc) + q_embed_data

        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        else:
            c_reg_loss = 0.0
        return q_embed_data, qa_embed_data, c_reg_loss

    def _forward_probs_core(self, q_data, qa_data, pid_data=None):
        """Factual path probability per position [B,S]."""
        q_embed_data, qa_embed_data, c_reg_loss = self._embed_forward(q_data, qa_data, pid_data)
        d_output = self.model(q_embed_data, qa_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        mastery_logits = output.squeeze(-1)
        preds = torch.sigmoid(mastery_logits)
        return preds, mastery_logits, c_reg_loss, concat_q

    def _choose_counterfactual_anchor(self, preds_row, labels_row):
        valid = labels_row > -0.9
        wrong = (labels_row < 0.5) & valid
        correct = (labels_row > 0.5) & valid
        dev = preds_row.device
        s = preds_row.size(0)

        slip_score = torch.full((s,), -1e9, dtype=preds_row.dtype, device=dev)
        guess_score = torch.full((s,), -1e9, dtype=preds_row.dtype, device=dev)

        if wrong.any():
            slip_score = torch.where(wrong, preds_row, slip_score)
            slip_gate = wrong & (preds_row > self.cf_slip_pred_thresh)
            slip_score = slip_score.masked_fill(~slip_gate, -1e9)

        # fallback guess anchor without slip/guess heads:
        # if a correct response has very low predicted correctness, it is likely a guess-like event.
        if correct.any():
            guess_score = torch.where(correct, 1.0 - preds_row, guess_score)
            guess_gate = correct & (preds_row < (1.0 - self.cf_slip_pred_thresh))
            guess_score = guess_score.masked_fill(~guess_gate, -1e9)

        best_slip_val, best_slip_ix = torch.max(slip_score, dim=0)
        best_guess_val, best_guess_ix = torch.max(guess_score, dim=0)
        if best_slip_val <= -1e8 and best_guess_val <= -1e8:
            return None, None
        if best_slip_val >= best_guess_val:
            return int(best_slip_ix.item()), "slip"
        return int(best_guess_ix.item()), "guess"

    def _counterfactual_consistency_loss(
        self,
        q_data,
        qa_data,
        target,
        pid_data,
        preds,
    ):
        """KT-CF-Lite: sparse trigger + same-skill-tail consistency regularization."""
        dev = q_data.device
        B, S = q_data.shape
        row_idx = torch.arange(S, device=dev)
        valid_all = target > -0.9
        uncertainty = torch.clamp(self.cf_uncertain_margin - (preds - 0.5).abs(), min=0.0)
        uncertainty = uncertainty.masked_fill(~valid_all, 0.0)
        valid_cnt = valid_all.long().sum(dim=1).clamp(min=1)
        row_uncertainty = uncertainty.sum(dim=1) / valid_cnt

        trigger_ratio = max(0.0, min(1.0, self.cf_trigger_ratio))
        if trigger_ratio <= 0.0:
            return preds.new_tensor(0.0)
        k = max(1, int(math.ceil(B * trigger_ratio)))
        topk_rows = torch.topk(row_uncertainty, k=min(k, B), dim=0).indices
        total_cf_loss = preds.new_tensor(0.0)
        total_anti_loss = preds.new_tensor(0.0)
        active_rows = 0

        for b in topk_rows.tolist():
            labels = target[b]
            valid = labels > -0.9
            t_ix, flip_kind = self._choose_counterfactual_anchor(
                preds_row=preds[b],
                labels_row=labels,
            )

            if t_ix is None or q_data[b, t_ix].item() <= 0:
                continue

            qa_cf = qa_data[b : b + 1].clone()
            qv = q_data[b, t_ix].item()
            if flip_kind == "slip":
                qa_cf[0, t_ix] = qv + self.n_question
            else:
                qa_cf[0, t_ix] = qv

            need_grad_cf = self.training and self.cf_train
            if need_grad_cf:
                preds_cf_skill, _, _, _ = self._forward_probs_core(
                    q_data[b : b + 1], qa_cf, pid_data[b : b + 1] if pid_data is not None else None
                )
                preds_cf_pos, _, _, _ = self._forward_probs_core(
                    q_data[b : b + 1],
                    self._build_counterfactual_qa(q_data[b : b + 1], qa_data[b : b + 1], t_ix, force_correct=True),
                    pid_data[b : b + 1] if pid_data is not None else None,
                )
                preds_cf_neg, _, _, _ = self._forward_probs_core(
                    q_data[b : b + 1],
                    self._build_counterfactual_qa(q_data[b : b + 1], qa_data[b : b + 1], t_ix, force_correct=False),
                    pid_data[b : b + 1] if pid_data is not None else None,
                )
            else:
                with torch.no_grad():
                    preds_cf_skill, _, _, _ = self._forward_probs_core(
                        q_data[b : b + 1], qa_cf, pid_data[b : b + 1] if pid_data is not None else None
                    )
                    preds_cf_pos, _, _, _ = self._forward_probs_core(
                        q_data[b : b + 1],
                        self._build_counterfactual_qa(q_data[b : b + 1], qa_data[b : b + 1], t_ix, force_correct=True),
                        pid_data[b : b + 1] if pid_data is not None else None,
                    )
                    preds_cf_neg, _, _, _ = self._forward_probs_core(
                        q_data[b : b + 1],
                        self._build_counterfactual_qa(q_data[b : b + 1], qa_data[b : b + 1], t_ix, force_correct=False),
                        pid_data[b : b + 1] if pid_data is not None else None,
                    )
            preds_cf_skill = preds_cf_skill[0]
            preds_cf_pos = preds_cf_pos[0]
            preds_cf_neg = preds_cf_neg[0]

            same_skill = q_data[b] == q_data[b, t_ix]
            tail = row_idx > t_ix
            tail_same = tail & same_skill & valid
            tail_count = int(tail_same.long().sum().item())
            if tail_count < max(1, self.cf_min_same_skill_tail):
                continue

            delta_skill, shift_metric = self._sde_distribution_delta(
                factual_probs=preds[b],
                cf_probs=preds_cf_skill,
                tail_mask=tail_same,
            )
            if shift_metric < self.cf_same_skill_eps:
                continue

            delta_pos, _ = self._sde_distribution_delta(
                factual_probs=preds[b],
                cf_probs=preds_cf_pos,
                tail_mask=tail_same,
            )
            delta_neg, _ = self._sde_distribution_delta(
                factual_probs=preds[b],
                cf_probs=preds_cf_neg,
                tail_mask=tail_same,
            )

            # slip: wrong->correct, expect same-skill tail to increase under CF.
            # guess: correct->wrong, expect same-skill tail to decrease under CF.
            direction = 1.0 if flip_kind == "slip" else -1.0
            loss_row = F.softplus(self.cf_consistency_margin - direction * delta_skill).mean()
            anti_row = torch.relu((delta_pos + delta_neg).abs() - self.cf_anti_margin).mean()
            row_weight = 1.0 + float(row_uncertainty[b].item()) + shift_metric
            total_cf_loss = total_cf_loss + row_weight * loss_row
            total_anti_loss = total_anti_loss + row_weight * anti_row
            active_rows += 1

        if active_rows == 0:
            z = preds.new_tensor(0.0)
            return z, z
        return total_cf_loss / float(active_rows), total_anti_loss / float(active_rows)

    def _build_counterfactual_qa(self, q_row, qa_row, t_ix, force_correct):
        qa_cf = qa_row.clone()
        qv = int(q_row[0, t_ix].item())
        if force_correct:
            qa_cf[0, t_ix] = qv + self.n_question
        else:
            qa_cf[0, t_ix] = qv
        return qa_cf

    @staticmethod
    def _logit_clamped(x, eps=1e-6):
        x = torch.clamp(x, min=eps, max=1.0 - eps)
        return torch.log(x) - torch.log(1.0 - x)

    def _sde_distribution_delta(self, factual_probs, cf_probs, tail_mask):
        """
        FPE-by-SDE approximation:
        sample factual/cf particle trajectories and use mean-shift on tail as Δp.
        """
        idx = torch.nonzero(tail_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            z = factual_probs.new_zeros((0,))
            return z, 0.0

        dev = factual_probs.device
        dtype = factual_probs.dtype
        pf = factual_probs[idx]
        pc = cf_probs[idx]

        zf_mu = self._logit_clamped(pf)
        zc_mu = self._logit_clamped(pc)
        uncert = torch.clamp(4.0 * pf * (1.0 - pf), min=0.0, max=1.0)
        sigma = self.cf_sde_base_sigma + self.cf_sde_uncertainty_scale * uncert
        sigma = torch.clamp(sigma, min=1e-4)

        m = max(2, self.cf_sde_samples)
        n_steps = max(1, self.cf_sde_steps)
        dt = max(1e-3, self.cf_sde_dt)
        sqrt_dt = math.sqrt(dt)
        theta = max(1e-4, self.cf_sde_theta)

        zf = zf_mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(m, idx.numel(), device=dev, dtype=dtype)
        zc = zc_mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(m, idx.numel(), device=dev, dtype=dtype)
        for _ in range(n_steps):
            zf = zf + theta * (zf_mu.unsqueeze(0) - zf) * dt + sigma.unsqueeze(0) * sqrt_dt * torch.randn_like(zf)
            zc = zc + theta * (zc_mu.unsqueeze(0) - zc) * dt + sigma.unsqueeze(0) * sqrt_dt * torch.randn_like(zc)

        pf_dist = torch.sigmoid(zf)
        pc_dist = torch.sigmoid(zc)
        pf_mean = pf_dist.mean(dim=0)
        pc_mean = pc_dist.mean(dim=0)
        pf_std = pf_dist.std(dim=0, unbiased=False)
        pc_std = pc_dist.std(dim=0, unbiased=False)
        delta = pc_mean - pf_mean
        shift_metric = float((delta.abs() + self.cf_sde_shift_weight * (pf_std + pc_std)).mean().item())
        return delta, shift_metric

    def _build_causal_targets(self, q_data, target):
        """Construct treatment T and outcome Y for KT causal objective.

        T_t: current correctness at t.
        Y_t: mean correctness of future same-skill interactions in [t+1, t+window].
        """
        dev = q_data.device
        B, S = q_data.shape
        treatment = torch.zeros_like(target, dtype=torch.float32, device=dev)
        outcome = torch.full_like(target, fill_value=-1.0, dtype=torch.float32, device=dev)
        valid = target > -0.9
        treatment = torch.where(valid, target.float(), treatment)
        max_w = max(1, int(self.causal_window))
        for b in range(B):
            for t in range(S):
                if not bool(valid[b, t]):
                    continue
                qv = int(q_data[b, t].item())
                if qv <= 0:
                    continue
                right = min(S, t + 1 + max_w)
                if right <= t + 1:
                    continue
                q_tail = q_data[b, t + 1 : right]
                v_tail = valid[b, t + 1 : right]
                same = (q_tail == qv) & v_tail
                same_count = int(same.long().sum().item())
                if same_count >= max(1, self.causal_min_future_same_skill):
                    vals = target[b, t + 1 : right][same].float()
                    outcome[b, t] = vals.mean()
        outcome_valid = outcome > -0.5
        return treatment, outcome, outcome_valid

    def _causal_heads_from_rep(self, rep_flat):
        mu0 = torch.sigmoid(self.mu0_head(rep_flat).squeeze(-1))
        mu1 = torch.sigmoid(self.mu1_head(rep_flat).squeeze(-1))
        e = torch.sigmoid(self.e_head(rep_flat).squeeze(-1))
        g = torch.sigmoid(self.g_head(rep_flat).squeeze(-1))
        return mu0, mu1, e, g

    def _overlap_gate(self, e_prob):
        low = min(self.causal_overlap_low, self.causal_overlap_high)
        high = max(self.causal_overlap_low, self.causal_overlap_high)
        return ((e_prob >= low) & (e_prob <= high)).float()

    def set_epoch(self, epoch_idx):
        self.current_epoch = int(epoch_idx)

    def _causal_weight_scale(self):
        start = max(1, int(self.causal_start_epoch))
        warm = max(0, int(self.causal_warmup_epochs))
        cur = max(1, int(self.current_epoch))
        if cur < start:
            return 0.0
        if warm == 0:
            return 1.0
        progress = (cur - start + 1) / float(warm)
        return float(min(1.0, max(0.0, progress)))

    def forward(self, q_data, qa_data, target, pid_data=None):
        preds, mastery_logits, c_reg_loss, rep = self._forward_probs_core(q_data, qa_data, pid_data)

        labels = target.reshape(-1)
        mask = labels > -0.9
        masked_labels = labels[mask].float()

        preds_flat = preds.reshape(-1)
        mastery_logits_flat = mastery_logits.reshape(-1)
        kt_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        kt_loss = kt_loss_fn(mastery_logits_flat[mask], masked_labels)

        cf_reg = preds.new_tensor(0.0)
        anti_reg = preds.new_tensor(0.0)
        if self.counterfactual_blend and self.training and self.cf_train:
            cf_reg, anti_reg = self._counterfactual_consistency_loss(
                q_data, qa_data, target, pid_data, preds
            )

        pred_return = preds_flat
        causal_loss = preds.new_tensor(0.0)
        if self.causal_enable:
            causal_scale = self._causal_weight_scale()
            rep_flat = rep.reshape(-1, rep.size(-1))
            mu0_all, mu1_all, e_all, g_all = self._causal_heads_from_rep(rep_flat)
            tau_all = mu1_all - mu0_all
            # Safer default: keep causal effect as training regularizer.
            # Only apply logit fusion at eval when explicitly enabled.
            if self.causal_fusion_enable and (not self.training):
                gate_all = self._overlap_gate(e_all)
                fused_logits = mastery_logits_flat + self.causal_fusion_beta * gate_all * tau_all
                pred_return = torch.sigmoid(fused_logits)

            treat_seq, outcome_seq, outcome_valid_seq = self._build_causal_targets(q_data, target)
            treat_flat = treat_seq.reshape(-1)
            outcome_flat = outcome_seq.reshape(-1)
            outcome_valid_flat = outcome_valid_seq.reshape(-1)
            causal_mask = mask & outcome_valid_flat

            if causal_scale > 0.0 and torch.any(causal_mask):
                t_obs = treat_flat[causal_mask].float()
                y_obs = outcome_flat[causal_mask].float()
                mu0 = mu0_all[causal_mask]
                mu1 = mu1_all[causal_mask]
                e = e_all[causal_mask]
                g = g_all[causal_mask]

                mu_t = t_obs * mu1 + (1.0 - t_obs) * mu0
                l_t = F.binary_cross_entropy(
                    torch.clamp(mu_t, min=1e-6, max=1.0 - 1e-6),
                    y_obs,
                    reduction="mean",
                )
                l_e = F.binary_cross_entropy(
                    torch.clamp(e, min=1e-6, max=1.0 - 1e-6),
                    t_obs,
                    reduction="mean",
                )
                l_g = F.binary_cross_entropy(
                    torch.clamp(g, min=1e-6, max=1.0 - 1e-6),
                    y_obs,
                    reduction="mean",
                )

                e_clip = torch.clamp(
                    e,
                    min=self.causal_clip_propensity,
                    max=1.0 - self.causal_clip_propensity,
                )
                # Orthogonal residual objective (DML spirit):
                #   y - g(X) ≈ (t - e(X)) * tau(X)
                # nuisance terms are detached to stabilize tau learning.
                tau = mu1 - mu0
                y_res = y_obs - g.detach()
                t_res = t_obs - e_clip.detach()
                l_r = F.mse_loss(t_res * tau, y_res, reduction="mean")

                causal_loss = causal_scale * (
                    self.causal_t_lambda * l_t
                    + self.causal_e_lambda * l_e
                    + self.causal_g_lambda * l_g
                    + self.causal_r_lambda * l_r
                )

        total_loss = (
            kt_loss.sum()
            + c_reg_loss
            + self.cf_lambda * cf_reg
            + self.cf_anti_lambda * anti_reg
            + causal_loss
        )
        return total_loss, pred_return, mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        kq_same_bool = kq_same == 1
        if model_type in {"SMKT"}:
            self.blocks_1 = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same_bool,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.blocks_2 = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same_bool,
                    )
                    for _ in range(n_blocks * 2)
                ]
            )

    def forward(self, q_embed_data, qa_embed_data):
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data
        y = qa_pos_embed
        x = q_pos_embed
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        if mask == 0:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        else:
            self.q_linear = None
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.q_linear is not None:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.q_linear is not None:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.q_linear is not None:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, gammas)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    dev = q.device
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    x1 = torch.arange(seqlen, device=dev).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()
    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(dev)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2).float().view(1, 1, seqlen, seqlen).to(dev)
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.0)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)
    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen, device=dev, dtype=scores.dtype)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    return torch.matmul(scores, v)


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]
