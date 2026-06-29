from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nhb_utils import NHB_MODELS, NHB_TABLES, append_manifest, append_registry, ensure_nhb_dirs


ANALYSIS_ID = "nhb_02_architecture_perturbation_gates"
SCRIPT_NAME = "scripts/nhb_revision/02_architecture_perturbation_gates.py"
MODEL_FAMILIES = ["vanilla_rnn", "gru", "lstm"]
SEEDS = [11, 23, 37]
TASKS = ["nback", "go_nogo", "context_xor"]
HIDDEN_SIZES = [1, 2, 3, 4, 6, 8]
SEQ_LEN = 36
INPUT_DIM = 16
TRAIN_EPOCHS = 10
TRAIN_BATCHES = 5
TRAIN_BATCH_SIZE = 128
EVAL_BATCHES = 8
EVAL_BATCH_SIZE = 128
LR = 3e-3
N_PERMUTATIONS = 300
RNG_SEED = 20260611


@dataclass(frozen=True)
class StateConfig:
    name: str
    tau: float = 1.0
    lapse: float = 0.0
    hidden_noise_sigma: float = 0.0
    update_gain: float = 1.0
    memory_decay: float = 1.0


STATE_CONFIGS = [
    StateConfig("state_tau_mild", tau=1.4),
    StateConfig("state_tau_strong", tau=2.4),
    StateConfig("state_lapse_mild", lapse=0.06),
    StateConfig("state_lapse_strong", lapse=0.16),
    StateConfig("state_noise_mild", hidden_noise_sigma=0.06),
    StateConfig("state_noise_strong", hidden_noise_sigma=0.16),
    StateConfig("state_gain_mild", update_gain=0.75),
    StateConfig("state_gain_strong", update_gain=0.45),
    StateConfig("state_decay_mild", memory_decay=0.82),
    StateConfig("state_decay_strong", memory_decay=0.55),
    StateConfig("state_combined_mild", tau=1.4, lapse=0.04, hidden_noise_sigma=0.04, update_gain=0.8, memory_decay=0.85),
    StateConfig("state_combined_strong", tau=2.2, lapse=0.12, hidden_noise_sigma=0.12, update_gain=0.55, memory_decay=0.6),
]
BASE_STATE = StateConfig("baseline")


def task_batch(task: str, batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    x = np.zeros((batch_size, seq_len, INPUT_DIM), dtype=np.float32)
    y = np.zeros((batch_size, seq_len), dtype=np.float32)
    mask = np.ones((batch_size, seq_len), dtype=np.float32)
    load = np.zeros(batch_size, dtype=np.int64)
    if task == "nback":
        tokens = rng.integers(0, 6, size=(batch_size, seq_len))
        loads = rng.choice(np.array([1, 2, 3]), size=batch_size)
        for b in range(batch_size):
            n_back = int(loads[b])
            load[b] = n_back
            for t in range(seq_len):
                x[b, t, tokens[b, t]] = 1.0
                x[b, t, 12] = n_back / 3.0
                if t < n_back:
                    mask[b, t] = 0
                else:
                    y[b, t] = float(tokens[b, t] == tokens[b, t - n_back])
    elif task == "go_nogo":
        delay = 2
        cue = rng.integers(0, 3, size=(batch_size, seq_len))
        is_go = rng.random(size=(batch_size, seq_len)) < 0.5
        fatigue = np.linspace(0, 1, seq_len, dtype=np.float32)
        for b in range(batch_size):
            load[b] = delay
            for t in range(seq_len):
                if t < delay:
                    stim = rng.integers(0, 3)
                    mask[b, t] = 0
                elif is_go[b, t]:
                    stim = cue[b, t - delay]
                    y[b, t] = 1.0
                else:
                    stim = rng.choice([c for c in range(3) if c != cue[b, t - delay]])
                    y[b, t] = 0.0
                x[b, t, stim] = 1.0
                x[b, t, 9 + cue[b, t]] = 1.0
                x[b, t, 13] = fatigue[t]
    elif task == "context_xor":
        bit_a = rng.integers(0, 2, size=(batch_size, seq_len))
        bit_b = rng.integers(0, 2, size=(batch_size, seq_len))
        context = rng.integers(0, 2, size=(batch_size, 1))
        delay = 3
        for b in range(batch_size):
            load[b] = delay
            for t in range(seq_len):
                x[b, t, 6 + bit_a[b, t]] = 1.0
                x[b, t, 8 + bit_b[b, t]] = 1.0
                x[b, t, 14] = context[b, 0]
                x[b, t, 15] = delay / 4.0
                if t < delay:
                    mask[b, t] = 0.0
                else:
                    xor = bit_a[b, t - delay] ^ bit_b[b, t]
                    y[b, t] = float(xor if context[b, 0] else 1 - xor)
    else:
        raise ValueError(task)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), load


def mixed_batch(batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], np.ndarray]:
    per = math.ceil(batch_size / len(TASKS))
    xs, ys, masks, tasks, loads = [], [], [], [], []
    for task in TASKS:
        x, y, mask, load = task_batch(task, per, seq_len, rng)
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        tasks.extend([task] * per)
        loads.extend(load.tolist())
    return torch.cat(xs)[:batch_size], torch.cat(ys)[:batch_size], torch.cat(masks)[:batch_size], tasks[:batch_size], np.asarray(loads[:batch_size])


class ArtificialRNN(nn.Module):
    def __init__(self, model_family: str, hidden_size: int):
        super().__init__()
        self.model_family = model_family
        self.hidden_size = hidden_size
        if model_family == "vanilla_rnn":
            self.rnn = nn.RNN(INPUT_DIM, hidden_size, batch_first=True, nonlinearity="tanh")
        elif model_family == "gru":
            self.rnn = nn.GRU(INPUT_DIM, hidden_size, batch_first=True)
        elif model_family == "lstm":
            self.rnn = nn.LSTM(INPUT_DIM, hidden_size, batch_first=True)
        else:
            raise ValueError(model_family)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, state: StateConfig = BASE_STATE, return_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.rnn(x)
        hidden = hidden * state.update_gain * state.memory_decay
        if state.hidden_noise_sigma > 0:
            hidden = hidden + torch.randn_like(hidden) * state.hidden_noise_sigma
        logits = self.out(hidden).squeeze(-1) / max(state.tau, 1e-4)
        return logits, hidden if return_hidden else torch.empty(0)

    def recurrence_matrix(self) -> np.ndarray:
        hh = self.rnn.weight_hh_l0.detach().cpu().numpy()
        h = self.hidden_size
        return hh[:h, :]


def masked_bce(prob: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    nll = -(y * torch.log(prob.clamp(eps, 1 - eps)) + (1 - y) * torch.log((1 - prob).clamp(eps, 1 - eps)))
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def apply_lapse(prob: torch.Tensor, lapse: float) -> torch.Tensor:
    return prob * (1 - lapse) + 0.5 * lapse if lapse > 0 else prob


def train_agent(model_family: str, hidden_size: int, seed: int) -> tuple[ArtificialRNN, float]:
    torch.manual_seed(seed + hidden_size + 37 * MODEL_FAMILIES.index(model_family))
    rng = np.random.default_rng(seed + hidden_size)
    model = ArtificialRNN(model_family, hidden_size)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for _ in range(TRAIN_EPOCHS):
        for _ in range(TRAIN_BATCHES):
            x, y, mask, _, _ = mixed_batch(TRAIN_BATCH_SIZE, SEQ_LEN, rng)
            logits, _ = model(x)
            loss = masked_bce(torch.sigmoid(logits), y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    return model, float(np.mean(losses[-TRAIN_BATCHES:]))


def slope(pairs: list[tuple[int | float, float]]) -> float:
    if len(pairs) < 3:
        return np.nan
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    x = (x - x.min()) / max(x.max() - x.min(), 1.0)
    return float(np.polyfit(x, y, 1)[0])


def dynamics_features(model: ArtificialRNN, hidden: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(centered.T)) if centered.shape[0] > 1 else np.zeros((hidden.shape[1], hidden.shape[1]))
    eig = np.linalg.eigvalsh(cov).clip(min=0)
    participation = float((eig.sum() ** 2) / np.sum(eig**2)) if np.sum(eig**2) > 0 else 0.0
    matrix = model.recurrence_matrix()
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > max(singular.max(initial=0.0) * 1e-3, 1e-8)))
    try:
        spectral = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    except np.linalg.LinAlgError:
        spectral = np.nan
    decoder = np.nan
    if hidden.shape[0] > 50 and len(np.unique(labels)) == 2:
        try:
            idx = np.random.default_rng(17).choice(hidden.shape[0], min(2000, hidden.shape[0]), replace=False)
            split = int(len(idx) * 0.7)
            clf = LogisticRegression(max_iter=200, solver="liblinear").fit(hidden[idx[:split]], labels[idx[:split]])
            decoder = float(clf.score(hidden[idx[split:]], labels[idx[split:]]))
        except Exception:
            decoder = np.nan
    return {
        "participation_ratio": participation,
        "trajectory_radius": float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
        "mean_hidden_abs": float(np.mean(np.abs(hidden))),
        "hidden_variability": float(np.mean(np.std(hidden, axis=0))),
        "recurrent_rank_dyn": rank,
        "spectral_radius": spectral,
        "local_memory_timescale": float(-1 / np.log(min(max(spectral, 1e-6), 0.999999))) if spectral > 0 else np.nan,
        "latent_decoder_accuracy": decoder,
    }


def evaluate(model: ArtificialRNN, model_family: str, hidden_size: int, seed: int, state: StateConfig, agent_id: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1000)
    model.eval()
    probs, ys, masks, hiddens, tasks, loads = [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y, mask, task_names, load = mixed_batch(EVAL_BATCH_SIZE, SEQ_LEN, rng)
            logits, hidden = model(x, state=state, return_hidden=True)
            prob = apply_lapse(torch.sigmoid(logits), state.lapse)
            probs.append(prob.cpu().numpy())
            ys.append(y.cpu().numpy())
            masks.append(mask.cpu().numpy())
            hiddens.append(hidden.cpu().numpy())
            tasks.extend(task_names)
            loads.extend(load.tolist())
    prob = np.concatenate(probs)
    y = np.concatenate(ys)
    mask = np.concatenate(masks).astype(bool)
    hidden = np.concatenate(hiddens)
    valid_prob, valid_y = prob[mask], y[mask]
    pred = (valid_prob >= 0.5).astype(float)
    eps = 1e-6
    nll = -(valid_y * np.log(np.clip(valid_prob, eps, 1 - eps)) + (1 - valid_y) * np.log(np.clip(1 - valid_prob, eps, 1 - eps)))
    entropy = -(valid_prob * np.log2(np.clip(valid_prob, eps, 1 - eps)) + (1 - valid_prob) * np.log2(np.clip(1 - valid_prob, eps, 1 - eps)))
    confident = np.abs(valid_prob - 0.5) >= 0.35
    errors = (pred != valid_y).astype(float)
    time_acc, time_ent, time_resp = [], [], []
    for t in range(prob.shape[1]):
        tm = mask[:, t]
        if tm.any():
            pp, yy = prob[:, t][tm], y[:, t][tm]
            time_acc.append((t, float(np.mean((pp >= 0.5) == yy))))
            ee = -(pp * np.log2(np.clip(pp, eps, 1 - eps)) + (1 - pp) * np.log2(np.clip(1 - pp, eps, 1 - eps)))
            time_ent.append((t, float(np.mean(ee))))
            time_resp.append((t, float(np.mean(pp >= 0.5))))
    adj = mask[:, 1:] & mask[:, :-1]
    error_seq = ((prob >= 0.5) != y).astype(float)
    if adj.any():
        prev_e = error_seq[:, :-1][adj]
        next_e = error_seq[:, 1:][adj]
        err_trans = float(np.mean((prev_e == 1) & (next_e == 1)))
        err_auto = float(np.corrcoef(prev_e, next_e)[0, 1]) if np.std(prev_e) > 0 and np.std(next_e) > 0 else np.nan
        step_norm = np.linalg.norm(np.diff(hidden, axis=1), axis=2)[adj]
    else:
        err_trans = err_auto = np.nan
        step_norm = np.asarray([np.nan])
    flat_task = np.repeat(np.asarray(tasks), SEQ_LEN)[mask.reshape(-1)]
    flat_load = np.repeat(np.asarray(loads), SEQ_LEN)[mask.reshape(-1)]
    flat_prob = prob.reshape(-1)[mask.reshape(-1)]
    flat_y = y.reshape(-1)[mask.reshape(-1)]
    by_task: dict[str, float] = {}
    for task in TASKS:
        tm = flat_task == task
        by_task[f"accuracy_{task}"] = float(np.mean((flat_prob[tm] >= 0.5) == flat_y[tm])) if tm.any() else np.nan
    load_pairs = []
    nback = flat_task == "nback"
    for load in sorted(set(flat_load[nback].tolist())):
        lm = nback & (flat_load == load)
        if lm.any():
            load_pairs.append((load, float(np.mean((flat_prob[lm] >= 0.5) == flat_y[lm]))))
    flat_hidden = hidden[mask]
    dyn = dynamics_features(model, flat_hidden, flat_y)
    row = {
        "agent_id": agent_id,
        "model_family": model_family,
        "hidden_size": hidden_size,
        "seed": seed,
        "state_name": state.name,
        "tau": state.tau,
        "lapse": state.lapse,
        "hidden_noise_sigma": state.hidden_noise_sigma,
        "update_gain": state.update_gain,
        "memory_decay": state.memory_decay,
        "state_severity": float(abs(np.log(state.tau)) + state.lapse + state.hidden_noise_sigma + (1 - state.update_gain) + (1 - state.memory_decay)),
        "capacity_level": float(np.log2(hidden_size)),
        "mean_accuracy": float(np.mean(pred == valid_y)),
        "negative_log_likelihood": float(np.mean(nll)),
        "brier_score": float(np.mean((valid_prob - valid_y) ** 2)),
        "response_rate": float(np.mean(pred)),
        "response_entropy": float(np.mean(entropy)),
        "lapse_proxy": float(np.mean(errors[confident])) if confident.any() else np.nan,
        "time_accuracy_slope": slope(time_acc),
        "early_late_accuracy_delta": float(np.mean((prob[:, -12:][mask[:, -12:]] >= 0.5) == y[:, -12:][mask[:, -12:]]) - np.mean((prob[:, :12][mask[:, :12]] >= 0.5) == y[:, :12][mask[:, :12]])),
        "entropy_time_slope": slope(time_ent),
        "response_time_slope": slope(time_resp),
        "probability_volatility": float(np.mean(np.abs(np.diff(prob, axis=1)[adj]))) if adj.any() else np.nan,
        "confidence_volatility": float(np.std(np.abs(valid_prob - 0.5))),
        "error_transition_rate": err_trans,
        "error_lag1_autocorrelation": err_auto,
        "sequential_dependence": np.nan,
        "nback_load_accuracy_slope": slope(load_pairs),
        "n_eval_events": int(len(valid_y)),
        "mean_hidden_step_norm": float(np.nanmean(step_norm)),
        "sd_hidden_step_norm": float(np.nanstd(step_norm)),
    }
    row.update(by_task)
    row.update(dyn)
    return row


BASE_FEATURES = [
    "brier_score", "response_rate", "response_entropy", "lapse_proxy",
    "time_accuracy_slope", "early_late_accuracy_delta", "entropy_time_slope",
    "response_time_slope", "probability_volatility", "confidence_volatility",
    "error_transition_rate", "error_lag1_autocorrelation", "nback_load_accuracy_slope",
    "accuracy_nback", "accuracy_go_nogo", "accuracy_context_xor", "participation_ratio",
    "trajectory_radius", "mean_hidden_abs", "hidden_variability", "mean_hidden_step_norm",
    "sd_hidden_step_norm", "recurrent_rank_dyn", "spectral_radius", "local_memory_timescale",
    "latent_decoder_accuracy",
]
CONFOUNDS = ["mean_accuracy", "negative_log_likelihood"]


def residualize(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    conf = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[CONFOUNDS]), columns=CONFOUNDS)
    for col in features:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        res = np.full_like(y, np.nan)
        if ok.sum() >= 3:
            lr = LinearRegression().fit(conf.loc[ok], y[ok])
            res[ok] = y[ok] - lr.predict(conf.loc[ok])
        out[col] = res
    return out


def X_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    x = (df[BASE_FEATURES] if feature_set == "raw_fingerprint" else residualize(df, BASE_FEATURES)[BASE_FEATURES]).copy()
    for col in x.columns:
        if x[col].isna().all():
            x[col] = 0.0
    return x


def clf_pipe() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=RNG_SEED))])


def reg_pipe() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])


def match_pairs(agents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, seed), g in agents.groupby(["model_family", "seed"]):
        states = g[g.perturbation_class == "state"].copy()
        caps = g[g.perturbation_class == "capacity"].copy()
        for _, s in states.iterrows():
            caps["delta"] = (caps["mean_accuracy"] - s["mean_accuracy"]).abs()
            c = caps.sort_values("delta").iloc[0]
            if c["delta"] <= 0.20:
                match_id = f"{arch}_s{seed}_m{len(rows)}"
                rows.append({**s.to_dict(), "match_id": match_id, "label": 1, "label_name": "state"})
                rows.append({**c.to_dict(), "match_id": match_id, "label": 0, "label_name": "capacity"})
    return pd.DataFrame(rows)


def grouped_score(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in splitter.split(X, y, groups):
        model = clf_pipe().fit(X.iloc[tr], y[tr])
        y_pred[te] = model.predict(X.iloc[te])
        y_prob[te] = model.predict_proba(X.iloc[te])[:, 1]
    return {"balanced_accuracy": float(balanced_accuracy_score(y, y_pred)), "auc": float(roc_auc_score(y, y_prob)), "pred": y_pred, "prob": y_prob}


def run_gates(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    rows, nulls = [], []
    scopes = [("overall", matched), *[(task, matched.copy()) for task in TASKS]]
    for arch in MODEL_FAMILIES:
        arch_df = matched[matched.model_family == arch]
        for scope_name, _ in scopes:
            task_df = arch_df if scope_name == "overall" else arch_df.dropna(subset=[f"accuracy_{scope_name}"])
            if len(task_df) < 20:
                continue
            for fs in ["raw_fingerprint", "residualized_fingerprint"]:
                X = X_frame(task_df, fs)
                y = task_df.label.to_numpy(int)
                groups = task_df.match_id.to_numpy()
                obs = grouped_score(X, y, groups)
                null = []
                for p in range(N_PERMUTATIONS):
                    yp = y.copy()
                    for group in np.unique(groups):
                        idx = np.where(groups == group)[0]
                        if len(idx) == 2 and rng.random() < 0.5:
                            yp[idx] = yp[idx[::-1]]
                    score = grouped_score(X, yp, groups)["balanced_accuracy"]
                    null.append(score)
                    nulls.append({"analysis_id": ANALYSIS_ID, "script_name": SCRIPT_NAME, "model_family": arch, "feature_set": fs, "task_family": scope_name, "target_axis": "state_vs_capacity", "permutation_index": p + 1, "metric": "balanced_accuracy", "metric_value": score})
                p95 = float(np.quantile(null, 0.95))
                perm_p = float((np.sum(np.asarray(null) >= obs["balanced_accuracy"]) + 1) / (len(null) + 1))
                rows.append({"analysis_id": ANALYSIS_ID, "script_name": SCRIPT_NAME, "model_family": arch, "feature_set": fs, "task_family": scope_name, "n_rows": len(task_df), "n_groups": task_df.match_id.nunique(), "balanced_accuracy": obs["balanced_accuracy"], "auc": obs["auc"], "permutation_p": perm_p, "null_95th_percentile": p95, "effect_over_null_ratio": obs["balanced_accuracy"] / max(p95, 1e-9), "control_status": "control_passed" if perm_p < 0.05 else "control_failed", "claim_strength": "strong" if obs["balanced_accuracy"] > 0.75 and perm_p < 0.05 else "negative", "interpretation": "True architecture-specific artificial perturbation family gate."})
    return pd.DataFrame(rows), pd.DataFrame(nulls)


def leave_one_arch(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fs in ["raw_fingerprint", "residualized_fingerprint"]:
        for heldout in MODEL_FAMILIES:
            train = matched[matched.model_family != heldout]
            test = matched[matched.model_family == heldout]
            if len(test) < 10:
                continue
            model = clf_pipe().fit(X_frame(train, fs), train.label.to_numpy(int))
            y = test.label.to_numpy(int)
            prob = model.predict_proba(X_frame(test, fs))[:, 1]
            pred = (prob >= 0.5).astype(int)
            ba = float(balanced_accuracy_score(y, pred))
            auc = float(roc_auc_score(y, prob))
            rows.append({"analysis_id": ANALYSIS_ID, "script_name": SCRIPT_NAME, "heldout_model_family": heldout, "feature_set": fs, "task_family": "overall", "n_train": len(train), "n_test": len(test), "balanced_accuracy": ba, "auc": auc, "decision_rule_pass": ba > 0.70, "claim_strength": "strong" if ba > 0.70 else "negative", "interpretation": "True held-out architecture evaluation."})
    return pd.DataFrame(rows)


def hybrid_recovery(agents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hybrids = agents[agents.perturbation_class == "hybrid"].copy()
    for arch, df in hybrids.groupby("model_family"):
        for fs in ["raw_fingerprint", "residualized_fingerprint"]:
            for analysis_type, target in [("hybrid_state_axis", "state_severity"), ("hybrid_capacity_axis", "capacity_level")]:
                data = df.dropna(subset=[target])
                X = X_frame(data, fs)
                y = data[target].to_numpy(float)
                groups = data.seed.to_numpy()
                pred = np.zeros(len(data))
                splitter = GroupKFold(n_splits=min(3, len(np.unique(groups))))
                for tr, te in splitter.split(X, y, groups):
                    model = reg_pipe().fit(X.iloc[tr], y[tr])
                    pred[te] = model.predict(X.iloc[te])
                rho, p = stats.spearmanr(pred, y)
                rows.append({"analysis_id": ANALYSIS_ID, "script_name": SCRIPT_NAME, "model_family": arch, "analysis_type": analysis_type, "feature_set": fs, "target_axis": target, "n_hybrid_agents": len(data), "spearman_rho": rho, "bootstrap_ci_low": "", "bootstrap_ci_high": "", "nominal_p_value": p, "pass_gate": bool(np.isfinite(rho) and abs(rho) > 0.40 and p < 0.05), "claim_strength": "strong" if np.isfinite(rho) and abs(rho) > 0.40 and p < 0.05 else "failed_gate", "interpretation": "True architecture-specific hybrid scalar recovery."})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_nhb_dirs()
    started = datetime.now(timezone.utc).isoformat()
    model_dir = NHB_MODELS / "true_architecture_artificial_agents"
    model_dir.mkdir(parents=True, exist_ok=True)
    registry_rows, fingerprint_rows = [], []
    trained: dict[tuple[str, int, int], ArtificialRNN] = {}
    for arch in MODEL_FAMILIES:
        for seed in SEEDS:
            for hidden in HIDDEN_SIZES:
                t0 = time.time()
                model, loss = train_agent(arch, hidden, seed)
                trained[(arch, seed, hidden)] = model
                ckpt = model_dir / f"{arch}_seed{seed}_h{hidden}.pt"
                torch.save({"model_state_dict": model.state_dict(), "model_family": arch, "hidden_size": hidden, "seed": seed}, ckpt)
                registry_rows.append({"agent_id": f"{arch}_s{seed}_h{hidden}_baseline", "model_family": arch, "hidden_size": hidden, "seed": seed, "perturbation_class": "baseline", "perturbation_name": "baseline", "checkpoint_path": ckpt.relative_to(ROOT).as_posix(), "training_final_loss": loss, "training_time_seconds": time.time() - t0})
                if hidden == 8:
                    for st in STATE_CONFIGS:
                        aid = f"{arch}_s{seed}_h8_{st.name}"
                        fingerprint_rows.append({**evaluate(model, arch, hidden, seed, st, aid), "perturbation_class": "state", "perturbation_name": st.name})
                        registry_rows.append({"agent_id": aid, "model_family": arch, "hidden_size": hidden, "seed": seed, "perturbation_class": "state", "perturbation_name": st.name, "checkpoint_path": ckpt.relative_to(ROOT).as_posix(), "training_final_loss": loss, "training_time_seconds": 0})
                aid = f"{arch}_s{seed}_h{hidden}_capacity"
                fingerprint_rows.append({**evaluate(model, arch, hidden, seed, BASE_STATE, aid), "perturbation_class": "capacity", "perturbation_name": f"hidden_size_{hidden}"})
                registry_rows.append({"agent_id": aid, "model_family": arch, "hidden_size": hidden, "seed": seed, "perturbation_class": "capacity", "perturbation_name": f"hidden_size_{hidden}", "checkpoint_path": ckpt.relative_to(ROOT).as_posix(), "training_final_loss": loss, "training_time_seconds": 0})
                if hidden in [2, 4, 8]:
                    for st in [STATE_CONFIGS[1], STATE_CONFIGS[3], STATE_CONFIGS[5], STATE_CONFIGS[-1]]:
                        aid = f"{arch}_s{seed}_h{hidden}_{st.name}_hybrid"
                        fingerprint_rows.append({**evaluate(model, arch, hidden, seed, st, aid), "perturbation_class": "hybrid", "perturbation_name": f"{st.name}_h{hidden}"})
                        registry_rows.append({"agent_id": aid, "model_family": arch, "hidden_size": hidden, "seed": seed, "perturbation_class": "hybrid", "perturbation_name": f"{st.name}_h{hidden}", "checkpoint_path": ckpt.relative_to(ROOT).as_posix(), "training_final_loss": loss, "training_time_seconds": 0})
                print(f"trained/evaluated {arch} seed={seed} hidden={hidden}")
    registry = pd.DataFrame(registry_rows)
    agents = pd.DataFrame(fingerprint_rows)
    matched = match_pairs(agents)
    gate, nulls = run_gates(matched)
    hybrid = hybrid_recovery(agents)
    loo = leave_one_arch(matched)
    outputs = {
        "architecture_artificial_agent_registry.csv": registry,
        "architecture_artificial_agent_fingerprints.csv": agents,
        "architecture_performance_matched_agent_pairs.csv": matched,
        "architecture_perturbation_gate_results.csv": gate,
        "architecture_hybrid_recovery_results.csv": hybrid,
        "leave_one_architecture_gate_results.csv": loo,
        "architecture_gate_null_distributions.csv": nulls,
    }
    paths = []
    for name, df in outputs.items():
        path = NHB_TABLES / name
        df.to_csv(path, index=False)
        paths.append(path)
    append_manifest(ANALYSIS_ID, SCRIPT_NAME, paths)
    append_registry(ANALYSIS_ID, SCRIPT_NAME, started, paths, notes="True vanilla RNN/GRU/LSTM artificial agents trained and evaluated for NHB architecture perturbation gates.")
    print("Wrote true architecture perturbation gate outputs")


if __name__ == "__main__":
    main()
