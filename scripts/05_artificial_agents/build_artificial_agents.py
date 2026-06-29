from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_capacity.audit.full_run import assert_full_run_allowed


TABLES = PROJECT_ROOT / "outputs" / "tables"
CHECKPOINTS = PROJECT_ROOT / "outputs" / "model_checkpoints" / "artificial_agents"
LOGS = PROJECT_ROOT / "outputs" / "logs"

TASKS = ["nback", "go_nogo", "context_xor"]
SEEDS = [11, 23, 37]
SEQ_LEN = 36
INPUT_DIM = 16
EVAL_BATCHES = 12
EVAL_BATCH_SIZE = 160
TRAIN_EPOCHS = 28
TRAIN_BATCHES_PER_EPOCH = 8
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 3e-3


@dataclass(frozen=True)
class CapacityConfig:
    recurrence_type: str
    hidden_size: int
    recurrent_rank: int
    bottleneck_width: int
    memory_window: int


@dataclass(frozen=True)
class StateConfig:
    tau: float = 1.0
    lapse: float = 0.0
    hidden_noise_sigma: float = 0.0
    update_gain: float = 1.0
    memory_decay: float = 1.0


def planned_capacity_grid() -> list[CapacityConfig]:
    base = CapacityConfig("tanh_rnn", 8, 8, 8, 4)
    configs = {base}

    for hidden in [1, 2, 3, 4, 6, 8, 12]:
        configs.add(CapacityConfig("tanh_rnn", hidden, min(hidden, 8), hidden, 4))
    for rank in [1, 2, 4, 6, 8]:
        configs.add(CapacityConfig("tanh_rnn", 8, rank, 8, 4))
    for bottleneck in [1, 2, 4, 6, 8]:
        configs.add(CapacityConfig("tanh_rnn", 8, 8, bottleneck, 4))
    for window in [1, 2, 3, 4, 6]:
        configs.add(CapacityConfig("tanh_rnn", 8, 8, 8, window))
    for recurrence_type in ["linear_rnn", "relu_rnn", "gru"]:
        configs.add(CapacityConfig(recurrence_type, 8, 8, 8, 4))

    return sorted(configs, key=lambda c: (c.recurrence_type, c.hidden_size, c.recurrent_rank, c.bottleneck_width, c.memory_window))


def planned_state_grid() -> list[StateConfig]:
    configs = {StateConfig()}
    for tau in [0.6, 1.4, 2.2, 3.0]:
        configs.add(StateConfig(tau=tau))
    for lapse in [0.03, 0.07, 0.12, 0.20]:
        configs.add(StateConfig(lapse=lapse))
    for sigma in [0.03, 0.07, 0.12, 0.20]:
        configs.add(StateConfig(hidden_noise_sigma=sigma))
    for gain in [0.85, 0.70, 0.55, 0.40]:
        configs.add(StateConfig(update_gain=gain))
    for decay in [0.92, 0.80, 0.65, 0.50]:
        configs.add(StateConfig(memory_decay=decay))
    configs.update(
        {
            StateConfig(tau=1.4, lapse=0.03, hidden_noise_sigma=0.03, update_gain=0.85, memory_decay=0.92),
            StateConfig(tau=2.2, lapse=0.07, hidden_noise_sigma=0.07, update_gain=0.70, memory_decay=0.80),
            StateConfig(tau=3.0, lapse=0.12, hidden_noise_sigma=0.12, update_gain=0.55, memory_decay=0.65),
        }
    )
    return sorted(configs, key=lambda s: (s.tau, s.lapse, s.hidden_noise_sigma, s.update_gain, s.memory_decay))


def hybrid_state_grid() -> list[StateConfig]:
    return [
        StateConfig(),
        StateConfig(tau=1.4, lapse=0.03, hidden_noise_sigma=0.03, update_gain=0.85, memory_decay=0.92),
        StateConfig(tau=2.2, lapse=0.07, hidden_noise_sigma=0.07, update_gain=0.70, memory_decay=0.80),
        StateConfig(tau=3.0, lapse=0.12, hidden_noise_sigma=0.12, update_gain=0.55, memory_decay=0.65),
        StateConfig(tau=2.2),
        StateConfig(lapse=0.12),
        StateConfig(hidden_noise_sigma=0.12),
        StateConfig(update_gain=0.55),
        StateConfig(memory_decay=0.65),
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def task_batch(task: str, batch_size: int, seq_len: int, memory_window: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    x = np.zeros((batch_size, seq_len, INPUT_DIM), dtype=np.float32)
    y = np.zeros((batch_size, seq_len), dtype=np.float32)
    mask = np.ones((batch_size, seq_len), dtype=np.float32)
    load = np.zeros(batch_size, dtype=np.int64)

    if task == "nback":
        vocab = 6
        tokens = rng.integers(0, vocab, size=(batch_size, seq_len))
        loads = rng.choice(np.array([1, 2, min(3, memory_window)]), size=batch_size)
        for b in range(batch_size):
            n_back = int(max(1, loads[b]))
            load[b] = n_back
            for t in range(seq_len):
                x[b, t, tokens[b, t]] = 1.0
                x[b, t, 12] = n_back / 3.0
                if t < n_back:
                    mask[b, t] = 0.0
                else:
                    y[b, t] = float(tokens[b, t] == tokens[b, t - n_back])
    elif task == "go_nogo":
        delay = max(1, min(memory_window, 3))
        cue = rng.integers(0, 3, size=(batch_size, seq_len))
        stim = np.zeros((batch_size, seq_len), dtype=np.int64)
        is_go_trial = rng.random(size=(batch_size, seq_len)) < 0.5
        fatigue = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
        for b in range(batch_size):
            load[b] = delay
            for t in range(seq_len):
                if t < delay:
                    stim[b, t] = rng.integers(0, 3)
                    mask[b, t] = 0.0
                    y[b, t] = 0.0
                elif is_go_trial[b, t]:
                    stim[b, t] = cue[b, t - delay]
                    y[b, t] = 1.0
                else:
                    alternatives = [candidate for candidate in range(3) if candidate != cue[b, t - delay]]
                    stim[b, t] = rng.choice(alternatives)
                    y[b, t] = 0.0
                x[b, t, stim[b, t]] = 1.0
                x[b, t, 9 + cue[b, t]] = 1.0
                x[b, t, 13] = fatigue[t]
    elif task == "context_xor":
        bit_a = rng.integers(0, 2, size=(batch_size, seq_len))
        bit_b = rng.integers(0, 2, size=(batch_size, seq_len))
        context = rng.integers(0, 2, size=(batch_size, 1))
        delay = max(1, min(memory_window, 4))
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
                    eq = 1 - xor
                    y[b, t] = float(xor if context[b, 0] else eq)
    else:
        raise ValueError(f"Unknown task {task}")

    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), load


def mixed_batch(batch_size: int, seq_len: int, memory_window: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], np.ndarray]:
    per_task = math.ceil(batch_size / len(TASKS))
    xs, ys, masks, task_names, loads = [], [], [], [], []
    for task in TASKS:
        x, y, mask, load = task_batch(task, per_task, seq_len, memory_window, rng)
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        task_names.extend([task] * per_task)
        loads.extend(load.tolist())
    x = torch.cat(xs, dim=0)[:batch_size]
    y = torch.cat(ys, dim=0)[:batch_size]
    mask = torch.cat(masks, dim=0)[:batch_size]
    return x, y, mask, task_names[:batch_size], np.asarray(loads[:batch_size])


class TinyRNNAgent(nn.Module):
    def __init__(self, config: CapacityConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        rank = max(1, min(config.recurrent_rank, hidden))
        bottleneck = max(1, min(config.bottleneck_width, hidden))
        self.input = nn.Linear(INPUT_DIM, hidden)
        self.u = nn.Parameter(torch.randn(hidden, rank) / math.sqrt(hidden))
        self.v = nn.Parameter(torch.randn(rank, hidden) / math.sqrt(rank))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.bottleneck = nn.Linear(hidden, bottleneck)
        self.output = nn.Linear(bottleneck, 1)
        self.gru = nn.GRU(INPUT_DIM, hidden, batch_first=True) if config.recurrence_type == "gru" else None

    def recurrent_weight(self) -> torch.Tensor:
        return self.u @ self.v

    def forward(self, x: torch.Tensor, state: StateConfig | None = None, return_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        state = state or StateConfig()
        if self.gru is not None:
            hidden_seq, _ = self.gru(x)
            if state.memory_decay != 1.0 or state.update_gain != 1.0 or state.hidden_noise_sigma > 0:
                hidden_seq = hidden_seq * state.update_gain * state.memory_decay
                if state.hidden_noise_sigma > 0:
                    hidden_seq = hidden_seq + torch.randn_like(hidden_seq) * state.hidden_noise_sigma
            logits = self.output(torch.tanh(self.bottleneck(hidden_seq))).squeeze(-1)
            logits = logits / max(state.tau, 1e-4)
            return logits, hidden_seq if return_hidden else torch.empty(0)

        batch, seq_len, hidden_size = x.shape[0], x.shape[1], self.config.hidden_size
        h = x.new_zeros(batch, hidden_size)
        weight = self.recurrent_weight()
        hidden_steps = []
        logits_steps = []
        for t in range(seq_len):
            recurrent = (h @ weight.T) * state.update_gain
            proposed = self.input(x[:, t, :]) + recurrent + self.bias
            if self.config.recurrence_type == "linear_rnn":
                candidate = torch.clamp(proposed, -5.0, 5.0)
            elif self.config.recurrence_type == "relu_rnn":
                candidate = torch.relu(proposed)
            else:
                candidate = torch.tanh(proposed)
            if state.hidden_noise_sigma > 0:
                candidate = candidate + torch.randn_like(candidate) * state.hidden_noise_sigma
            h = state.memory_decay * h + (1.0 / max(state.tau, 1e-4)) * (candidate - h)
            hidden_steps.append(h)
            logits_steps.append(self.output(torch.tanh(self.bottleneck(h))).squeeze(-1))
        hidden_seq = torch.stack(hidden_steps, dim=1)
        logits = torch.stack(logits_steps, dim=1) / max(state.tau, 1e-4)
        return logits, hidden_seq if return_hidden else torch.empty(0)

    def recurrence_matrix_for_dynamics(self) -> np.ndarray:
        if self.gru is not None:
            hh = self.gru.weight_hh_l0.detach().cpu().numpy()
            hidden = self.config.hidden_size
            return hh[:hidden, :]
        return self.recurrent_weight().detach().cpu().numpy()


def apply_lapse(prob: torch.Tensor, lapse: float) -> torch.Tensor:
    if lapse <= 0:
        return prob
    return prob * (1.0 - lapse) + 0.5 * lapse


def masked_bce_from_prob(prob: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    nll = -(y * torch.log(prob.clamp(eps, 1 - eps)) + (1 - y) * torch.log((1 - prob).clamp(eps, 1 - eps)))
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def train_capacity_agent(config: CapacityConfig, seed: int) -> tuple[TinyRNNAgent, dict[str, float]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyRNNAgent(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_history = []

    model.train()
    for _ in range(TRAIN_EPOCHS):
        for _ in range(TRAIN_BATCHES_PER_EPOCH):
            x, y, mask, _, _ = mixed_batch(TRAIN_BATCH_SIZE, SEQ_LEN, config.memory_window, rng)
            logits, _ = model(x)
            prob = torch.sigmoid(logits)
            loss = masked_bce_from_prob(prob, y, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_history.append(float(loss.detach()))

    return model, {"training_final_loss": float(np.mean(loss_history[-TRAIN_BATCHES_PER_EPOCH:]))}


def collect_evaluation(model: TinyRNNAgent, config: CapacityConfig, state: StateConfig, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1000)
    model.eval()
    all_prob, all_y, all_mask, all_hidden, all_task, all_load = [], [], [], [], [], []

    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y, mask, tasks, loads = mixed_batch(EVAL_BATCH_SIZE, SEQ_LEN, config.memory_window, rng)
            logits, hidden = model(x, state=state, return_hidden=True)
            prob = apply_lapse(torch.sigmoid(logits), state.lapse)
            all_prob.append(prob.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_mask.append(mask.detach().cpu().numpy())
            all_hidden.append(hidden.detach().cpu().numpy())
            all_task.extend(tasks)
            all_load.extend(loads.tolist())

    prob = np.concatenate(all_prob, axis=0)
    y = np.concatenate(all_y, axis=0)
    mask = np.concatenate(all_mask, axis=0).astype(bool)
    hidden = np.concatenate(all_hidden, axis=0)
    task_per_sequence = np.asarray(all_task)
    load_per_sequence = np.asarray(all_load)

    valid_prob = prob[mask]
    valid_y = y[mask]
    pred = (valid_prob >= 0.5).astype(float)
    eps = 1e-6
    nll = -(valid_y * np.log(np.clip(valid_prob, eps, 1 - eps)) + (1 - valid_y) * np.log(np.clip(1 - valid_prob, eps, 1 - eps)))
    entropy = -(valid_prob * np.log2(np.clip(valid_prob, eps, 1 - eps)) + (1 - valid_prob) * np.log2(np.clip(1 - valid_prob, eps, 1 - eps)))
    confident = np.abs(valid_prob - 0.5) >= 0.35
    lapse_proxy = float(np.mean((pred != valid_y)[confident])) if np.any(confident) else np.nan
    errors = (pred != valid_y).astype(float)
    time_index = np.arange(prob.shape[1], dtype=float)
    time_accuracy = []
    time_entropy = []
    time_response = []
    for t in range(prob.shape[1]):
        t_mask = mask[:, t].astype(bool)
        if np.any(t_mask):
            t_prob = prob[:, t][t_mask]
            t_y = y[:, t][t_mask]
            t_pred = (t_prob >= 0.5).astype(float)
            t_entropy = -(
                t_prob * np.log2(np.clip(t_prob, eps, 1 - eps))
                + (1 - t_prob) * np.log2(np.clip(1 - t_prob, eps, 1 - eps))
            )
            time_accuracy.append((t, float(np.mean(t_pred == t_y))))
            time_entropy.append((t, float(np.mean(t_entropy))))
            time_response.append((t, float(np.mean(t_pred))))

    def slope_from_pairs(pairs: list[tuple[int, float]]) -> float:
        if len(pairs) < 3:
            return np.nan
        xs = np.asarray([item[0] for item in pairs], dtype=float)
        ys = np.asarray([item[1] for item in pairs], dtype=float)
        xs = (xs - xs.min()) / max(xs.max() - xs.min(), 1.0)
        return float(np.polyfit(xs, ys, 1)[0])

    time_accuracy_slope = slope_from_pairs(time_accuracy)
    entropy_time_slope = slope_from_pairs(time_entropy)
    response_time_slope = slope_from_pairs(time_response)
    valid_counts = mask.sum(axis=0)
    early_steps = time_index <= np.quantile(time_index[valid_counts > 0], 0.33)
    late_steps = time_index >= np.quantile(time_index[valid_counts > 0], 0.67)
    early_mask = mask & early_steps[None, :]
    late_mask = mask & late_steps[None, :]
    early_acc = np.mean((prob[early_mask] >= 0.5) == y[early_mask]) if early_mask.any() else np.nan
    late_acc = np.mean((prob[late_mask] >= 0.5) == y[late_mask]) if late_mask.any() else np.nan
    early_late_accuracy_delta = float(late_acc - early_acc) if not (np.isnan(early_acc) or np.isnan(late_acc)) else np.nan

    adjacent_mask = mask[:, 1:] & mask[:, :-1]
    probability_volatility = float(np.mean(np.abs(np.diff(prob, axis=1)[adjacent_mask]))) if adjacent_mask.any() else np.nan
    confidence_volatility = float(np.std(np.abs(valid_prob - 0.5)))
    error_sequences = ((prob >= 0.5) != y).astype(float)
    if adjacent_mask.any():
        prev_errors = error_sequences[:, :-1][adjacent_mask]
        next_errors = error_sequences[:, 1:][adjacent_mask]
        error_transition_rate = float(np.mean((prev_errors == 1) & (next_errors == 1)))
        if np.std(prev_errors) > 0 and np.std(next_errors) > 0:
            error_lag1_autocorrelation = float(np.corrcoef(prev_errors, next_errors)[0, 1])
        else:
            error_lag1_autocorrelation = np.nan
    else:
        error_transition_rate = np.nan
        error_lag1_autocorrelation = np.nan

    flat_hidden = hidden[mask]
    flat_task = np.repeat(task_per_sequence, SEQ_LEN)[mask.reshape(-1)]
    flat_load = np.repeat(load_per_sequence, SEQ_LEN)[mask.reshape(-1)]
    flat_prob = prob.reshape(-1)[mask.reshape(-1)]
    flat_y = y.reshape(-1)[mask.reshape(-1)]
    hidden_adjacent = mask[:, 1:] & mask[:, :-1]
    if hidden_adjacent.any():
        hidden_step_norms = np.linalg.norm(np.diff(hidden, axis=1), axis=2)[hidden_adjacent]
        mean_hidden_step_norm = float(np.mean(hidden_step_norms))
        sd_hidden_step_norm = float(np.std(hidden_step_norms))
    else:
        mean_hidden_step_norm = np.nan
        sd_hidden_step_norm = np.nan

    by_task_accuracy = {}
    for task in TASKS:
        task_mask = flat_task == task
        if np.any(task_mask):
            by_task_accuracy[f"accuracy_{task}"] = float(np.mean((flat_prob[task_mask] >= 0.5) == flat_y[task_mask]))

    load_slope = np.nan
    nback_mask = flat_task == "nback"
    if np.any(nback_mask):
        load_acc = []
        for load in sorted(set(flat_load[nback_mask].tolist())):
            load_mask = nback_mask & (flat_load == load)
            if np.any(load_mask):
                load_acc.append((float(load), float(np.mean((flat_prob[load_mask] >= 0.5) == flat_y[load_mask]))))
        if len(load_acc) >= 2:
            xs = np.asarray([item[0] for item in load_acc])
            ys = np.asarray([item[1] for item in load_acc])
            load_slope = float(np.polyfit(xs, ys, 1)[0])

    previous_target = y[:, :-1][mask[:, 1:]]
    current_pred = (prob[:, 1:][mask[:, 1:]] >= 0.5).astype(float)
    if len(previous_target) > 2 and np.std(previous_target) > 0 and np.std(current_pred) > 0:
        sequential_dependence = float(np.corrcoef(previous_target, current_pred)[0, 1])
    else:
        sequential_dependence = np.nan

    behavior = {
        "mean_accuracy": float(np.mean(pred == valid_y)),
        "negative_log_likelihood": float(np.mean(nll)),
        "brier_score": float(np.mean((valid_prob - valid_y) ** 2)),
        "response_rate": float(np.mean(pred)),
        "response_entropy": float(np.mean(entropy)),
        "lapse_proxy": lapse_proxy,
        "time_accuracy_slope": time_accuracy_slope,
        "early_late_accuracy_delta": early_late_accuracy_delta,
        "entropy_time_slope": entropy_time_slope,
        "response_time_slope": response_time_slope,
        "probability_volatility": probability_volatility,
        "confidence_volatility": confidence_volatility,
        "error_transition_rate": error_transition_rate,
        "error_lag1_autocorrelation": error_lag1_autocorrelation,
        "sequential_dependence": sequential_dependence,
        "nback_load_accuracy_slope": load_slope,
        "n_eval_events": int(len(valid_y)),
    }
    behavior.update(by_task_accuracy)

    dynamics = compute_dynamics(model, flat_hidden, flat_y)
    dynamics["mean_hidden_step_norm"] = mean_hidden_step_norm
    dynamics["sd_hidden_step_norm"] = sd_hidden_step_norm
    return {"behavior": behavior, "dynamics": dynamics}


def compute_dynamics(model: TinyRNNAgent, hidden: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    if centered.shape[0] > 1:
        covariance = np.atleast_2d(np.cov(centered.T))
    else:
        covariance = np.zeros((centered.shape[1], centered.shape[1]))
    eigvals = np.linalg.eigvalsh(covariance).clip(min=0)
    participation = float((eigvals.sum() ** 2) / np.sum(eigvals**2)) if np.sum(eigvals**2) > 0 else 0.0
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    matrix = model.recurrence_matrix_for_dynamics()
    singular = np.linalg.svd(matrix, compute_uv=False)
    numerical_rank = int(np.sum(singular > max(singular.max(initial=0.0) * 1e-3, 1e-8)))
    try:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    except np.linalg.LinAlgError:
        spectral_radius = np.nan
    memory_timescale = float(-1.0 / np.log(min(max(spectral_radius, 1e-6), 0.999999))) if spectral_radius > 0 else np.nan

    decoder_accuracy = np.nan
    if hidden.shape[0] > 50 and len(np.unique(labels)) == 2:
        rng = np.random.default_rng(17)
        idx = rng.choice(hidden.shape[0], size=min(4000, hidden.shape[0]), replace=False)
        split = int(len(idx) * 0.7)
        train_idx, test_idx = idx[:split], idx[split:]
        try:
            clf = LogisticRegression(max_iter=200, solver="liblinear")
            clf.fit(hidden[train_idx], labels[train_idx])
            decoder_accuracy = float(clf.score(hidden[test_idx], labels[test_idx]))
        except Exception:
            decoder_accuracy = np.nan

    return {
        "participation_ratio": participation,
        "trajectory_radius": radius,
        "mean_hidden_abs": float(np.mean(np.abs(hidden))),
        "hidden_variability": float(np.mean(np.std(hidden, axis=0))),
        "recurrent_rank": numerical_rank,
        "spectral_radius": spectral_radius,
        "local_memory_timescale": memory_timescale,
        "latent_decoder_accuracy": decoder_accuracy,
    }


def agent_id(prefix: str, seed: int, config: CapacityConfig, state: StateConfig) -> str:
    parts = [
        prefix,
        f"s{seed}",
        config.recurrence_type,
        f"h{config.hidden_size}",
        f"r{config.recurrent_rank}",
        f"b{config.bottleneck_width}",
        f"w{config.memory_window}",
        f"tau{state.tau:g}",
        f"lap{state.lapse:g}",
        f"sig{state.hidden_noise_sigma:g}",
        f"gain{state.update_gain:g}",
        f"decay{state.memory_decay:g}",
    ]
    return "_".join(parts).replace(".", "p")


def infer_capacity_axis(config: CapacityConfig) -> str:
    baseline = CapacityConfig("tanh_rnn", 8, 8, 8, 4)
    if config.recurrence_type != baseline.recurrence_type:
        return "recurrence_type"
    if config.hidden_size != baseline.hidden_size:
        return "hidden_size"
    if config.recurrent_rank != baseline.recurrent_rank:
        return "recurrent_rank"
    if config.bottleneck_width != baseline.bottleneck_width:
        return "bottleneck_width"
    if config.memory_window != baseline.memory_window:
        return "memory_window"
    return "baseline"


def infer_state_axis(state: StateConfig) -> str:
    changed = []
    if state.tau != 1.0:
        changed.append("tau")
    if state.lapse != 0.0:
        changed.append("lapse")
    if state.hidden_noise_sigma != 0.0:
        changed.append("hidden_noise_sigma")
    if state.update_gain != 1.0:
        changed.append("update_gain")
    if state.memory_decay != 1.0:
        changed.append("memory_decay")
    if not changed:
        return "baseline"
    if len(changed) == 1:
        return changed[0]
    return "composite_state"


def save_checkpoint(path: Path, model: TinyRNNAgent, agent_record: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "capacity_config": {k: agent_record[k] for k in ["recurrence_type", "hidden_size", "recurrent_rank", "bottleneck_width", "memory_window"]},
            "state_config": {k: agent_record[k] for k in ["tau", "lapse", "hidden_noise_sigma", "update_gain", "memory_decay"]},
            "agent_record": agent_record,
        },
        path,
    )


def add_agent_rows(
    model: TinyRNNAgent,
    training_metrics: dict[str, float],
    config: CapacityConfig,
    state: StateConfig,
    seed: int,
    family: str,
    perturbation_family: str,
    base_agent_id: str | None,
    records: dict[str, list[dict[str, Any]]],
) -> str:
    aid = agent_id(family, seed, config, state)
    checkpoint_path = CHECKPOINTS / f"{aid}.pt"
    evaluation = collect_evaluation(model, config, state, seed)
    capacity_axis = infer_capacity_axis(config)
    state_severity = (
        abs(math.log(state.tau))
        + state.lapse * 5
        + state.hidden_noise_sigma * 4
        + abs(1 - state.update_gain)
        + abs(1 - state.memory_decay)
    )
    capacity_level = (
        math.log2(max(config.hidden_size, 1))
        + math.log2(max(config.recurrent_rank, 1))
        + math.log2(max(config.bottleneck_width, 1))
        + math.log2(max(config.memory_window, 1))
    )

    registry = {
        "agent_id": aid,
        "family": family,
        "perturbation_family": perturbation_family,
        "seed": seed,
        "task_set": ",".join(TASKS),
        "base_agent_id": base_agent_id or "",
        "checkpoint_path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "recurrence_type": config.recurrence_type,
        "hidden_size": config.hidden_size,
        "recurrent_rank": config.recurrent_rank,
        "bottleneck_width": config.bottleneck_width,
        "memory_window": config.memory_window,
        "tau": state.tau,
        "lapse": state.lapse,
        "hidden_noise_sigma": state.hidden_noise_sigma,
        "update_gain": state.update_gain,
        "memory_decay": state.memory_decay,
        "capacity_axis": capacity_axis,
        "state_axis": infer_state_axis(state),
        "capacity_level": capacity_level,
        "state_severity": state_severity,
        "training_final_loss": training_metrics.get("training_final_loss", np.nan),
    }
    records["registry"].append(registry)
    records["perturbations"].append(
        {
            "agent_id": aid,
            "family": family,
            "perturbation_family": perturbation_family,
            "capacity_axis": capacity_axis,
            "state_axis": infer_state_axis(state),
            "capacity_level": capacity_level,
            "state_severity": state_severity,
            **asdict(config),
            **asdict(state),
        }
    )
    records["behavior"].append({"agent_id": aid, **evaluation["behavior"]})
    records["dynamics"].append({"agent_id": aid, **evaluation["dynamics"]})
    save_checkpoint(checkpoint_path, model, registry)
    return aid


def main() -> int:
    assert_full_run_allowed()
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    capacity_configs = planned_capacity_grid()
    state_configs = planned_state_grid()
    hybrid_states = hybrid_state_grid()
    baseline_config = CapacityConfig("tanh_rnn", 8, 8, 8, 4)
    hybrid_capacity_configs = [
        CapacityConfig("tanh_rnn", hidden, min(hidden, 8), hidden, 4)
        for hidden in [1, 2, 3, 4, 6, 8]
    ]

    records: dict[str, list[dict[str, Any]]] = {"registry": [], "perturbations": [], "behavior": [], "dynamics": []}
    trained: dict[tuple[int, CapacityConfig], tuple[TinyRNNAgent, dict[str, float], str]] = {}
    log_lines = []

    for seed in SEEDS:
        for config in capacity_configs:
            model, metrics = train_capacity_agent(config, seed)
            family = "baseline" if config == baseline_config else "capacity_perturbation"
            aid = add_agent_rows(
                model=model,
                training_metrics=metrics,
                config=config,
                state=StateConfig(),
                seed=seed,
                family=family,
                perturbation_family=infer_capacity_axis(config),
                base_agent_id=None,
                records=records,
            )
            trained[(seed, config)] = (model, metrics, aid)
            log_lines.append(f"TRAINED {aid} loss={metrics['training_final_loss']:.4f}")

        base_model, base_metrics, base_id = trained[(seed, baseline_config)]
        for state in state_configs:
            if state == StateConfig():
                continue
            add_agent_rows(
                model=base_model,
                training_metrics=base_metrics,
                config=baseline_config,
                state=state,
                seed=seed,
                family="state_perturbation",
                perturbation_family="state_operating_regime",
                base_agent_id=base_id,
                records=records,
            )

        for config in hybrid_capacity_configs:
            model, metrics, cap_id = trained[(seed, config)]
            for state in hybrid_states:
                if state == StateConfig() and config == baseline_config:
                    continue
                add_agent_rows(
                    model=model,
                    training_metrics=metrics,
                    config=config,
                    state=state,
                    seed=seed,
                    family="hybrid",
                    perturbation_family="capacity_x_state",
                    base_agent_id=cap_id,
                    records=records,
                )

    registry = pd.DataFrame(records["registry"]).sort_values("agent_id")
    perturbations = pd.DataFrame(records["perturbations"]).sort_values("agent_id")
    behavior = pd.DataFrame(records["behavior"]).sort_values("agent_id")
    dynamics = pd.DataFrame(records["dynamics"]).sort_values("agent_id")

    registry.to_csv(TABLES / "artificial_agent_registry.csv", index=False)
    perturbations.to_csv(TABLES / "artificial_perturbation_parameters.csv", index=False)
    behavior.to_csv(TABLES / "artificial_behavioral_fingerprints.csv", index=False)
    dynamics.to_csv(TABLES / "artificial_dynamics_fingerprints.csv", index=False)

    summary = {
        "n_agents": int(len(registry)),
        "n_checkpoints": len(list(CHECKPOINTS.glob("*.pt"))),
        "families": registry["family"].value_counts().to_dict(),
        "seeds": SEEDS,
        "tasks": TASKS,
        "train_epochs": TRAIN_EPOCHS,
        "train_batches_per_epoch": TRAIN_BATCHES_PER_EPOCH,
        "eval_events_per_agent": int(behavior["n_eval_events"].median()) if len(behavior) else 0,
    }
    (LOGS / "artificial_agents_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (LOGS / "artificial_agents_training.log").write_text("\n".join(log_lines), encoding="utf-8")

    print(
        "ARTIFICIAL_AGENTS complete "
        f"agents={summary['n_agents']} checkpoints={summary['n_checkpoints']} "
        f"families={summary['families']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
