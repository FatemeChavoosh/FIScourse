# visualization.py
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
)

# -------------------------
# Helpers
# -------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(window=w, min_periods=max(1, w//4)).mean().to_numpy()

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def _bool_success(done_reason: pd.Series) -> np.ndarray:
    # matches your train/eval logic
    return done_reason.isin(["goal", "goal2"]).to_numpy(dtype=int)

def _pick_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Choose reward threshold that maximizes F1.
    Return (best_thr, best_f1).
    """
    # candidate thresholds = unique scores (subsample if too many)
    uniq = np.unique(scores)
    if len(uniq) > 2000:
        # sample quantiles to avoid O(N^2)
        qs = np.linspace(0.0, 1.0, 2000)
        uniq = np.quantile(scores, qs)

    best_thr = float(uniq[0])
    best_f1 = -1.0

    for thr in uniq:
        y_pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return best_thr, best_f1

def _plot_save(fig, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

# -------------------------
# Config
# -------------------------
@dataclass
class VizConfig:
    q_run_dir: str = "runs/run_01"
    sarsa_run_dir: str = "runs/sarsa_run_01"
    out_dir: str = "runs/compare_viz"
    rolling_window: int = 200

# -------------------------
# Core
# -------------------------
def load_metrics(run_dir: str) -> pd.DataFrame:
    df = _safe_read_csv(os.path.join(run_dir, "metrics.csv"))

    # expected columns:
    # episode, scenario_id, scenario_type, epsilon, episode_reward, episode_length, done_reason, success
    needed = {
        "episode", "scenario_id", "scenario_type", "epsilon",
        "episode_reward", "episode_length", "done_reason", "success"
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{run_dir}/metrics.csv missing columns: {sorted(missing)}")

    # normalize dtypes
    df["episode"] = df["episode"].astype(int)
    df["scenario_id"] = df["scenario_id"].astype(int)
    df["episode_reward"] = df["episode_reward"].astype(float)
    df["episode_length"] = df["episode_length"].astype(int)
    df["epsilon"] = df["epsilon"].astype(float)
    df["scenario_type"] = df["scenario_type"].astype(str)
    df["done_reason"] = df["done_reason"].astype(str)
    df["success"] = df["success"].astype(int)
    return df.sort_values("episode").reset_index(drop=True)

def load_eval(run_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(run_dir, "eval_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # expected: scenario_id, scenario_type, scenario_name, episodes, success_rate, avg_length, avg_reward, ...
    return df

def macro_table(df: pd.DataFrame) -> Dict[str, float]:
    # macro across episodes (not across scenarios)
    y = df["success"].to_numpy(dtype=int)
    avg_reward = float(df["episode_reward"].mean())
    avg_len = float(df["episode_length"].mean())
    succ = float(y.mean())
    timeout_rate = float((df["done_reason"] == "timeout").mean())
    return {
        "episodes": float(len(df)),
        "success_rate": succ,
        "avg_reward": avg_reward,
        "avg_length": avg_len,
        "timeout_rate": timeout_rate,
    }

def plot_learning_curves(df_q: pd.DataFrame, df_s: pd.DataFrame, cfg: VizConfig) -> None:
    out = cfg.out_dir
    w = cfg.rolling_window

    # Success curve
    fig = plt.figure()
    xq = df_q["episode"].to_numpy()
    xs = df_s["episode"].to_numpy()
    yq = _rolling_mean(df_q["success"].to_numpy(dtype=float), w)
    ys = _rolling_mean(df_s["success"].to_numpy(dtype=float), w)
    plt.plot(xq, yq, label="Q-learning (rolling)")
    plt.plot(xs, ys, label="SARSA (rolling)")
    plt.xlabel("Episode")
    plt.ylabel(f"Success Rate (rolling window={w})")
    plt.title("Learning Curve: Success Rate")
    plt.legend()
    _plot_save(fig, os.path.join(out, "learning_success.png"))

    # Reward curve
    fig = plt.figure()
    yq = _rolling_mean(df_q["episode_reward"].to_numpy(dtype=float), w)
    ys = _rolling_mean(df_s["episode_reward"].to_numpy(dtype=float), w)
    plt.plot(xq, yq, label="Q-learning (rolling)")
    plt.plot(xs, ys, label="SARSA (rolling)")
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Reward (rolling window={w})")
    plt.title("Learning Curve: Episode Reward")
    plt.legend()
    _plot_save(fig, os.path.join(out, "learning_reward.png"))

    # Length curve
    fig = plt.figure()
    yq = _rolling_mean(df_q["episode_length"].to_numpy(dtype=float), w)
    ys = _rolling_mean(df_s["episode_length"].to_numpy(dtype=float), w)
    plt.plot(xq, yq, label="Q-learning (rolling)")
    plt.plot(xs, ys, label="SARSA (rolling)")
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Length (rolling window={w})")
    plt.title("Learning Curve: Episode Length")
    plt.legend()
    _plot_save(fig, os.path.join(out, "learning_length.png"))

def plot_done_reason_bars(df_q: pd.DataFrame, df_s: pd.DataFrame, cfg: VizConfig) -> None:
    out = cfg.out_dir
    # done_reason distribution
    reasons = sorted(set(df_q["done_reason"].unique()).union(df_s["done_reason"].unique()))

    q_counts = [(df_q["done_reason"] == r).mean() for r in reasons]
    s_counts = [(df_s["done_reason"] == r).mean() for r in reasons]

    x = np.arange(len(reasons))
    width = 0.35

    fig = plt.figure()
    plt.bar(x - width/2, q_counts, width, label="Q-learning")
    plt.bar(x + width/2, s_counts, width, label="SARSA")
    plt.xticks(x, reasons, rotation=25)
    plt.ylabel("Fraction of episodes")
    plt.title("Outcome Distribution (done_reason)")
    plt.legend()
    _plot_save(fig, os.path.join(out, "done_reason_distribution.png"))

def plot_eval_per_scenario(eval_q: pd.DataFrame, eval_s: pd.DataFrame, cfg: VizConfig) -> None:
    out = cfg.out_dir

    # Align by scenario_id
    q = eval_q[["scenario_id", "scenario_type", "scenario_name", "success_rate"]].copy()
    s = eval_s[["scenario_id", "success_rate"]].copy()
    m = q.merge(s, on="scenario_id", how="inner", suffixes=("_q", "_s"))

    # sort by type then id
    type_order = {"maze": 0, "traps": 1, "key": 2}
    m["type_rank"] = m["scenario_type"].map(type_order).fillna(99)
    m = m.sort_values(["type_rank", "scenario_id"]).reset_index(drop=True)

    x = np.arange(len(m))
    width = 0.4

    fig = plt.figure(figsize=(max(10, len(m) * 0.5), 4.8))
    plt.bar(x - width/2, m["success_rate_q"], width, label="Q-learning")
    plt.bar(x + width/2, m["success_rate_s"], width, label="SARSA")
    plt.xticks(x, [f"{int(sid)}" for sid in m["scenario_id"]], rotation=0)
    plt.ylabel("Success Rate")
    plt.title("Per-Scenario Success Rate (Eval)")
    plt.legend()
    _plot_save(fig, os.path.join(out, "eval_success_per_scenario.png"))

    # Also per-type macro bars
    grp = m.groupby("scenario_type")[["success_rate_q", "success_rate_s"]].mean().reset_index()
    fig = plt.figure()
    x = np.arange(len(grp))
    plt.bar(x - width/2, grp["success_rate_q"], width, label="Q-learning")
    plt.bar(x + width/2, grp["success_rate_s"], width, label="SARSA")
    plt.xticks(x, grp["scenario_type"])
    plt.ylabel("Mean Success Rate")
    plt.title("Macro Success Rate by Scenario Type (Eval)")
    plt.legend()
    _plot_save(fig, os.path.join(out, "eval_success_by_type.png"))

def plot_pr_roc_and_confusion(df: pd.DataFrame, algo_name: str, cfg: VizConfig) -> Dict[str, float]:
    """
    Builds:
      - ROC curve + AUC
      - PR curve
      - Confusion matrix at best F1 threshold
      - Precision/Recall/F1 at best threshold
    Using:
      y_true = success (goal/goal2)
      score = episode_reward
    """
    out = cfg.out_dir
    y_true = df["success"].to_numpy(dtype=int)
    scores = df["episode_reward"].to_numpy(dtype=float)

    # ROC
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({algo_name})  [score=episode_reward]")
    plt.legend()
    _plot_save(fig, os.path.join(out, f"roc_{algo_name.lower()}.png"))

    # PR curve
    prec, rec, pr_thr = precision_recall_curve(y_true, scores)
    fig = plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({algo_name})  [score=episode_reward]")
    _plot_save(fig, os.path.join(out, f"pr_{algo_name.lower()}.png"))

    # Best threshold for F1
    best_thr, best_f1 = _pick_best_threshold(y_true, scores)
    y_pred = (scores >= best_thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Confusion matrix plot
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({algo_name})  thr={best_thr:.2f}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred Fail", "Pred Success"])
    plt.yticks(tick_marks, ["True Fail", "True Success"])

    # annotate
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    _plot_save(fig, os.path.join(out, f"confusion_{algo_name.lower()}.png"))

    return {
        "roc_auc": float(roc_auc),
        "best_thr_reward": float(best_thr),
        "precision_at_best_thr": float(p),
        "recall_at_best_thr": float(r),
        "f1_at_best_thr": float(f1),
        "tn": float(cm[0, 0]),
        "fp": float(cm[0, 1]),
        "fn": float(cm[1, 0]),
        "tp": float(cm[1, 1]),
    }

def run_all(cfg: VizConfig) -> None:
    _ensure_dir(cfg.out_dir)

    df_q = load_metrics(cfg.q_run_dir)
    df_s = load_metrics(cfg.sarsa_run_dir)

    # 1) Learning curves (Success/Reward/Length)
    plot_learning_curves(df_q, df_s, cfg)

    # 2) Outcome distribution (done_reason)
    plot_done_reason_bars(df_q, df_s, cfg)

    # 3) Eval per scenario (if exists)
    eval_q = load_eval(cfg.q_run_dir)
    eval_s = load_eval(cfg.sarsa_run_dir)
    if eval_q is not None and eval_s is not None:
        plot_eval_per_scenario(eval_q, eval_s, cfg)

    # 4) PR/ROC/Confusion for each algorithm (reward as score)
    stats_q = plot_pr_roc_and_confusion(df_q, "Q-learning", cfg)
    stats_s = plot_pr_roc_and_confusion(df_s, "SARSA", cfg)

    # 5) Summary CSV (for your report section 7)
    macro_q = macro_table(df_q)
    macro_s = macro_table(df_s)

    summary = []
    summary.append({"algo": "Q-learning", **macro_q, **stats_q})
    summary.append({"algo": "SARSA", **macro_s, **stats_s})

    out_csv = os.path.join(cfg.out_dir, "viz_summary.csv")
    pd.DataFrame(summary).to_csv(out_csv, index=False)

    print("Saved visualizations to:", cfg.out_dir)
    print("Saved summary to:", out_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_run_dir", default="runs/run_01", help="Run dir for Q-learning (contains metrics.csv)")
    ap.add_argument("--sarsa_run_dir", default="runs/sarsa_run_01", help="Run dir for SARSA (contains metrics.csv)")
    ap.add_argument("--out_dir", default="runs/compare_viz", help="Output directory for plots and summary CSV")
    ap.add_argument("--rolling_window", type=int, default=200, help="Rolling window for learning curves")
    args = ap.parse_args()

    cfg = VizConfig(
        q_run_dir=args.q_run_dir,
        sarsa_run_dir=args.sarsa_run_dir,
        out_dir=args.out_dir,
        rolling_window=int(args.rolling_window),
    )
    run_all(cfg)

if __name__ == "__main__":
    main()
