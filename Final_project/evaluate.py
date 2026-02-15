# evaluate.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Any, List

from config import CFG
from env import GridWorldEnv
from agent_qlearning import QLearningAgent


@dataclass
class EvalConfig:
    qtable_path: str = "runs/run_01/qtable.pkl"
    episodes_per_scenario: int = 100
    out_dir: str = "runs/run_01"
    out_csv: str = "eval_summary.csv"
    verbose: bool = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    if not os.path.exists(cfg.qtable_path):
        raise FileNotFoundError(f"Q-table not found: {cfg.qtable_path}")

    env = GridWorldEnv(seed=CFG.seed)
    agent = QLearningAgent.load(cfg.qtable_path)

    _ensure_dir(cfg.out_dir)
    out_path = os.path.join(cfg.out_dir, cfg.out_csv)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario_id",
            "scenario_type",
            "scenario_name",
            "episodes",
            "success_rate",
            "avg_length",
            "avg_reward",
            "goal_count",
            "timeout_count",
        ])

    overall_rewards: List[float] = []
    overall_lengths: List[float] = []
    overall_success: List[float] = []

    rows = []

    for sc in env.library:
        sid = sc.scenario_id
        goal_count = 0
        timeout_count = 0
        total_reward = 0.0
        total_len = 0

        for ep in range(cfg.episodes_per_scenario):
            s = env.reset(scenario_id=sid, seed_offset=ep + 1)

            done = False
            ep_reward = 0.0
            ep_len = 0
            last_info: Dict[str, Any] = {}

            while not done:
                valid = env.valid_actions(s) if hasattr(env, "valid_actions") else None
                a = agent.greedy_action(s, valid_actions=valid)

                s, r, done, info = env.step(a)
                ep_reward += float(r)
                ep_len += 1
                last_info = info

            done_reason = last_info.get("done_reason", None)
            if done_reason in ("goal", "goal2"):
                goal_count += 1
            elif done_reason == "timeout":
                timeout_count += 1

            total_reward += ep_reward
            total_len += ep_len

            if cfg.verbose:
                print(f"[sid={sid}] ep={ep} done={done_reason} len={ep_len} R={ep_reward:.1f}")

        episodes = cfg.episodes_per_scenario
        success_rate = goal_count / episodes
        avg_len = total_len / episodes
        avg_reward = total_reward / episodes

        rows.append([
            sid, sc.scenario_type, sc.name,
            episodes,
            round(success_rate, 4),
            round(avg_len, 2),
            round(avg_reward, 2),
            goal_count,
            timeout_count,
        ])

        overall_rewards.append(avg_reward)
        overall_lengths.append(avg_len)
        overall_success.append(success_rate)

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)

    overall = {
        "scenarios": len(env.library),
        "episodes_per_scenario": cfg.episodes_per_scenario,
        "macro_success_rate": sum(overall_success) / len(overall_success),
        "macro_avg_length": sum(overall_lengths) / len(overall_lengths),
        "macro_avg_reward": sum(overall_rewards) / len(overall_rewards),
        "saved": out_path,
    }
    return overall


def main():
    cfg = EvalConfig()
    overall = evaluate(cfg)

    print("\n=== EVALUATION SUMMARY (macro-average across scenarios) ===")
    print(f"Scenarios: {overall['scenarios']}")
    print(f"Episodes per scenario: {overall['episodes_per_scenario']}")
    print(f"Macro success rate: {overall['macro_success_rate']:.2%}")
    print(f"Macro avg length: {overall['macro_avg_length']:.2f}")
    print(f"Macro avg reward: {overall['macro_avg_reward']:.2f}")
    print(f"Saved: {overall['saved']}")


if __name__ == "__main__":
    main()
