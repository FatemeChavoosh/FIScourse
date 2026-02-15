# train.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from config import CFG
from env import GridWorldEnv
from agent_qlearning import QLearningAgent, State


@dataclass
class TrainConfig:
    episodes: int = 20000

    # epsilon-greedy schedule
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9995

    # curriculum (optional)
    use_curriculum: bool = True
    curriculum_phase1: int = 4000   # only maze
    curriculum_phase2: int = 9000   # maze + traps
    # after => all scenarios

    out_dir: str = "runs"
    run_name: str = "run_01"
    save_every: int = 2000


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_scenario_id(env: GridWorldEnv, episode_idx: int, cfg: TrainConfig) -> Optional[int]:
    if not cfg.use_curriculum:
        return None

    maze_ids = [s.scenario_id for s in env.library if s.scenario_type == "maze"]
    trap_ids = [s.scenario_id for s in env.library if s.scenario_type == "traps"]
    key_ids  = [s.scenario_id for s in env.library if s.scenario_type == "key"]

    if episode_idx < cfg.curriculum_phase1:
        return env.rng.choice(maze_ids)
    if episode_idx < cfg.curriculum_phase2:
        return env.rng.choice(maze_ids + trap_ids)

    return env.rng.choice(maze_ids + trap_ids + key_ids)


def train(env: GridWorldEnv, agent: QLearningAgent, cfg: TrainConfig) -> Dict[str, Any]:
    out_path = os.path.join(cfg.out_dir, cfg.run_name)
    _ensure_dir(out_path)

    metrics_path = os.path.join(out_path, "metrics.csv")
    qtable_path = os.path.join(out_path, "qtable.pkl")

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode",
            "scenario_id",
            "scenario_type",
            "epsilon",
            "episode_reward",
            "episode_length",
            "done_reason",
            "success",
        ])

    epsilon = cfg.eps_start
    rolling_window = 200
    recent_success: List[int] = []
    best_rolling = 0.0

    for ep in range(cfg.episodes):
        sid_choice = _pick_scenario_id(env, ep, cfg)
        s: State = env.reset(scenario_id=sid_choice)

        scenario_id = s[0]
        scenario = next(x for x in env.library if x.scenario_id == scenario_id)
        s_type = scenario.scenario_type

        done = False
        total_reward = 0.0
        steps = 0
        last_info: Dict[str, Any] = {}

        while not done:
            valid = env.valid_actions(s) if hasattr(env, "valid_actions") else None
            a = agent.choose_action(s, epsilon=epsilon, valid_actions=valid)

            s2, r, done, info = env.step(a)

            valid2 = env.valid_actions(s2) if hasattr(env, "valid_actions") else None
            agent.update(s, a, r, s2, done, valid_actions_s2=valid2)

            s = s2
            total_reward += float(r)
            steps += 1
            last_info = info

        done_reason = last_info.get("done_reason", None)
        success = 1 if (done_reason == "goal" or done_reason == "goal2") else 0

        recent_success.append(success)
        if len(recent_success) > rolling_window:
            recent_success.pop(0)

        rolling_rate = sum(recent_success) / len(recent_success)
        best_rolling = max(best_rolling, rolling_rate)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep,
                scenario_id,
                s_type,
                round(epsilon, 6),
                round(total_reward, 4),
                steps,
                done_reason,
                success,
            ])

        epsilon = max(cfg.eps_min, epsilon * cfg.eps_decay)

        if (ep + 1) % cfg.save_every == 0:
            agent.save(qtable_path)

        if (ep + 1) % 500 == 0:
            print(
                f"ep={ep+1:6d} eps={epsilon:.4f} "
                f"R={total_reward:8.1f} len={steps:4d} "
                f"done={done_reason} roll_succ({rolling_window})={rolling_rate:.2%} best={best_rolling:.2%} "
                f"type={s_type}"
            )

    agent.save(qtable_path)
    return {
        "out_dir": out_path,
        "metrics_csv": metrics_path,
        "qtable_path": qtable_path,
        "best_success_rolling": best_rolling,
    }


def main():
    env = GridWorldEnv(seed=CFG.seed)
    agent = QLearningAgent(alpha=0.10, gamma=0.95, seed=123)

    cfg = TrainConfig(
        episodes=20000,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.9995,
        use_curriculum=True,
        curriculum_phase1=4000,
        curriculum_phase2=9000,
        out_dir="runs",
        run_name="run_01",
        save_every=2000,
    )

    res = train(env, agent, cfg)
    print("Training finished. Outputs:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
