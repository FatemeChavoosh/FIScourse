# record_replays_sarsa.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from config import CFG
from env import GridWorldEnv
from agent_sarsa import SARSAAgent
from replays import ReplayStore, EpisodeReplay


@dataclass
class RecordConfig:
    qtable_path: str = "runs/sarsa_run_01/qtable.pkl"
    out_path: str = "runs/sarsa_run_01/replays_all.json"

    episodes_per_scenario: int = 120

    # show early mistakes -> later improvements
    eps_start: float = 1.0
    eps_end: float = 0.0


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def snapshot_grid(env: GridWorldEnv) -> List[str]:
    if env.episode is None:
        return []
    g = env.episode.grid
    return ["".join(row) for row in g]


def run_episode(env: GridWorldEnv, agent: SARSAAgent, scenario_id: int, epsilon: float, episode_idx: int) -> EpisodeReplay:
    s = env.reset(scenario_id=scenario_id, seed_offset=episode_idx + 1)
    sc = next(x for x in env.library if x.scenario_id == scenario_id)

    grid0 = snapshot_grid(env)

    positions = [env.pos]
    actions: List[int] = []

    blocked_steps: List[int] = []
    trap_steps: List[int] = []
    picked_key_step: Optional[int] = None

    done = False
    total_reward = 0.0
    last_info: Dict[str, Any] = {}
    t = 0

    while not done:
        valid = env.valid_actions(s) if hasattr(env, "valid_actions") else None
        a = agent.choose_action(s, epsilon=epsilon, valid_actions=valid)

        s2, r, done, info = env.step(a)

        actions.append(a)
        positions.append(env.pos)
        total_reward += float(r)

        if info.get("blocked", False):
            blocked_steps.append(t)
        if info.get("hit_trap", False):
            trap_steps.append(t)
        if info.get("picked_key_now", False) and picked_key_step is None:
            picked_key_step = t

        s = s2
        last_info = info
        t += 1

    done_reason = str(last_info.get("done_reason") or "unknown")
    success = done_reason in ("goal", "goal2")

    return EpisodeReplay(
        scenario_id=scenario_id,
        scenario_type=sc.scenario_type,
        scenario_name=sc.name,
        episode_idx=episode_idx,
        epsilon=float(round(epsilon, 4)),
        done_reason=done_reason,
        success=bool(success),
        total_reward=float(round(total_reward, 3)),
        steps=len(actions),
        positions=[(int(r), int(c)) for (r, c) in positions],
        actions=[int(x) for x in actions],
        blocked_steps=[int(x) for x in blocked_steps],
        trap_steps=[int(x) for x in trap_steps],
        picked_key_step=None if picked_key_step is None else int(picked_key_step),
        grid0=grid0,
    )


def main():
    cfg = RecordConfig()

    if not os.path.exists(cfg.qtable_path):
        raise FileNotFoundError(cfg.qtable_path)

    env = GridWorldEnv(seed=CFG.seed)
    agent = SARSAAgent.load(cfg.qtable_path)

    eps_list = linspace(cfg.eps_start, cfg.eps_end, cfg.episodes_per_scenario)
    store = ReplayStore(cfg.out_path)

    for sc in env.library:
        print(f"[record] sid={sc.scenario_id} type={sc.scenario_type} name={sc.name}")
        for i, eps in enumerate(eps_list):
            rep = run_episode(env, agent, sc.scenario_id, eps, episode_idx=i)
            store.add(rep)

    store.save()
    print("Saved:", cfg.out_path)


if __name__ == "__main__":
    main()
