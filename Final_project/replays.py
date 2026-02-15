# replays.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class EpisodeReplay:
    scenario_id: int
    scenario_type: str
    scenario_name: str
    episode_idx: int
    epsilon: float
    done_reason: str
    success: bool
    total_reward: float
    steps: int

    # video trace
    positions: List[Tuple[int, int]]
    actions: List[int]

    # markers
    blocked_steps: List[int] = field(default_factory=list)
    trap_steps: List[int] = field(default_factory=list)
    picked_key_step: Optional[int] = None

    # IMPORTANT for GUI: store initial grid (before agent moves)
    # list of 15 strings each length 15
    grid0: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # tuples -> lists for json safety
        d["positions"] = [[int(r), int(c)] for (r, c) in self.positions]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EpisodeReplay":
        # backward compatible: grid0 may be missing
        positions = d.get("positions", [])
        pos2 = [(int(p[0]), int(p[1])) for p in positions]

        return EpisodeReplay(
            scenario_id=int(d["scenario_id"]),
            scenario_type=str(d.get("scenario_type", "")),
            scenario_name=str(d.get("scenario_name", "")),
            episode_idx=int(d.get("episode_idx", 0)),
            epsilon=float(d.get("epsilon", 0.0)),
            done_reason=str(d.get("done_reason", "unknown")),
            success=bool(d.get("success", False)),
            total_reward=float(d.get("total_reward", 0.0)),
            steps=int(d.get("steps", len(d.get("actions", [])))),

            positions=pos2,
            actions=[int(x) for x in d.get("actions", [])],

            blocked_steps=[int(x) for x in d.get("blocked_steps", [])],
            trap_steps=[int(x) for x in d.get("trap_steps", [])],
            picked_key_step=None if d.get("picked_key_step", None) is None else int(d["picked_key_step"]),
            grid0=d.get("grid0", None),
        )


class ReplayStore:
    def __init__(self, path: str):
        self.path = path
        # data[sid] = {"episodes":[EpisodeReplay,...]}
        self.data: Dict[int, Dict[str, Any]] = {}

    def add(self, rep: EpisodeReplay) -> None:
        sid = int(rep.scenario_id)
        if sid not in self.data:
            self.data[sid] = {"episodes": []}
        self.data[sid]["episodes"].append(rep)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload: Dict[str, Any] = {}
        for sid, pack in self.data.items():
            payload[str(sid)] = [r.to_dict() for r in pack["episodes"]]

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)

        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.data = {}
        for sid_str, eps_list in payload.items():
            sid = int(sid_str)
            reps = [EpisodeReplay.from_dict(x) for x in eps_list]
            self.data[sid] = {"episodes": reps}

    def scenario_ids(self) -> List[int]:
        return sorted(self.data.keys())

    def get_episodes(self, scenario_id: int) -> List[EpisodeReplay]:
        sid = int(scenario_id)
        if sid not in self.data:
            return []
        return self.data[sid]["episodes"]
