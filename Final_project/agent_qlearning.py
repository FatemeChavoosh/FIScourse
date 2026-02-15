# agent_qlearning.py
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from config import ACTIONS

State = Tuple[int, int, int, int]  # (scenario_id, r, c, has_key)


@dataclass
class QLearningAgent:
    alpha: float = 0.10
    gamma: float = 0.95
    seed: int = 123

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        # Q[state][action] = value
        self.q: Dict[State, Dict[int, float]] = {}

    def _ensure_state(self, s: State) -> None:
        if s not in self.q:
            self.q[s] = {a: 0.0 for a in ACTIONS}

    def choose_action(self, state: State, epsilon: float, valid_actions: Optional[List[int]] = None) -> int:
        """
        epsilon-greedy, but if valid_actions provided, sample only from valid actions.
        """
        self._ensure_state(state)
        acts = valid_actions if (valid_actions is not None and len(valid_actions) > 0) else list(ACTIONS)

        if self.rng.random() < float(epsilon):
            return self.rng.choice(acts)

        # greedy among valid actions
        best_a = acts[0]
        best_q = self.q[state][best_a]
        for a in acts[1:]:
            v = self.q[state][a]
            if v > best_q:
                best_q = v
                best_a = a
        return best_a

    def greedy_action(self, state: State, valid_actions: Optional[List[int]] = None) -> int:
        self._ensure_state(state)
        acts = valid_actions if (valid_actions is not None and len(valid_actions) > 0) else list(ACTIONS)

        best_a = acts[0]
        best_q = self.q[state][best_a]
        for a in acts[1:]:
            v = self.q[state][a]
            if v > best_q:
                best_q = v
                best_a = a
        return best_a

    def update(self, s: State, a: int, r: float, s2: State, done: bool, valid_actions_s2: Optional[List[int]] = None) -> None:
        self._ensure_state(s)
        self._ensure_state(s2)

        if done:
            target = r
        else:
            acts2 = valid_actions_s2 if (valid_actions_s2 is not None and len(valid_actions_s2) > 0) else list(ACTIONS)
            max_next = max(self.q[s2][aa] for aa in acts2)
            target = r + self.gamma * max_next

        self.q[s][a] = self.q[s][a] + self.alpha * (target - self.q[s][a])

    def save(self, path: str) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "seed": self.seed,
            "q": self.q,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = QLearningAgent(alpha=payload["alpha"], gamma=payload["gamma"], seed=payload.get("seed", 123))
        agent.q = payload["q"]
        return agent
