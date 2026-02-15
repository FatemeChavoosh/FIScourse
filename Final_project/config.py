# config.py
from __future__ import annotations
from dataclasses import dataclass

# ========= Cell legend =========
WALL = "#"
EMPTY = "."
GOAL = "G"        # terminal (goal 1)
GOAL2 = "H"       # terminal (goal 2) for corridor-maze only
TRAP = "T"
KEY = "K"
DOOR = "D"        # optional (not used here) - keep for backward compat

# ========= Actions =========
# 0: up, 1: down, 2: left, 3: right
ACTIONS = (0, 1, 2, 3)
ACTION_DELTAS = {
    0: (-1, 0),
    1: ( 1, 0),
    2: ( 0,-1),
    3: ( 0, 1),
}
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

@dataclass(frozen=True)
class _CFG:
    # Grid
    height: int = 15
    width: int = 15

    # Scenarios
    n_maze: int = 7
    n_trap: int = 7
    n_key: int = 8
    # total
    n_scenarios: int = n_maze + n_trap + n_key  # 22

    # Episode
    step_limit: int = 450

    # Start sampling constraints
    start_min_bfs_to_any_goal: int = 12   # ensure start not near exits
    max_start_sampling_tries: int = 4000

    # Randomness
    seed: int = 42

    # Rewards
    r_step: float = -0.3
    r_trap: float = -25.0
    r_key: float = +20.0
    r_goal: float = +150.0

    # Key constraint
    r_locked_goal: float = -3.0  # attempting to step into goal without key

CFG = _CFG()
