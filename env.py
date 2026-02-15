# env.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

from config import CFG, WALL, EMPTY, GOAL, TRAP, KEY, ACTIONS, ACTION_DELTAS
from scenarios import get_scenario_library, Scenario

# optional symbols
try:
    from config import GOAL2  # e.g. "H"
except Exception:
    GOAL2 = None  # type: ignore

try:
    from config import DOOR  # optional (not required to exist in maps)
except Exception:
    DOOR = None  # type: ignore

import scenario_builder as sb

State = Tuple[int, int, int, int]  # (scenario_id, r, c, has_key)


@dataclass
class EpisodeRuntime:
    scenario: Scenario
    grid: List[List[str]]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    goal2: Optional[Tuple[int, int]] = None
    key_pos: Optional[Tuple[int, int]] = None


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_cell(grid: List[List[str]], ch: str) -> Optional[Tuple[int, int]]:
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == ch:
                return (r, c)
    return None


def _find_all_cells(grid: List[List[str]], ch: str) -> List[Tuple[int, int]]:
    out = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == ch:
                out.append((r, c))
    return out


def _choose_builder_fn():
    candidates = ["build_episode", "build", "make_episode", "sample_episode", "create_episode"]
    for name in candidates:
        fn = getattr(sb, name, None)
        if callable(fn):
            return fn, name
    raise ImportError(
        "scenario_builder module does not expose any known builder function. "
        "Expected one of: build_episode/build/make_episode/sample_episode/create_episode"
    )


def _call_builder(builder_fn, scenario: Scenario, seed_offset: int):
    try:
        return builder_fn(scenario, seed_offset=seed_offset)
    except TypeError:
        try:
            return builder_fn(scenario, seed_offset)
        except TypeError:
            rng = random.Random(seed_offset)
            return builder_fn(scenario, rng)


class GridWorldEnv:
    """
    Expectations:
    - train/evaluate uses: reset(), step(), env.rng.choice(...)
    - record_replays expects info keys: blocked/hit_trap/picked_key_now + done_reason
    - gui expects env.episode.grid exists and is 15x15
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = CFG.seed if seed is None else int(seed)
        self.rng = random.Random(self.seed)

        self.library = get_scenario_library()

        self.episode: Optional[EpisodeRuntime] = None
        self.pos: Tuple[int, int] = (0, 0)
        self.has_key: int = 0
        self.steps: int = 0
        self.done: bool = False
        self.done_reason: Optional[str] = None

        self._reset_calls: int = 0
        self._builder_fn, self._builder_name = _choose_builder_fn()

    def reset(self, scenario_id: Optional[int] = None, seed_offset: Optional[int] = None) -> State:
        if scenario_id is None:
            scenario_id = 0
        scenario = self.library[int(scenario_id)]

        if seed_offset is None:
            self._reset_calls += 1
            seed_offset = self._reset_calls
        seed_offset = int(seed_offset)

        ep_obj = _call_builder(self._builder_fn, scenario, seed_offset=seed_offset)

        grid_raw = getattr(ep_obj, "grid", None)
        if grid_raw is None:
            grid_raw = scenario.layout

        if isinstance(grid_raw, tuple):
            grid = [list(row) for row in grid_raw]
        else:
            grid = [list(row) for row in grid_raw]

        start = getattr(ep_obj, "start", None)
        if start is None:
            empties = _find_all_cells(grid, EMPTY)
            start = self.rng.choice(empties) if empties else (1, 1)

        goal = _find_cell(grid, GOAL)
        if goal is None:
            raise ValueError("No GOAL cell found in episode grid.")

        goal2 = None
        if GOAL2 is not None:
            goal2 = _find_cell(grid, GOAL2)

        key_pos = _find_cell(grid, KEY)

        self.episode = EpisodeRuntime(
            scenario=scenario,
            grid=grid,
            start=tuple(start),
            goal=tuple(goal),
            goal2=None if goal2 is None else tuple(goal2),
            key_pos=None if key_pos is None else tuple(key_pos),
        )

        self.pos = self.episode.start
        self.has_key = 0
        self.steps = 0
        self.done = False
        self.done_reason = None

        # If start is on key
        r0, c0 = self.pos
        if self.episode.grid[r0][c0] == KEY:
            self.has_key = 1
            self.episode.grid[r0][c0] = EMPTY
            self.episode.key_pos = None

        sid = self.episode.scenario.scenario_id
        return (sid, self.pos[0], self.pos[1], self.has_key)

    def step(self, action: int):
        if self.episode is None:
            raise RuntimeError("Call reset() before step().")
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be one of {ACTIONS}.")

        # already terminal
        if self.done:
            sid = self.episode.scenario.scenario_id
            r, c = self.pos
            return (sid, r, c, self.has_key), 0.0, True, {
                "steps": self.steps,
                "done_reason": self.done_reason,
                "blocked": False,
                "hit_trap": False,
                "picked_key_now": False,
            }

        self.steps += 1

        sid = self.episode.scenario.scenario_id
        r, c = self.pos
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc

        reward = float(getattr(CFG, "r_step", -0.3))

        blocked_now = False
        hit_trap_now = False
        picked_key_now = False

        # optional shaping (kept compatible)
        prev_dist = None
        if getattr(CFG, "use_shaping", False):
            if (self.episode.scenario.scenario_type == "key") and (not self.has_key) and (self.episode.key_pos is not None):
                target = self.episode.key_pos
            else:
                target = self.episode.goal
            prev_dist = _manhattan(self.pos, target)

        # bounds
        if not (0 <= nr < CFG.height and 0 <= nc < CFG.width):
            blocked_now = True
            reward += float(getattr(CFG, "r_blocked", -2.0))
            nr, nc = r, c
        else:
            cell = self.episode.grid[nr][nc]

            # wall
            if cell == WALL:
                blocked_now = True
                reward += float(getattr(CFG, "r_blocked", -2.0))
                nr, nc = r, c

            # door tile exists in some versions (optional)
            elif (DOOR is not None) and (cell == DOOR) and (not self.has_key):
                blocked_now = True
                reward += float(getattr(CFG, "r_blocked", -2.0))
                nr, nc = r, c

            # ✅ KEY SCENARIO RULE: goal WITHOUT key must NOT finish the episode
            elif (self.episode.scenario.scenario_type == "key") and (cell == GOAL) and (not self.has_key):
                blocked_now = True
                reward += float(getattr(CFG, "r_locked_goal", -3.0))
                nr, nc = r, c

            else:
                # move
                self.pos = (nr, nc)

                if cell == TRAP:
                    hit_trap_now = True
                    reward += float(getattr(CFG, "r_trap", -25.0))

                elif cell == KEY and (not self.has_key):
                    picked_key_now = True
                    self.has_key = 1
                    reward += float(getattr(CFG, "r_key", 20.0))
                    self.episode.grid[nr][nc] = EMPTY
                    self.episode.key_pos = None

                elif cell == GOAL:
                    reward += float(getattr(CFG, "r_goal", 150.0))
                    self.done = True
                    self.done_reason = "goal"

                elif (GOAL2 is not None) and (cell == GOAL2):
                    reward += float(getattr(CFG, "r_goal", 150.0))
                    self.done = True
                    self.done_reason = "goal2"

        # shaping after move
        if (prev_dist is not None) and (not self.done) and getattr(CFG, "use_shaping", False):
            if (self.episode.scenario.scenario_type == "key") and (not self.has_key) and (self.episode.key_pos is not None):
                target2 = self.episode.key_pos
            else:
                target2 = self.episode.goal
            new_dist = _manhattan(self.pos, target2)
            reward += float(getattr(CFG, "shaping_scale", 1.0)) * (prev_dist - new_dist)

        # timeout
        if (self.steps >= CFG.step_limit) and (not self.done):
            self.done = True
            self.done_reason = "timeout"

        r2, c2 = self.pos
        next_state: State = (sid, r2, c2, self.has_key)

        info: Dict[str, Any] = {
            "steps": self.steps,
            "done_reason": self.done_reason,
            "blocked": blocked_now,
            "hit_trap": hit_trap_now,
            "picked_key_now": picked_key_now,
        }
        return next_state, float(reward), bool(self.done), info

    def current_state(self) -> State:
        if self.episode is None:
            raise RuntimeError("Call reset() first.")
        sid = self.episode.scenario.scenario_id
        r, c = self.pos
        return (sid, r, c, self.has_key)

    def render_ascii(self) -> str:
        if self.episode is None:
            return "<no episode>"
        g = [row[:] for row in self.episode.grid]
        r, c = self.pos
        g[r][c] = "A"
        return "\n".join("".join(row) for row in g)

    def list_scenarios(self) -> List[Scenario]:
        return self.library
