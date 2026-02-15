# scenario_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
from collections import deque

from config import CFG, WALL, GOAL, GOAL2, TRAP, KEY
from scenarios import Scenario

Pos = Tuple[int, int]


@dataclass
class Episode:
    grid: List[List[str]]
    start: Pos
    goals: List[Pos]        # 1 or 2
    key_pos: Optional[Pos]  # only for key scenario


def parse_layout(layout: Tuple[str, ...]) -> List[List[str]]:
    return [list(row) for row in layout]


def find_cells(grid: List[List[str]], ch: str) -> List[Pos]:
    out: List[Pos] = []
    for r in range(CFG.height):
        for c in range(CFG.width):
            if grid[r][c] == ch:
                out.append((r, c))
    return out


def bfs_dist(grid: List[List[str]], start: Pos) -> List[List[int]]:
    INF = 10**9
    dist = [[INF] * CFG.width for _ in range(CFG.height)]
    q = deque([start])
    dist[start[0]][start[1]] = 0

    while q:
        r, c = q.popleft()
        nd = dist[r][c] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < CFG.height and 0 <= nc < CFG.width):
                continue
            if grid[nr][nc] == WALL:
                continue
            if dist[nr][nc] > nd:
                dist[nr][nc] = nd
                q.append((nr, nc))
    return dist


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_walkable_for_start(ch: str) -> bool:
    # start should not be on walls, traps, terminals, or key
    return ch not in (WALL, TRAP, GOAL, GOAL2, KEY)


def interior_candidates(grid: List[List[str]]) -> List[Pos]:
    """All walkable cells inside the outer border."""
    out: List[Pos] = []
    for r in range(1, CFG.height - 1):
        for c in range(1, CFG.width - 1):
            if is_walkable_for_start(grid[r][c]):
                out.append((r, c))
    return out


def ring_candidates(grid: List[List[str]]) -> List[Pos]:
    """Inner ring only (kept for maze preference)."""
    out: List[Pos] = []
    for r in range(1, CFG.height - 1):
        for c in range(1, CFG.width - 1):
            if not (r in (1, CFG.height - 2) or c in (1, CFG.width - 2)):
                continue
            if is_walkable_for_start(grid[r][c]):
                out.append((r, c))
    return out


def build_episode(scenario: Scenario, seed_offset: int = 0) -> Episode:
    rng = random.Random(CFG.seed + int(seed_offset) + 1000 * scenario.scenario_id)
    grid = parse_layout(scenario.layout)

    # ---- goals ----
    goals = find_cells(grid, GOAL)
    if scenario.scenario_type == "maze":
        goals2 = find_cells(grid, GOAL2)
        goals = goals + goals2

    if scenario.scenario_type == "maze":
        assert len(goals) == 2
    else:
        assert len(goals) == 1

    # ---- key ----
    key_pos: Optional[Pos] = None
    if scenario.scenario_type == "key":
        ks = find_cells(grid, KEY)
        key_pos = ks[0] if ks else None

    # ---- BFS dist maps ----
    dists_to_goals = [bfs_dist(grid, g) for g in goals]

    def dist_to_any_goal(p: Pos) -> int:
        return min(d[p[0]][p[1]] for d in dists_to_goals)

    dist_to_key = bfs_dist(grid, key_pos) if (scenario.scenario_type == "key" and key_pos is not None) else None

    # ---- hard rule: start NOT adjacent to terminals ----
    # یعنی فاصله Manhattan از هر ترمینال باید >= 2 باشد
    def not_adjacent_to_terminals(p: Pos) -> bool:
        for g in goals:
            if manhattan(p, g) <= 1:
                return False
        return True

    # ---- key scenario reachability sanity (to avoid impossible episodes) ----
    # start باید به key برسد و key باید به goal برسد (اگر key موجود باشد)
    key_to_goal_ok = True
    if scenario.scenario_type == "key" and key_pos is not None:
        if dist_to_any_goal(key_pos) >= 10**9:
            key_to_goal_ok = False

    def key_ok(p: Pos) -> bool:
        if scenario.scenario_type != "key":
            return True
        if key_pos is None or dist_to_key is None:
            return True  # نقشه ناقص؛ کرش نکن
        if not key_to_goal_ok:
            return True  # اگر key->goal غیرممکن است، سخت‌گیری را حذف می‌کنیم که کرش نکند
        if p == key_pos:
            return False
        return dist_to_key[p[0]][p[1]] < 10**9

    # =========================
    # ✅ NEW: NON-MAZE START SAMPLING (UNIFORM RANDOM)
    # traps/key: pick ANYWHERE in interior (uniform), only constraints:
    # - walkable
    # - not adjacent to terminal(s)
    # - reachable to goal
    # - for key: reachable to key and key->goal if possible
    # =========================
    if scenario.scenario_type in ("traps", "key"):
        pool = interior_candidates(grid)
        if not pool:
            return Episode(grid=grid, start=(1, 1), goals=goals, key_pos=key_pos)

        good = [
            p for p in pool
            if not_adjacent_to_terminals(p)
            and dist_to_any_goal(p) < 10**9
            and key_ok(p)
        ]

        if good:
            start = rng.choice(good)  # ✅ uniform among ALL valid cells
        else:
            # relax 1: drop reachability-to-goal, keep only "not adjacent"
            good2 = [p for p in pool if not_adjacent_to_terminals(p) and key_ok(p)]
            start = rng.choice(good2) if good2 else rng.choice(pool)

        return Episode(grid=grid, start=start, goals=goals, key_pos=key_pos)

    # =========================
    # MAZE: keep your previous preference (ring first), but still safe
    # =========================
    primary_pool = ring_candidates(grid)
    fallback_pool = interior_candidates(grid)
    if not primary_pool:
        primary_pool = fallback_pool
    if not primary_pool:
        return Episode(grid=grid, start=(1, 1), goals=goals, key_pos=key_pos)

    # For maze, keep "not adjacent" + reachable; (maze already constrained by exits etc.)
    # Use CFG.start_min_bfs_to_any_goal to avoid trivial starts, but don't over-bias.
    min_d = max(2, int(getattr(CFG, "start_min_bfs_to_any_goal", 12)))
    relax_levels = [min_d, 10, 6, 2, 0]

    def pick_maze(pool: List[Pos]) -> Optional[Pos]:
        for md in relax_levels:
            good = [
                p for p in pool
                if not_adjacent_to_terminals(p)
                and dist_to_any_goal(p) < 10**9
                and dist_to_any_goal(p) >= md
            ]
            if good:
                return rng.choice(good)  # uniform
        good2 = [p for p in pool if not_adjacent_to_terminals(p) and dist_to_any_goal(p) < 10**9]
        if good2:
            return rng.choice(good2)
        good3 = [p for p in pool if not_adjacent_to_terminals(p)]
        if good3:
            return rng.choice(good3)
        return None

    start = pick_maze(primary_pool)
    if start is None:
        start = pick_maze(fallback_pool)
    if start is None:
        start = rng.choice(primary_pool)

    return Episode(grid=grid, start=start, goals=goals, key_pos=key_pos)
