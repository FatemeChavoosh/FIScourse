# scenarios.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from config import CFG, WALL, GOAL, GOAL2, TRAP, KEY

ScenarioType = Literal["maze", "traps", "key"]


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    scenario_type: ScenarioType
    layout: Tuple[str, ...]  # CFG.height strings each len CFG.width
    name: str


def _pos_of(layout: Tuple[str, ...], ch: str) -> List[Tuple[int, int]]:
    out = []
    for r, row in enumerate(layout):
        for c, cc in enumerate(row):
            if cc == ch:
                out.append((r, c))
    return out


def _validate(layout: Tuple[str, ...], s_type: ScenarioType) -> None:
    # shape
    assert len(layout) == CFG.height, f"Expected {CFG.height} rows, got {len(layout)}"
    assert all(len(r) == CFG.width for r in layout), "Row length mismatch"

    # top/bottom borders must be walls
    assert all(ch == WALL for ch in layout[0]), "Top border must be walls"
    assert all(ch == WALL for ch in layout[-1]), "Bottom border must be walls"

    # counts
    g1 = sum(r.count(GOAL) for r in layout)
    g2 = sum(r.count(GOAL2) for r in layout)

    if s_type == "maze":
        assert g1 == 1 and g2 == 1, "Maze must have exactly 2 terminals: one G and one H"
    else:
        assert g1 == 1 and g2 == 0, "Non-maze must have exactly 1 terminal: G only"

    if s_type == "key":
        k = sum(r.count(KEY) for r in layout)
        assert k == 1, "Key scenario must have exactly 1 key K"

    # border rules
    if s_type != "maze":
        # strict walls on side borders
        assert all(row[0] == WALL and row[-1] == WALL for row in layout), "Side borders must be walls"
    else:
        gpos = _pos_of(layout, GOAL)[0]
        hpos = _pos_of(layout, GOAL2)[0]

        # exits must be on side borders
        assert gpos[1] in (0, CFG.width - 1), "G must be on left/right border"
        assert hpos[1] in (0, CFG.width - 1), "H must be on left/right border"

        # not on top/bottom row
        assert gpos[0] not in (0, CFG.height - 1), "G cannot be on top/bottom row"
        assert hpos[0] not in (0, CFG.height - 1), "H cannot be on top/bottom row"

        # must be on different sides
        assert gpos[1] != hpos[1], "G and H must be on different sides"

        # neighbor inside must be open (not wall)
        if gpos[1] == 0:
            assert layout[gpos[0]][1] != WALL, "Cell next to G (inside) must be open"
        else:
            assert layout[gpos[0]][CFG.width - 2] != WALL, "Cell next to G (inside) must be open"

        if hpos[1] == 0:
            assert layout[hpos[0]][1] != WALL, "Cell next to H (inside) must be open"
        else:
            assert layout[hpos[0]][CFG.width - 2] != WALL, "Cell next to H (inside) must be open"

        # all other side-border cells must be WALL
        for r in range(CFG.height):
            left = layout[r][0]
            right = layout[r][-1]
            if (r, 0) == gpos or (r, 0) == hpos:
                pass
            else:
                assert left == WALL, f"Left border must be wall except exits (row {r})"
            if (r, CFG.width - 1) == gpos or (r, CFG.width - 1) == hpos:
                pass
            else:
                assert right == WALL, f"Right border must be wall except exits (row {r})"


def _fix_nonmaze(layout: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    For traps/key: guarantee 15x15 and strict side walls.
    - trunc/pad each row to width
    - force first/last char of every row to '#'
    - force top/bottom rows to all '#'
    """
    assert len(layout) == CFG.height, f"Expected {CFG.height} rows, got {len(layout)}"

    fixed_rows: List[str] = []
    for r, row in enumerate(layout):
        row2 = (row[:CFG.width]).ljust(CFG.width, WALL)
        # force side walls
        row2 = WALL + row2[1:CFG.width - 1] + WALL
        fixed_rows.append(row2)

    fixed_rows[0] = WALL * CFG.width
    fixed_rows[-1] = WALL * CFG.width
    return tuple(fixed_rows)


# -------------------------
# Maze corridors (7)
# - exact 15x15 already
# - G and H on opposite side borders
# - inside-neighbor of exits is '.'
# - all other side border cells are '#'
# -------------------------
def _maze_corridors() -> List[Tuple[str, ...]]:
    mazes: List[Tuple[str, ...]] = []

    # Maze 0: G left row 3, H right row 11
    mazes.append((
        "###############",
        "#..#.....#....#",
        "#..#.###.#.##.#",
        "G..#...#.#....#",
        "#.###.#.#.###.#",
        "#.....#.#.....#",
        "#####.#.#####.#",
        "#...#.#.....#.#",
        "#.#.#.#####.#.#",
        "#.#...#...#...#",
        "#.#####.#.###.#",
        "#....#..#.....H",
        "#.##.#.####.#.#",
        "#....#......#.#",
        "###############",
    ))

    # Maze 1: G right row 9, H left row 12
    mazes.append((
        "###############",
        "#....#....#...#",
        "#.##.#.##.#.#.#",
        "#.#..#..#...#.#",
        "#.#.###.#####.#",
        "#.#.....#.....#",
        "#.#####.#.###.#",
        "#.....#.#...#.#",
        "#.###.#.###.#.#",
        "#....#..#.....G",
        "###.#######.#.#",
        "#.....#.....#.#",
        "H.###.#.###...#",
        "#.....#.......#",
        "###############",
    ))

    # Maze 2: G left row 7, H right row 5
    mazes.append((
        "###############",
        "#....#........#",
        "#.##.#.######.#",
        "#....#......#.#",
        "#.####.####.#.#",
        "#.#....#......H",
        "#.#.##.#.##.#.#",
        "G.#....#....#.#",
        "#.######.####.#",
        "#......#......#",
        "#.####.#.####.#",
        "#.#....#....#.#",
        "#.#.#########.#",
        "#.............#",
        "###############",
    ))

    # Maze 3: G right row 4, H left row 10
    mazes.append((
        "###############",
        "#.....#.......#",
        "#.###.#.#####.#",
        "#.#...#.....#.#",
        "#.#.#####.#...G",
        "#.#.....#.#.#.#",
        "#.#####.#.#.#.#",
        "#.....#...#...#",
        "###.#.#####.###",
        "#...#.....#...#",
        "H.#######.###.#",
        "#.......#.....#",
        "#.#####.#####.#",
        "#.............#",
        "###############",
    ))

    # Maze 4: G left row 12, H right row 2
    mazes.append((
        "###############",
        "#..#..#.......#",
        "#..#..#.###...H",
        "#..#..#.....#.#",
        "#..####.###.#.#",
        "#.......#...#.#",
        "#.#####.#.###.#",
        "#.#...#.#.....#",
        "#.#.#.#.#####.#",
        "#...#.#.....#.#",
        "###.#.#####.#.#",
        "#...#.....#...#",
        "G.#######.#.###",
        "#.............#",
        "###############",
    ))

    # Maze 5: G right row 6, H left row 8  (FIXED: row length + border correctness)
    mazes.append((
        "###############",
        "#........#....#",
        "#.######.#.##.#",
        "#.#....#.#....#",
        "#.#.##.#.####.#",
        "#.#.##.#......#",
        "#.#....#####..G",   # 15 chars, last is G, inside-neighbor (col 13) is '.'
        "#.##########..#",
        "H.....#.......#",
        "#####.#.#####.#",
        "#.....#.....#.#",
        "#.#########.#.#",
        "#.........#...#",
        "#.#######.###.#",
        "###############",
    ))

    # Maze 6: G left row 5, H right row 13
    mazes.append((
        "###############",
        "#..#..........#",
        "#..#.#######..#",
        "#..#.....#....#",
        "#.#####..#.##.#",
        "G.....#..#....#",
        "#####.#..####.#",
        "#...#.#.......#",
        "#.#.#.#######.#",
        "#.#...#.....#.#",
        "#.#####.###.#.#",
        "#.......#...#.#",
        "#.#######.###.#",
        "#.............H",
        "###############",
    ))

    assert len(mazes) == CFG.n_maze, (len(mazes), CFG.n_maze)
    return mazes


# -------------------------
# Trap maps (7)
# - strict side walls (enforced by _fix_nonmaze)
# - exactly one G
# -------------------------
def _trap_open() -> List[Tuple[str, ...]]:
    raw: List[Tuple[str, ...]] = []

    raw.append((
        "###############",
        "#..###.....###.",
        "#..T..#..T..#..",
        "#..#..#..#..#..",
        "#..#..##..###..",
        "#..#....T......",
        "#..####...#.#..",
        "#..T....##...T.",
        "#..###...####..",
        "#......T.......",
        "#..###.###..#..",
        "#..#..#..T..#..",
        "#........#..#..",
        "#..T..###G.##..",
        "###############",
    ))

    raw.append((
        "###############",
        "#..####...####.",
        "#..T..#T..#..T.",
        "#..#..#....#..#",
        "#..#..###.###..",
        "#.......T..#..#",
        "#..####...#...#",
        "#..T.....##.T..",
        "#..##.#..##...#",
        "#..#..T..#....#",
        "#..#.###...#..#",
        "#..#..#..T..#..",
        "#..T..##..#.#..",
        "#.....##G..T..#",
        "###############",
    ))

    raw.append((
        "###############",
        "#..###..###...#",
        "#..T..#..T..#..",
        "#.....#.....#..",
        "#..#.###.###.#.",
        "#.T...TT...#...",
        "#..##.#..#.##..",
        "#T.T....##....T",
        "#..####..##.#..",
        "#...T....T..#..",
        "#.#.###.###.#..",
        "#..#..#..T..#..",
        "#.....#..#..#..",
        "#..T..##G.T..T.",
        "###############",
    ))

    raw.append((
        "###############",
        "#....###...##.#",
        "#..T..#..T..#..",
        "#..#..#..##.#..",
        "#..#.##..#....#",
        "#..#....T..#...",
        "#..####..#.##..",
        "#..T....##..T..",
        "#..###...####..",
        "#...#..T....T..",
        "#..##..#.#.##..",
        "#..#..#..T..#..",
        "#..#..#..T..#..",
        "#..T..##.G###..",
        "###############",
    ))

    raw.append((
        "###############",
        "#.......###...#",
        "#.T.T.#..T..#..",
        "#..#.....##.#..",
        "#..#.###.#...#.",
        "#..#..T..T.T#..",
        "#######..#..#..",
        "#..T....##T....",
        "#..##.#..####..",
        "#..T..#........",
        "#..#.T#T.#.##T.",
        "#..#..#..T.....",
        "#...T.#.T#..#..",
        "#.T...##G....T.",
        "###############",
    ))

    raw.append((
        "###############",
        "#..####....##.#",
        "#..T.....T..#..",
        "#..#..#..#.....",
        "#..#.###..##.#.",
        "#..#....T..#...",
        "#..##.#..#.##..",
        "#..T..T.##.T..T",
        "#....##..####..",
        "#...#....T.....",
        "#..##..###.##..",
        "#..#..#..T..#..",
        "#..#.....#..#..",
        "#..T..##.G.##..",
        "###############",
    ))

    raw.append((
        "###############",
        "#..#....###...#",
        "#..T.T#..T..T..",
        "#..#..#..##.#..",
        "#..#T.##.###...",
        "#..#..T........",
        "#######..####..",
        "#..T....##.T...",
        "#....##..#.##..",
        "#..T..#T...#...",
        "#..#.#.###.##..",
        "#..#..#..T..#..",
        "#..#..#..#..#..",
        "#..T....G##....",
        "###############",
    ))

    fixed = [_fix_nonmaze(lay) for lay in raw]
    assert len(fixed) == CFG.n_trap, (len(fixed), CFG.n_trap)
    return fixed


# -------------------------
# Key scenarios (8)
# - strict side walls (enforced by _fix_nonmaze)
# - exactly one K and one G
# - 4 maze-like + 4 trap-like (as requested)
# -------------------------
def _key_maps() -> List[Tuple[str, ...]]:
    raw: List[Tuple[str, ...]] = []

    # 4 maze-like (corridor-ish)
    raw.append((
        "###############",
        "#K....#.......#",
        "#.###.#.#####.#",
        "#.#...#.....#.#",
        "#.#.#####.#.#.#",
        "#.#.....#.#.#.#",
        "#.#####.#.#.#.#",
        "#.....#...#...#",
        "###.#.#####.###",
        "#...#.....#...#",
        "#.#######.###.#",
        "#.......#.....#",
        "#.#####.#####.#",
        "#...........G.#",
        "###############",
    ))

    raw.append((
        "###############",
        "#.....#...K...#",
        "#.###.#.#####.#",
        "#...#.#.....#.#",
        "###.#.#####.#.#",
        "#...#.....#.#.#",
        "#.#######.#.#.#",
        "#.......#.#...#",
        "#.#####.#.###.#",
        "#.#...#.#.....#",
        "#.#.#.#.#######",
        "#...#.#.......#",
        "#.###.#####.#.#",
        "#..G..........#",
        "###############",
    ))

    raw.append((
        "###############",
        "#..K..#.......#",
        "#.###.#.#####.#",
        "#.#...#...#.#.#",
        "#.#.#####.#.#.#",
        "#.#.....#.#...#",
        "#.#####.#.###.#",
        "#.....#...#...#",
        "###.#.#####.###",
        "#...#.....#...#",
        "#.#######.###.#",
        "#.......#.....#",
        "#.#####.#####.#",
        "#....G........#",
        "###############",
    ))

    raw.append((
        "###############",
        "#.....#.....K.#",
        "#.###.#.#####.#",
        "#...#.#.....#.#",
        "###.#.#####.#.#",
        "#...#.....#.#.#",
        "#.#######.#.#.#",
        "#.......#.#...#",
        "#.#####.#.###.#",
        "#.#...#.#.....#",
        "#.#.#.#.#######",
        "#...#.#.......#",
        "#.###.#####.#.#",
        "#...........G.#",
        "###############",
    ))

    # 4 trap-like (clutter + traps + key)
    raw.append((
        "###############",
        "#..##......###.",
        "#..T..#..T..#..",
        "#..#..#..#..#..",
        "#..#..###..##..",
        "#.....K.T......",
        "#..####..##.#..",
        "#..T....##...T.",
        "#..####..####..",
        "#......T....#..",
        "#..###..##.....",
        "#..#..T..T..#..",
        "#..#.....#..#..",
        "#..T..###G.##..",
        "###############",
    ))

    raw.append((
        "###############",
        "#..##.......##.",
        "#..T..#T..#..T.",
        "#..#..#..#....#",
        "#.....#.#.###..",
        "#..#.T..T..#..#",
        "#..####..#....#",
        "#..T..K..##.T..",
        "#T.####..##...#",
        "#..#..T..#....#",
        "#..#..##.#.#..#",
        "#.##..#..T..#..",
        "#..T..##..#.#..",
        "#.....##G..T..#",
        "###############",
    ))

    raw.append((
        "###############",
        "#..##...###...#",
        "##.T..#..T..#..",
        "#.....#..#..#..",
        "#..#.###.##..#.",
        "#T.#..T....#...",
        "#..####..#.##..",
        "#..T..K.##....T",
        "#.T####..##.#..",
        "#...#....T..#..",
        "#T#.###.###.#..",
        "#.....#..T..#..",
        "#..#..#..#..#..",
        "#..T..##G##..T.",
        "###############",
    ))

    raw.append((
        "###############",
        "#..##....####.#",
        "#..T..#..T.....",
        "#..#..#..##.#..",
        "#..#.###..#.###",
        "#.......T......",
        "#..####..####..",
        "#..T....##..T..",
        "#..####..##.#..",
        "#..K#..T....#..",
        "#..##..###.###.",
        "#TT...#..T.....",
        "#..#........#..",
        "#..T..##.G###..",
        "###############",
    ))

    fixed = [_fix_nonmaze(lay) for lay in raw]
    assert len(fixed) == CFG.n_key, (len(fixed), CFG.n_key)
    return fixed


def get_scenario_library() -> List[Scenario]:
    scenarios: List[Scenario] = []
    sid = 0

    for i, lay in enumerate(_maze_corridors()):
        _validate(lay, "maze")
        scenarios.append(Scenario(sid, "maze", lay, f"maze_{i:02d}"))
        sid += 1

    for i, lay in enumerate(_trap_open()):
        _validate(lay, "traps")
        scenarios.append(Scenario(sid, "traps", lay, f"traps_{i:02d}"))
        sid += 1

    for i, lay in enumerate(_key_maps()):
        _validate(lay, "key")
        scenarios.append(Scenario(sid, "key", lay, f"key_{i:02d}"))
        sid += 1

    assert len(scenarios) == CFG.n_scenarios, (len(scenarios), CFG.n_scenarios)
    return scenarios


if __name__ == "__main__":
    lib = get_scenario_library()
    print("OK scenarios:", len(lib))
