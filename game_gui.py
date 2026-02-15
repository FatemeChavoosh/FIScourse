# # game_gui.py
# from __future__ import annotations

# import os
# import sys
# import pygame
# from typing import List, Tuple, Optional

# from config import CFG, WALL, EMPTY, GOAL, TRAP, KEY

# # GOAL2 optional
# try:
#     from config import GOAL2
# except Exception:
#     GOAL2 = None  # type: ignore

# from scenarios import get_scenario_library
# from replays import ReplayStore, EpisodeReplay


# CELL = 34
# PAD = 12
# PANEL_W = 460
# FPS = 60


# def clamp(x: int, a: int, b: int) -> int:
#     return max(a, min(b, x))


# def draw_text(screen, font, text, x, y, color=(255, 255, 255)):
#     s = font.render(text, True, color)
#     screen.blit(s, (x, y))


# def parse_grid0(rep: EpisodeReplay, fallback_layout: Tuple[str, ...]) -> List[str]:
#     if rep.grid0 and len(rep.grid0) == CFG.height:
#         return rep.grid0
#     return list(fallback_layout)


# class Button:
#     def __init__(self, rect: pygame.Rect, label: str):
#         self.rect = rect
#         self.label = label
#         self.enabled = True
#         self._hover = False

#     def update_hover(self, mouse_pos):
#         self._hover = self.rect.collidepoint(mouse_pos)

#     def draw(self, screen, font, *, primary=False):
#         if not self.enabled:
#             bg = (55, 55, 55)
#             border = (80, 80, 80)
#             txt = (160, 160, 160)
#         else:
#             if primary:
#                 bg = (60, 110, 180) if not self._hover else (75, 130, 210)
#             else:
#                 bg = (60, 60, 60) if not self._hover else (80, 80, 80)
#             border = (110, 110, 110)
#             txt = (245, 245, 245)

#         pygame.draw.rect(screen, bg, self.rect, border_radius=8)
#         pygame.draw.rect(screen, border, self.rect, 2, border_radius=8)

#         # centered text
#         surf = font.render(self.label, True, txt)
#         tx = self.rect.x + (self.rect.w - surf.get_width()) // 2
#         ty = self.rect.y + (self.rect.h - surf.get_height()) // 2
#         screen.blit(surf, (tx, ty))

#     def clicked(self, event) -> bool:
#         if not self.enabled:
#             return False
#         if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
#             return self.rect.collidepoint(event.pos)
#         return False


# def main():
#     replay_path = os.path.join("runs", "run_01", "replays_all.json")
#     if len(sys.argv) >= 2:
#         replay_path = sys.argv[1]

#     store = ReplayStore(replay_path)
#     store.load()

#     library = get_scenario_library()
#     scenario_ids = store.scenario_ids()
#     if not scenario_ids:
#         raise RuntimeError("No replays found in replay store.")

#     pygame.init()
#     W = PAD * 3 + PANEL_W + CFG.width * CELL
#     H = PAD * 2 + CFG.height * CELL

#     screen = pygame.display.set_mode((W, H))
#     pygame.display.set_caption("GridWorld Replay Viewer (Mouse Controls)")

#     font = pygame.font.SysFont("Menlo", 18)
#     font2 = pygame.font.SysFont("Menlo", 16)
#     font_btn = pygame.font.SysFont("Menlo", 16)

#     sc_idx = 0
#     ep_idx = 0

#     playing = True
#     paused = False
#     speed = 10  # steps per second
#     frame = 0

#     clock = pygame.time.Clock()
#     step_accum = 0.0

#     def current():
#         nonlocal ep_idx
#         sid = scenario_ids[sc_idx]
#         eps = store.get_episodes(sid)
#         if not eps:
#             return sid, None, []
#         ep_idx = clamp(ep_idx, 0, len(eps) - 1)
#         return sid, eps[ep_idx], eps

#     def scenario_obj(sid: int):
#         return next(x for x in library if x.scenario_id == sid)

#     def jump_episode(delta: int):
#         nonlocal ep_idx, frame, paused, step_accum
#         sid, rep, eps = current()
#         if not eps:
#             return
#         ep_idx = clamp(ep_idx + delta, 0, len(eps) - 1)
#         frame = 0
#         step_accum = 0.0
#         paused = False

#     def jump_scenario(delta: int):
#         nonlocal sc_idx, ep_idx, frame, paused, step_accum
#         sc_idx = (sc_idx + delta) % len(scenario_ids)
#         ep_idx = 0
#         frame = 0
#         step_accum = 0.0
#         paused = False

#     def draw_grid(grid_lines: List[str], agent_pos: Tuple[int, int], picked_key: bool):
#         ox = PAD * 2 + PANEL_W
#         oy = PAD

#         for r in range(CFG.height):
#             row = grid_lines[r]
#             for c in range(CFG.width):
#                 ch = row[c]
#                 x = ox + c * CELL
#                 y = oy + r * CELL
#                 rect = pygame.Rect(x, y, CELL, CELL)

#                 # background
#                 if ch == WALL:
#                     pygame.draw.rect(screen, (60, 60, 60), rect)
#                 else:
#                     pygame.draw.rect(screen, (25, 25, 25), rect)

#                 # special tiles
#                 if ch == GOAL:
#                     pygame.draw.rect(screen, (0, 110, 0), rect.inflate(-6, -6))
#                 if (GOAL2 is not None) and (ch == GOAL2):
#                     pygame.draw.rect(screen, (0, 80, 140), rect.inflate(-6, -6))
#                 if ch == TRAP:
#                     pygame.draw.rect(screen, (140, 0, 0), rect.inflate(-10, -10))
#                 if ch == KEY and not picked_key:
#                     pygame.draw.rect(screen, (180, 160, 0), rect.inflate(-10, -10))

#                 pygame.draw.rect(screen, (40, 40, 40), rect, 1)

#         # agent
#         ar, ac = agent_pos
#         ax = ox + ac * CELL + CELL // 2
#         ay = oy + ar * CELL + CELL // 2
#         pygame.draw.circle(screen, (230, 230, 230), (ax, ay), CELL // 3)

#     # -------------------------
#     # Build mouse buttons (left panel)
#     # -------------------------
#     bx = PAD
#     by = PAD + 150
#     bw = (PANEL_W - PAD * 2 - 10) // 2
#     bh = 40
#     gap = 10

#     # Scenario buttons
#     btn_sc_prev = Button(pygame.Rect(bx, by, bw, bh), "Scenario Prev")
#     btn_sc_next = Button(pygame.Rect(bx + bw + gap, by, bw, bh), "Scenario Next")

#     # Episode buttons
#     by2 = by + bh + 12
#     btn_ep_prev = Button(pygame.Rect(bx, by2, bw, bh), "Episode Prev")
#     btn_ep_next = Button(pygame.Rect(bx + bw + gap, by2, bw, bh), "Episode Next")

#     # Episode +/-10
#     by3 = by2 + bh + 12
#     btn_ep_m10 = Button(pygame.Rect(bx, by3, bw, bh), "Episode -10")
#     btn_ep_p10 = Button(pygame.Rect(bx + bw + gap, by3, bw, bh), "Episode +10")

#     # Play / speed
#     by4 = by3 + bh + 16
#     btn_play = Button(pygame.Rect(bx, by4, bw, bh), "Play/Pause")
#     btn_reset = Button(pygame.Rect(bx + bw + gap, by4, bw, bh), "Restart")

#     by5 = by4 + bh + 12
#     btn_slower = Button(pygame.Rect(bx, by5, bw, bh), "Slower")
#     btn_faster = Button(pygame.Rect(bx + bw + gap, by5, bw, bh), "Faster")

#     buttons = [
#         btn_sc_prev, btn_sc_next,
#         btn_ep_prev, btn_ep_next,
#         btn_ep_m10, btn_ep_p10,
#         btn_play, btn_reset,
#         btn_slower, btn_faster
#     ]

#     while playing:
#         dt = clock.tick(FPS) / 1000.0
#         mouse_pos = pygame.mouse.get_pos()
#         for b in buttons:
#             b.update_hover(mouse_pos)

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 playing = False

#             if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
#                 playing = False

#             # Mouse clicks
#             if btn_sc_prev.clicked(event):
#                 jump_scenario(-1)
#             if btn_sc_next.clicked(event):
#                 jump_scenario(+1)

#             if btn_ep_prev.clicked(event):
#                 jump_episode(-1)
#             if btn_ep_next.clicked(event):
#                 jump_episode(+1)

#             if btn_ep_m10.clicked(event):
#                 jump_episode(-10)
#             if btn_ep_p10.clicked(event):
#                 jump_episode(+10)

#             if btn_play.clicked(event):
#                 paused = not paused

#             if btn_reset.clicked(event):
#                 frame = 0
#                 step_accum = 0.0
#                 paused = False

#             if btn_slower.clicked(event):
#                 speed = max(1, speed - 1)
#             if btn_faster.clicked(event):
#                 speed = min(60, speed + 1)

#         sid, rep, eps = current()
#         sc = scenario_obj(sid)

#         screen.fill((15, 15, 15))

#         # left panel info
#         x0 = PAD
#         y0 = PAD

#         draw_text(screen, font, "Replay Viewer", x0, y0)
#         y0 += 28

#         draw_text(screen, font2, f"Replay file:", x0, y0)
#         y0 += 18
#         draw_text(screen, font2, f"{replay_path}", x0, y0, (200, 200, 200))
#         y0 += 26

#         draw_text(screen, font2, f"Scenario: {sid}  ({sc.scenario_type})", x0, y0)
#         y0 += 20
#         draw_text(screen, font2, f"Name: {sc.name}", x0, y0)
#         y0 += 20

#         # Draw buttons
#         for b in buttons:
#             b.draw(screen, font_btn, primary=(b in (btn_sc_next, btn_ep_next, btn_ep_p10)))

#         # Rep info
#         info_y = by5 + bh + 18
#         if rep is None:
#             draw_text(screen, font2, "No episodes for this scenario.", x0, info_y, (255, 120, 120))
#             pygame.display.flip()
#             continue

#         draw_text(screen, font2, f"Episode idx: {rep.episode_idx}   ({ep_idx+1}/{len(eps)})", x0, info_y)
#         info_y += 20
#         draw_text(screen, font2, f"epsilon={rep.epsilon:.3f}  steps={rep.steps}  R={rep.total_reward:.1f}", x0, info_y)
#         info_y += 20
#         col = (120, 255, 120) if rep.success else (255, 120, 120)
#         draw_text(screen, font2, f"done={rep.done_reason}  success={rep.success}", x0, info_y, col)
#         info_y += 20
#         draw_text(screen, font2, f"paused={paused}  speed={speed} step/s", x0, info_y)
#         info_y += 10

#         # advance video
#         if not paused:
#             step_accum += dt * speed
#             while step_accum >= 1.0:
#                 step_accum -= 1.0
#                 frame += 1
#                 if frame >= len(rep.positions):
#                     frame = len(rep.positions) - 1
#                     paused = True
#                     break

#         # key picked marker
#         picked_key = False
#         if rep.picked_key_step is not None and frame >= (rep.picked_key_step + 1):
#             picked_key = True

#         # draw grid
#         grid_lines = parse_grid0(rep, sc.layout)
#         agent_pos = rep.positions[frame]
#         draw_grid(grid_lines, agent_pos, picked_key=picked_key)

#         # timeline
#         t_y = H - PAD - 22
#         bar_x = PAD
#         bar_w = PANEL_W - PAD
#         pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(bar_x, t_y, bar_w, 10), 1)
#         if len(rep.positions) > 1:
#             p = frame / (len(rep.positions) - 1)
#             pygame.draw.rect(screen, (180, 180, 180), pygame.Rect(bar_x, t_y, int(bar_w * p), 10))

#         draw_text(screen, font2, f"frame {frame+1}/{len(rep.positions)}", PAD, t_y - 18)

#         pygame.display.flip()

#     pygame.quit()


# if __name__ == "__main__":
#     main()


# game_gui.py
from __future__ import annotations

import os
import sys
import pygame
from typing import List, Tuple, Optional, Dict, Any

from config import CFG, WALL, EMPTY, GOAL, TRAP, KEY

# GOAL2 optional
try:
    from config import GOAL2
except Exception:
    GOAL2 = None  # type: ignore

from scenarios import get_scenario_library
from replays import ReplayStore, EpisodeReplay


CELL = 34
PAD = 12
PANEL_W = 460
FPS = 60


def clamp(x: int, a: int, b: int) -> int:
    return max(a, min(b, x))


def draw_text(screen, font, text, x, y, color=(255, 255, 255)):
    s = font.render(text, True, color)
    screen.blit(s, (x, y))


def parse_grid0(rep: EpisodeReplay, fallback_layout: Tuple[str, ...]) -> List[str]:
    if rep.grid0 and len(rep.grid0) == CFG.height:
        return rep.grid0
    return list(fallback_layout)


class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label
        self.enabled = True
        self._hover = False

    def update_hover(self, mouse_pos):
        self._hover = self.rect.collidepoint(mouse_pos)

    def draw(self, screen, font, *, primary=False):
        if not self.enabled:
            bg = (55, 55, 55)
            border = (80, 80, 80)
            txt = (160, 160, 160)
        else:
            if primary:
                bg = (60, 110, 180) if not self._hover else (75, 130, 210)
            else:
                bg = (60, 60, 60) if not self._hover else (80, 80, 80)
            border = (110, 110, 110)
            txt = (245, 245, 245)

        pygame.draw.rect(screen, bg, self.rect, border_radius=8)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=8)

        # centered text
        surf = font.render(self.label, True, txt)
        tx = self.rect.x + (self.rect.w - surf.get_width()) // 2
        ty = self.rect.y + (self.rect.h - surf.get_height()) // 2
        screen.blit(surf, (tx, ty))

    def clicked(self, event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False


def _safe_load_store(path: str) -> Tuple[Optional[ReplayStore], Optional[str]]:
    """
    Returns (ReplayStore or None, error_message or None)
    """
    try:
        st = ReplayStore(path)
        st.load()
        return st, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    # Backward compatible CLI:
    #   python game_gui.py                       -> uses defaults
    #   python game_gui.py <q_replays.json>      -> Q path override
    #   python game_gui.py <q.json> <sarsa.json> -> both overrides
    q_path = os.path.join("runs", "run_01", "replays_all.json")
    sarsa_path = os.path.join("runs", "sarsa_run_01", "replays_all.json")

    if len(sys.argv) >= 2:
        q_path = sys.argv[1]
    if len(sys.argv) >= 3:
        sarsa_path = sys.argv[2]

    store_q, err_q = _safe_load_store(q_path)
    store_s, err_s = _safe_load_store(sarsa_path)

    if store_q is None and store_s is None:
        raise RuntimeError(
            "No replay stores could be loaded.\n"
            f"Q path: {q_path}\n  error: {err_q}\n"
            f"SARSA path: {sarsa_path}\n  error: {err_s}\n"
        )

    # active algorithm: prefer Q if available, else SARSA
    algo = "q" if store_q is not None else "sarsa"

    library = get_scenario_library()

    # scenario ids: union of both stores (so you can keep same scenario while toggling)
    ids = set()
    if store_q is not None:
        ids.update(store_q.scenario_ids())
    if store_s is not None:
        ids.update(store_s.scenario_ids())
    scenario_ids = sorted(ids)

    if not scenario_ids:
        raise RuntimeError("No replays found in loaded replay stores.")

    pygame.init()
    W = PAD * 3 + PANEL_W + CFG.width * CELL
    H = PAD * 2 + CFG.height * CELL

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("GridWorld Replay Viewer (Q-learning / SARSA)")

    font = pygame.font.SysFont("Menlo", 18)
    font2 = pygame.font.SysFont("Menlo", 16)
    font_btn = pygame.font.SysFont("Menlo", 16)

    sc_idx = 0
    ep_idx = 0

    playing = True
    paused = False
    speed = 10  # steps per second
    frame = 0

    clock = pygame.time.Clock()
    step_accum = 0.0

    def get_store() -> Optional[ReplayStore]:
        return store_q if algo == "q" else store_s

    def get_replay_path() -> str:
        return q_path if algo == "q" else sarsa_path

    def current() -> Tuple[int, Optional[EpisodeReplay], List[EpisodeReplay]]:
        nonlocal ep_idx
        sid = scenario_ids[sc_idx]
        st = get_store()
        if st is None:
            return sid, None, []
        eps = st.get_episodes(sid)
        if not eps:
            return sid, None, []
        ep_idx = clamp(ep_idx, 0, len(eps) - 1)
        return sid, eps[ep_idx], eps

    def scenario_obj(sid: int):
        return next(x for x in library if x.scenario_id == sid)

    def reset_video_state():
        nonlocal frame, paused, step_accum
        frame = 0
        step_accum = 0.0
        paused = False

    def jump_episode(delta: int):
        nonlocal ep_idx
        sid, rep, eps = current()
        if not eps:
            return
        ep_idx = clamp(ep_idx + delta, 0, len(eps) - 1)
        reset_video_state()

    def jump_scenario(delta: int):
        nonlocal sc_idx, ep_idx
        sc_idx = (sc_idx + delta) % len(scenario_ids)
        ep_idx = 0
        reset_video_state()

    def set_algo(new_algo: str):
        nonlocal algo, ep_idx
        if new_algo == algo:
            return
        if new_algo == "q" and store_q is None:
            return
        if new_algo == "sarsa" and store_s is None:
            return

        # keep same scenario id (sc_idx stays), but episode list changes -> clamp ep_idx
        algo = new_algo
        sid, rep, eps = current()
        if not eps:
            ep_idx = 0
        else:
            ep_idx = clamp(ep_idx, 0, len(eps) - 1)
        reset_video_state()

    def draw_grid(grid_lines: List[str], agent_pos: Tuple[int, int], picked_key: bool):
        ox = PAD * 2 + PANEL_W
        oy = PAD

        for r in range(CFG.height):
            row = grid_lines[r]
            for c in range(CFG.width):
                ch = row[c]
                x = ox + c * CELL
                y = oy + r * CELL
                rect = pygame.Rect(x, y, CELL, CELL)

                # background
                if ch == WALL:
                    pygame.draw.rect(screen, (60, 60, 60), rect)
                else:
                    pygame.draw.rect(screen, (25, 25, 25), rect)

                # special tiles
                if ch == GOAL:
                    pygame.draw.rect(screen, (0, 110, 0), rect.inflate(-6, -6))
                if (GOAL2 is not None) and (ch == GOAL2):
                    pygame.draw.rect(screen, (0, 80, 140), rect.inflate(-6, -6))
                if ch == TRAP:
                    pygame.draw.rect(screen, (140, 0, 0), rect.inflate(-10, -10))
                if ch == KEY and not picked_key:
                    pygame.draw.rect(screen, (180, 160, 0), rect.inflate(-10, -10))

                pygame.draw.rect(screen, (40, 40, 40), rect, 1)

        # agent
        ar, ac = agent_pos
        ax = ox + ac * CELL + CELL // 2
        ay = oy + ar * CELL + CELL // 2
        pygame.draw.circle(screen, (230, 230, 230), (ax, ay), CELL // 3)

    # -------------------------
    # Build mouse buttons (left panel)
    # -------------------------
    bx = PAD
    by = PAD + 150
    bw = (PANEL_W - PAD * 2 - 10) // 2
    bh = 40
    gap = 10

    # Algo buttons (top of the control cluster)
    btn_algo_q = Button(pygame.Rect(bx, by - (bh + 12), bw, bh), "Show Q-learning")
    btn_algo_s = Button(pygame.Rect(bx + bw + gap, by - (bh + 12), bw, bh), "Show SARSA")
    btn_algo_q.enabled = store_q is not None
    btn_algo_s.enabled = store_s is not None

    # Scenario buttons
    btn_sc_prev = Button(pygame.Rect(bx, by, bw, bh), "Scenario Prev")
    btn_sc_next = Button(pygame.Rect(bx + bw + gap, by, bw, bh), "Scenario Next")

    # Episode buttons
    by2 = by + bh + 12
    btn_ep_prev = Button(pygame.Rect(bx, by2, bw, bh), "Episode Prev")
    btn_ep_next = Button(pygame.Rect(bx + bw + gap, by2, bw, bh), "Episode Next")

    # Episode +/-10
    by3 = by2 + bh + 12
    btn_ep_m10 = Button(pygame.Rect(bx, by3, bw, bh), "Episode -10")
    btn_ep_p10 = Button(pygame.Rect(bx + bw + gap, by3, bw, bh), "Episode +10")

    # Play / speed
    by4 = by3 + bh + 16
    btn_play = Button(pygame.Rect(bx, by4, bw, bh), "Play/Pause")
    btn_reset = Button(pygame.Rect(bx + bw + gap, by4, bw, bh), "Restart")

    by5 = by4 + bh + 12
    btn_slower = Button(pygame.Rect(bx, by5, bw, bh), "Slower")
    btn_faster = Button(pygame.Rect(bx + bw + gap, by5, bw, bh), "Faster")

    buttons = [
        btn_algo_q, btn_algo_s,
        btn_sc_prev, btn_sc_next,
        btn_ep_prev, btn_ep_next,
        btn_ep_m10, btn_ep_p10,
        btn_play, btn_reset,
        btn_slower, btn_faster
    ]

    while playing:
        dt = clock.tick(FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()
        for b in buttons:
            b.update_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                playing = False

            # Mouse clicks: algorithm toggle
            if btn_algo_q.clicked(event):
                set_algo("q")
            if btn_algo_s.clicked(event):
                set_algo("sarsa")

            # Mouse clicks: navigation
            if btn_sc_prev.clicked(event):
                jump_scenario(-1)
            if btn_sc_next.clicked(event):
                jump_scenario(+1)

            if btn_ep_prev.clicked(event):
                jump_episode(-1)
            if btn_ep_next.clicked(event):
                jump_episode(+1)

            if btn_ep_m10.clicked(event):
                jump_episode(-10)
            if btn_ep_p10.clicked(event):
                jump_episode(+10)

            if btn_play.clicked(event):
                paused = not paused

            if btn_reset.clicked(event):
                reset_video_state()

            if btn_slower.clicked(event):
                speed = max(1, speed - 1)
            if btn_faster.clicked(event):
                speed = min(60, speed + 1)

        sid, rep, eps = current()
        sc = scenario_obj(sid)

        screen.fill((15, 15, 15))

        # left panel info
        x0 = PAD
        y0 = PAD

        title = "Replay Viewer"
        draw_text(screen, font, title, x0, y0)
        y0 += 28

        # Active algorithm indicator
        algo_label = "Q-learning" if algo == "q" else "SARSA"
        draw_text(screen, font2, f"Algo: {algo_label}", x0, y0, (220, 220, 220))
        y0 += 20

        draw_text(screen, font2, "Replay file:", x0, y0)
        y0 += 18
        draw_text(screen, font2, f"{get_replay_path()}", x0, y0, (200, 200, 200))
        y0 += 26

        # If one store failed, show a small warning (non-fatal)
        if err_q and store_q is None:
            draw_text(screen, font2, "Q-learning replays not loaded.", x0, y0, (255, 150, 150))
            y0 += 18
        if err_s and store_s is None:
            draw_text(screen, font2, "SARSA replays not loaded.", x0, y0, (255, 150, 150))
            y0 += 18
        if (err_q and store_q is None) or (err_s and store_s is None):
            y0 += 6

        draw_text(screen, font2, f"Scenario: {sid}  ({sc.scenario_type})", x0, y0)
        y0 += 20
        draw_text(screen, font2, f"Name: {sc.name}", x0, y0)
        y0 += 20

        # Draw buttons (highlight the active algo button as primary)
        for b in buttons:
            primary = False
            if b is btn_algo_q and algo == "q":
                primary = True
            if b is btn_algo_s and algo == "sarsa":
                primary = True
            # keep your original "primary" accents for next buttons too
            if b in (btn_sc_next, btn_ep_next, btn_ep_p10):
                primary = True if not primary else True
            b.draw(screen, font_btn, primary=primary)

        # Rep info
        info_y = by5 + bh + 18
        if rep is None:
            draw_text(screen, font2, "No episodes for this scenario in this algo.", x0, info_y, (255, 120, 120))
            pygame.display.flip()
            continue

        draw_text(screen, font2, f"Episode idx: {rep.episode_idx}   ({ep_idx+1}/{len(eps)})", x0, info_y)
        info_y += 20
        draw_text(screen, font2, f"epsilon={rep.epsilon:.3f}  steps={rep.steps}  R={rep.total_reward:.1f}", x0, info_y)
        info_y += 20
        col = (120, 255, 120) if rep.success else (255, 120, 120)
        draw_text(screen, font2, f"done={rep.done_reason}  success={rep.success}", x0, info_y, col)
        info_y += 20
        draw_text(screen, font2, f"paused={paused}  speed={speed} step/s", x0, info_y)
        info_y += 10

        # advance video
        if not paused:
            step_accum += dt * speed
            while step_accum >= 1.0:
                step_accum -= 1.0
                frame += 1
                if frame >= len(rep.positions):
                    frame = len(rep.positions) - 1
                    paused = True
                    break

        # key picked marker
        picked_key = False
        if rep.picked_key_step is not None and frame >= (rep.picked_key_step + 1):
            picked_key = True

        # draw grid
        grid_lines = parse_grid0(rep, sc.layout)
        agent_pos = rep.positions[frame]
        draw_grid(grid_lines, agent_pos, picked_key=picked_key)

        # timeline
        t_y = H - PAD - 22
        bar_x = PAD
        bar_w = PANEL_W - PAD
        pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(bar_x, t_y, bar_w, 10), 1)
        if len(rep.positions) > 1:
            p = frame / (len(rep.positions) - 1)
            pygame.draw.rect(screen, (180, 180, 180), pygame.Rect(bar_x, t_y, int(bar_w * p), 10))

        draw_text(screen, font2, f"frame {frame+1}/{len(rep.positions)}", PAD, t_y - 18)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
