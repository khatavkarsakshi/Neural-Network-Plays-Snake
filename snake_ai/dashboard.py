import time

import numpy as np
import pygame

from snake_game import WIDTH as BOARD_W, HEIGHT as BOARD_H

WIN_W, WIN_H = 1280, 720

BG = (8, 10, 14)
PANEL = (14, 18, 25)
PANEL_EDGE = (30, 38, 50)
TEXT = (226, 232, 240)
MUTED = (120, 134, 154)
TEAL = (45, 212, 191)
TEAL_DIM = (22, 84, 80)
TEAL_LIGHT = (204, 251, 241)
CORAL = (251, 113, 133)
NODE_OFF = (24, 32, 44)

INPUT_LABELS = [
    "danger straight", "danger right", "danger left",
    "going up", "going down", "going left", "going right",
    "food up", "food down", "food left", "food right",
]
OUTPUT_LABELS = ["straight", "left", "right"]

HIDDEN_SHOWN = 16


def moving_average(values, window=10):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _lerp(c1, c2, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Dashboard:
    """Pygame window: board + live network diagram + live score chart + formulas."""

    def __init__(self, fps=30, turbo=False):
        pygame.init()
        pygame.display.set_caption("Neural network that plays Snake")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.turbo = turbo
        self.paused = False
        self.quit = False
        self._last_draw = 0.0

        family = "consolas,menlo,dejavusansmono,couriernew,monospace"
        self.f_title = pygame.font.SysFont(family, 26, bold=True)
        self.f_mid = pygame.font.SysFont(family, 18, bold=True)
        self.f_small = pygame.font.SysFont(family, 15)
        self.f_tiny = pygame.font.SysFont(family, 13)

        self.board_rect = pygame.Rect(20, 68, BOARD_W, BOARD_H)
        self.net_rect = pygame.Rect(680, 68, 580, 294)
        self.chart_rect = pygame.Rect(680, 378, 580, 324)
        self.info_rect = pygame.Rect(20, 560, 640, 142)

    # ------------------------------------------------------------ control
    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit = True
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.quit = True
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_t:
                    self.turbo = not self.turbo
                elif e.key == pygame.K_UP:
                    self.fps = min(240, int(self.fps * 1.5) + 1)
                elif e.key == pygame.K_DOWN:
                    self.fps = max(5, int(self.fps / 1.5))

    def frame(self, game, agent, stats):
        """Call once per game step. Draws, handles keys, and paces the game."""
        self._handle_events()
        if self.quit:
            return
        now = time.time()
        if self.turbo and not self.paused and now - self._last_draw < 0.05:
            return  # turbo: draw ~20 times per second, train at full speed
        self._draw(game, agent, stats)
        self._last_draw = now
        while self.paused and not self.quit:
            self.clock.tick(30)
            self._handle_events()
        if not self.turbo:
            self.clock.tick(self.fps)

    def close(self):
        pygame.quit()

    # ------------------------------------------------------------ helpers
    def _text(self, font, text, color, pos, anchor="topleft"):
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def _panel(self, rect):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, width=1, border_radius=10)

    # ------------------------------------------------------------ drawing
    def _draw(self, game, agent, stats):
        self.screen.fill(BG)
        self._draw_header()
        self._draw_board(game, stats)
        self._draw_network(agent)
        self._draw_chart(agent, stats)
        self._draw_info(agent, stats)
        pygame.display.flip()

    def _draw_header(self):
        self._text(self.f_title, "Neural network that plays Snake", TEXT, (20, 14))
        self._text(self.f_tiny, "11 > 64 > 3  |  NumPy only  |  Q-learning", MUTED, (20, 46))
        self._text(self.f_tiny, "SPACE pause  |  T turbo  |  UP/DOWN speed  |  ESC quit",
                   MUTED, (WIN_W - 20, 24), anchor="topright")

    def _draw_board(self, game, stats):
        r = self.board_rect
        game.draw(self.screen, r.x, r.y)
        self._text(self.f_mid, f"score {stats['score']}", TEXT, (r.x + 10, r.y + 8))
        if self.paused:
            self._text(self.f_title, "PAUSED", TEAL, r.center, anchor="center")

    def _draw_network(self, agent):
        r = self.net_rect
        s = self.screen
        self._panel(r)
        self._text(self.f_tiny, f"live network  ({HIDDEN_SHOWN} of {len(agent.last_h)} hidden units shown)",
                   MUTED, (r.x + 12, r.y + 10))
        self._text(self.f_tiny, "11 > 64 > 3, NumPy only", MUTED, (r.right - 12, r.y + 10), anchor="topright")

        x = agent.last_state
        h = agent.last_h
        q = agent.last_q
        W1 = agent.model.params["W1"]
        W2 = agent.model.params["W2"]
        n_hidden = len(h)

        top, bottom = r.y + 44, r.bottom - 18
        x_in, x_hid, x_out = r.x + 135, r.x + 320, r.x + 475
        ys_in = np.linspace(top, bottom, 11)
        ys_hid = np.linspace(top + 6, bottom - 6, HIDDEN_SHOWN)
        ys_out = np.linspace(top + 60, bottom - 60, 3)
        shown = np.linspace(0, n_hidden - 1, HIDDEN_SHOWN).astype(int)

        w1max = float(np.abs(W1).max()) + 1e-9
        w2max = float(np.abs(W2).max()) + 1e-9
        hmax = float(h.max()) + 1e-9

        # edges (drawn on a transparent overlay so alpha works)
        overlay = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        for a in range(11):
            for b, hj in enumerate(shown):
                w = W1[a, hj]
                strength = min(1.0, abs(w) / w1max) * (0.2 + 0.8 * x[a])
                color = (*(TEAL if w > 0 else CORAL), int(12 + 165 * strength))
                pygame.draw.line(overlay, color,
                                 (int(x_in - r.x), int(ys_in[a] - r.y)),
                                 (int(x_hid - r.x), int(ys_hid[b] - r.y)), 1)
        for b, hj in enumerate(shown):
            for o in range(3):
                w = W2[hj, o]
                strength = min(1.0, abs(w) / w2max) * (0.2 + 0.8 * h[hj] / hmax)
                color = (*(TEAL if w > 0 else CORAL), int(12 + 165 * strength))
                pygame.draw.line(overlay, color,
                                 (int(x_hid - r.x), int(ys_hid[b] - r.y)),
                                 (int(x_out - r.x), int(ys_out[o] - r.y)), 1)
        s.blit(overlay, r.topleft)

        # input nodes + labels
        for a in range(11):
            center = (x_in, int(ys_in[a]))
            active = x[a] > 0.5
            if active:
                pygame.draw.circle(s, TEAL_DIM, center, 10)
                pygame.draw.circle(s, TEAL, center, 6)
            else:
                pygame.draw.circle(s, NODE_OFF, center, 6)
                pygame.draw.circle(s, (60, 72, 90), center, 6, 1)
            self._text(self.f_tiny, INPUT_LABELS[a], TEXT if active else MUTED,
                       (x_in - 15, center[1]), anchor="midright")

        # hidden nodes
        for b, hj in enumerate(shown):
            center = (x_hid, int(ys_hid[b]))
            act = (h[hj] / hmax) ** 0.7
            pygame.draw.circle(s, _lerp(NODE_OFF, TEAL, act), center, 5)
            pygame.draw.circle(s, (60, 72, 90), center, 5, 1)

        # output nodes + labels
        for o in range(3):
            center = (x_out, int(ys_out[o]))
            chosen = (o == agent.last_action)
            if chosen:
                color = CORAL if agent.last_random else TEAL
                pygame.draw.circle(s, _lerp(NODE_OFF, color, 0.35), center, 14)
                pygame.draw.circle(s, color, center, 9)
            else:
                pygame.draw.circle(s, NODE_OFF, center, 9)
                pygame.draw.circle(s, (60, 72, 90), center, 9, 1)
            self._text(self.f_tiny, OUTPUT_LABELS[o], TEXT if chosen else MUTED,
                       (x_out + 18, center[1] - 8), anchor="midleft")
            self._text(self.f_tiny, f"Q {q[o]:+.1f}", MUTED, (x_out + 18, center[1] + 8), anchor="midleft")

    def _draw_chart(self, agent, stats):
        r = self.chart_rect
        s = self.screen
        self._panel(r)
        scores = stats["scores"]
        best = stats["record"]

        self._text(self.f_small, "score per game while training", TEXT, (r.x + 14, r.y + 12))
        self._text(self.f_tiny, f"best game: {best}", TEAL, (r.right - 14, r.y + 14), anchor="topright")

        plot = pygame.Rect(r.x + 56, r.y + 46, r.w - 56 - 18, r.h - 46 - 46)
        xmax = max(stats["total"], len(scores), 1)
        ymax = max(10, int(np.ceil((best + 1) / 5.0) * 5))

        def px(g):
            return plot.x + plot.w * g / xmax

        def py(v):
            return plot.bottom - plot.h * v / ymax

        # grid + y labels
        for tick in np.linspace(0, ymax, 5):
            y = int(py(tick))
            pygame.draw.line(s, PANEL_EDGE, (plot.x, y), (plot.right, y), 1)
            self._text(self.f_tiny, f"{int(round(tick))}", MUTED, (plot.x - 8, y), anchor="midright")
        # x labels
        for k in range(5):
            g = xmax * k / 4
            self._text(self.f_tiny, f"{int(round(g))}", MUTED, (int(px(g)), plot.bottom + 6), anchor="midtop")
        self._text(self.f_tiny, "game", MUTED, (plot.centerx, plot.bottom + 24), anchor="midtop")
        ylab = pygame.transform.rotate(self.f_tiny.render("apples eaten", True, MUTED), 90)
        s.blit(ylab, (r.x + 10, plot.centery - ylab.get_height() // 2))

        # "mostly random moves" marker
        g_a = agent.games_until_epsilon(0.5)
        if g_a > 0 and len(scores) >= g_a:
            xa = int(px(g_a))
            pygame.draw.line(s, (71, 85, 105), (xa, plot.top + 2), (xa, plot.bottom), 1)
            self._text(self.f_tiny, "mostly random moves", MUTED, (xa + 6, plot.top + 2))

        if len(scores) == 1:
            pygame.draw.circle(s, TEAL, (int(px(1)), int(py(scores[0]))), 3)
        elif len(scores) >= 2:
            pts = [(px(g + 1), py(v)) for g, v in enumerate(scores)]

            overlay = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
            rel = [(int(x - r.x), int(y - r.y)) for x, y in pts]
            poly = rel + [(rel[-1][0], int(plot.bottom - r.y)), (rel[0][0], int(plot.bottom - r.y))]
            pygame.draw.polygon(overlay, (*TEAL, 45), poly)
            pygame.draw.lines(overlay, (*TEAL, 150), False, rel, 1)
            s.blit(overlay, r.topleft)

            ma = moving_average(scores, 10)
            ma_pts = [(int(px(g + 1)), int(py(v))) for g, v in enumerate(ma)]
            pygame.draw.lines(s, TEAL_LIGHT, False, ma_pts, 2)

        # legend
        lx = plot.right - 190
        ly = r.bottom - 18
        w1 = self._text(self.f_tiny, "raw", TEAL, (lx, ly)).width
        self._text(self.f_tiny, "10-game average", TEAL_LIGHT, (lx + w1 + 14, ly))

    def _draw_info(self, agent, stats):
        r = self.info_rect
        self._panel(r)
        self._text(self.f_mid,
                   f"game {stats['game']}/{stats['total']}   score {stats['score']}   "
                   f"best {stats['record']}   mean {stats['mean']:.1f}",
                   TEXT, (r.x + 14, r.y + 10))

        mode = "PAUSED" if self.paused else ("TURBO" if self.turbo else f"{self.fps} fps")
        move = "explore (random)" if agent.last_random else "exploit (network)"
        self._text(self.f_small,
                   f"epsilon {agent.epsilon:.3f}   loss {agent.last_loss:.3f}   move: {move}   [{mode}]",
                   TEAL, (r.x + 14, r.y + 34))

        pygame.draw.line(self.screen, PANEL_EDGE, (r.x + 14, r.y + 58), (r.right - 14, r.y + 58), 1)
        formulas = [
            "h  = max(0, x @ W1 + b1)        Q = h @ W2 + b2",
            "y  = r + gamma * max Q_target(s')     r = +10 food, -10 crash",
            "dQ = 2 * (Q(s,a) - y)",
            "dW2 = h.T @ dQ    dh = (dQ @ W2.T) * (h > 0)    dW1 = x.T @ dh",
        ]
        for i, line in enumerate(formulas):
            self._text(self.f_tiny, line, MUTED, (r.x + 14, r.y + 66 + i * 17))