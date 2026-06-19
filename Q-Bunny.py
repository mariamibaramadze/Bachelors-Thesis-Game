import pygame
import numpy as np
import random
import sys
import time
from collections import defaultdict, deque
import csv

# CONSTANTS
# Colors (earthy / natural palette)
C_BG         = (34,  40,  34)   # dark forest green background
C_GRID_LINE  = (50,  60,  50)
C_CELL_LIGHT = (58,  80,  52)   # grass light
C_CELL_DARK  = (48,  68,  44)   # grass dark
C_WALL       = (90,  75,  55)   # brown wall
C_PANEL      = ( 22,  28,  22)
C_TEXT       = (200, 210, 190)
C_ACCENT     = (180, 210,  80)
C_REWARD_POS = ( 90, 200, 100)
C_REWARD_NEG = (200,  80,  80)
C_EPSILON    = (100, 160, 230)
C_GOLD       = (240, 190,  50)
C_WHITE      = (255, 255, 255)

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ACTION_NAMES  = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

# Training config
TRAIN_EPISODES_EASY   = 500    # Easy:   small state space, fast convergence
TRAIN_EPISODES_MEDIUM = 500   # Medium: walls + wolf, needs more exploration
TRAIN_EPISODES_HARD   = 500   # Hard:   largest grid, most complex dynamics
REPLAY_EPISODES  = 3      # how many best-policy runs to show
MAX_STEPS        = 200    # max steps per episode
WOLF_MOVE_EVERY  = 3      # wolf moves every N agent steps

# Pygame layout
PANEL_WIDTH      = 280
CELL_SIZE_EASY   = 80
CELL_SIZE_MEDIUM = 60
CELL_SIZE_HARD   = 48
FPS_TRAIN        = 30
FPS_REPLAY       = 8


# Q-LEARNING AGENT
class QLearningAgent:

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table: default value 0.0 for any unseen (state, action)
        self.q_table       = defaultdict(lambda: np.zeros(4))

    def choose_action(self, state):
        """Epsilon-greedy: random action with prob epsilon, else best known."""
        if random.random() < self.epsilon:
            return random.randint(0, 3)          # explore
        return int(np.argmax(self.q_table[state]))  # exploit

    def update(self, state, action, reward, next_state, done):
        """Bellman equation update."""
        current_q  = self.q_table[state][action]
        future_q   = 0.0 if done else np.max(self.q_table[next_state])
        target     = reward + self.gamma * future_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Reduce exploration after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Full reset for a new level."""
        self.q_table  = defaultdict(lambda: np.zeros(4))
        self.epsilon  = 1.0

    def get_best_action(self, state):
        """Pure exploitation — used during replay."""
        return int(np.argmax(self.q_table[state]))



# ENVIRONMENT BASE
class Level:

    def __init__(self):
        self.grid_size  = 5
        self.walls      = set()
        self.wolf_pos   = None
        self.step_count = 0

    def _random_free_cell(self, exclude=None):
        """Return a random cell that is not a wall and not in exclude set."""
        exclude = exclude or set()
        while True:
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            if (r, c) not in self.walls and (r, c) not in exclude:
                return (r, c)

    def _build_snake_path(self):

        path = []
        for r in range(self.grid_size):
            cols = range(self.grid_size) if r % 2 == 0 else range(self.grid_size - 1, -1, -1)
            for c in cols:
                if (r, c) not in self.walls:
                    path.append((r, c))
        return path

    def _next_carrot_pos(self):
        """Advance carrot one step along the snake path (wraps around)."""
        self._snake_idx = (self._snake_idx + 1) % len(self._snake_path)
        return self._snake_path[self._snake_idx]

    def get_state(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


# LEVEL 1 — EASY
class EasyLevel(Level):

    GRID     = 5
    EPISODES = TRAIN_EPISODES_EASY

    def __init__(self):
        super().__init__()
        self.grid_size   = self.GRID
        self._snake_path = self._build_snake_path()  # 25 cells, no walls
        self._snake_idx  = -1   # first reset() advances to index 0 → (0,0)

    def reset(self):
        # Carrot advances one step along the snake path each episode
        self.carrot_pos = self._next_carrot_pos()
        self.rabbit_pos = self._random_free_cell(exclude={self.carrot_pos})
        self.step_count = 0
        self.done       = False
        return self.get_state()

    def get_state(self):
            # Calculate relative direction to the carrot
            dr = self.carrot_pos[0] - self.rabbit_pos[0]
            dc = self.carrot_pos[1] - self.rabbit_pos[1]

            # State space size: 3 * 3 = 9 states total!
            return (int(np.sign(dr)), int(np.sign(dc)))

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        dr, dc = ACTION_DELTAS[action]
        nr = self.rabbit_pos[0] + dr
        nc = self.rabbit_pos[1] + dc

        # Clamp to grid (hitting wall costs a step but no extra penalty)
        nr = max(0, min(self.grid_size - 1, nr))
        nc = max(0, min(self.grid_size - 1, nc))
        self.rabbit_pos = (nr, nc)

        self.step_count += 1
        reward = -1  # step cost

        if self.rabbit_pos == self.carrot_pos:
            reward   = 10
            self.done = True

        if self.step_count >= MAX_STEPS:
            self.done = True

        return self.get_state(), reward, self.done



# LEVEL 2 — MEDIUM
class MediumLevel(Level):

    EPISODES = TRAIN_EPISODES_MEDIUM

    def __init__(self):
        super().__init__()
        self.grid_size = 8
        self.max_steps = 100

        # Define a simple, fixed set of walls in the center of the map
        self.walls = {(3, 2), (3, 5), (4, 2), (4, 5)}

        # An expanded sweeping patrol path that weaves through the grid
        self.patrol_points = [
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 6), (4, 6), (5, 6),
            (5, 5), (5, 4), (5, 3), (5, 2), (5, 1),
            (4, 1), (3, 1)
        ]
        self.patrol_index = 0

    def reset(self):
        self.steps_taken = 0

        # Random Spawning away from walls
        while True:
            self.rabbit_pos = (random.randint(0, 7), random.randint(0, 7))
            if self.rabbit_pos not in self.walls:
                break

        while True:
            self.carrot_pos = (random.randint(0, 7), random.randint(0, 7))
            if self.carrot_pos != self.rabbit_pos and self.carrot_pos not in self.walls:
                break

        self.patrol_index = 0
        self.wolf_pos = self.patrol_points[self.patrol_index]

        # Universal carrot snake path pattern (matching Easy Level)
        self.carrot_path = []
        for r in range(self.grid_size):
            cols = range(self.grid_size) if r % 2 == 0 else range(self.grid_size - 1, -1, -1)
            for c in cols:
                self.carrot_path.append((r, c))

        if self.carrot_pos in self.carrot_path:
            self.carrot_idx = self.carrot_path.index(self.carrot_pos)
        else:
            self.carrot_idx = 0
            self.carrot_pos = self.carrot_path[0]

        return self.get_state()

    def get_state(self):
        dr_c = self.carrot_pos[0] - self.rabbit_pos[0]
        dc_c = self.carrot_pos[1] - self.rabbit_pos[1]

        dr_w = self.wolf_pos[0] - self.rabbit_pos[0]
        dc_w = self.wolf_pos[1] - self.rabbit_pos[1]

        # 2-step wolf radar
        if abs(dr_w) + abs(dc_w) <= 2:
            w_r = int(np.sign(dr_w))
            w_c = int(np.sign(dc_w))
        else:
            w_r, w_c = 0, 0

            # Surrounding boundaries and walls radar (checks 4 adjacent directions)
        r, c = self.rabbit_pos
        wall_up = 1 if r - 1 < 0 or (r - 1, c) in self.walls else 0
        wall_down = 1 if r + 1 >= self.grid_size or (r + 1, c) in self.walls else 0
        wall_left = 1 if c - 1 < 0 or (r, c - 1) in self.walls else 0
        wall_right = 1 if c + 1 >= self.grid_size or (r, c + 1) in self.walls else 0

        return (
            int(np.sign(dr_c)), int(np.sign(dc_c)),
            w_r, w_c,
            wall_up, wall_down, wall_left, wall_right
        )

    def step(self, action):
        self.steps_taken += 1

        # Rabbit movement tracking wall collisions
        r, c = self.rabbit_pos
        next_pos = (r, c)
        if action == 0 and r > 0:
            next_pos = (r - 1, c)
        elif action == 1 and r < 7:
            next_pos = (r + 1, c)
        elif action == 2 and c > 0:
            next_pos = (r, c - 1)
        elif action == 3 and c < 7:
            next_pos = (r, c + 1)

        if next_pos not in self.walls:
            self.rabbit_pos = next_pos

        # Wolf patrol movement with 20% hesitation
        if random.random() > 0.20:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
            self.wolf_pos = self.patrol_points[self.patrol_index]

        # Carrot movement
        self.carrot_idx = (self.carrot_idx + 1) % len(self.carrot_path)
        self.carrot_pos = self.carrot_path[self.carrot_idx]

        reward = -1
        done = False

        if self.rabbit_pos == self.wolf_pos:
            reward = -150
            done = True
        elif self.rabbit_pos == self.carrot_pos:
            reward = 150
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True

        return self.get_state(), reward, done



# LEVEL 3 — HARD
class HardLevel(Level):

    EPISODES = TRAIN_EPISODES_HARD

    def __init__(self):
        super().__init__()
        self.grid_size = 11
        self.max_steps = 150  # More steps allowed to solve the large maze

        # Define a proper maze layout with corridors and obstacles
        self.walls = {
            (1, 1), (1, 2), (1, 3), (1, 5), (1, 7), (1, 8), (1, 9),
            (3, 1), (3, 3), (3, 5), (3, 7), (3, 9),
            (5, 1), (5, 2), (5, 3), (5, 5), (5, 7), (5, 8), (5, 9),
            (7, 1), (7, 3), (7, 5), (7, 7), (7, 9),
            (9, 1), (9, 2), (9, 3), (9, 5), (9, 7), (9, 8), (9, 9)
        }

    def reset(self):
        self.steps_taken = 0

        # Spawning rabbit safely outside of walls
        while True:
            self.rabbit_pos = (random.randint(0, 10), random.randint(0, 10))
            if self.rabbit_pos not in self.walls:
                break

        while True:
            self.carrot_pos = (random.randint(0, 10), random.randint(0, 10))
            if self.carrot_pos != self.rabbit_pos and self.carrot_pos not in self.walls:
                break

        # Wolf starts in the absolute center cell
        self.wolf_pos = (5, 4) if (5, 4) not in self.walls else (5, 0)

        # Establish universal carrot path mapping for 11x11 grid
        self.carrot_path = []
        for r in range(self.grid_size):
            cols = range(self.grid_size) if r % 2 == 0 else range(self.grid_size - 1, -1, -1)
            for c in cols:
                self.carrot_path.append((r, c))

        if self.carrot_pos in self.carrot_path:
            self.carrot_idx = self.carrot_path.index(self.carrot_pos)
        else:
            self.carrot_idx = 0
            self.carrot_pos = self.carrot_path[0]

        return self.get_state()

    def get_state(self):
        dr_c = self.carrot_pos[0] - self.rabbit_pos[0]
        dc_c = self.carrot_pos[1] - self.rabbit_pos[1]

        dr_w = self.wolf_pos[0] - self.rabbit_pos[0]
        dc_w = self.wolf_pos[1] - self.rabbit_pos[1]

        # 3-step wolf chase radar (expanded tracking for aggressive threat)
        if abs(dr_w) + abs(dc_w) <= 3:
            w_r = int(np.sign(dr_w))
            w_c = int(np.sign(dc_w))
        else:
            w_r, w_c = 0, 0

            # Surrounding walls radar
        r, c = self.rabbit_pos
        wall_up = 1 if r - 1 < 0 or (r - 1, c) in self.walls else 0
        wall_down = 1 if r + 1 >= self.grid_size or (r + 1, c) in self.walls else 0
        wall_left = 1 if c - 1 < 0 or (r, c - 1) in self.walls else 0
        wall_right = 1 if c + 1 >= self.grid_size or (r, c + 1) in self.walls else 0

        return (
            int(np.sign(dr_c)), int(np.sign(dc_c)),
            w_r, w_c,
            wall_up, wall_down, wall_left, wall_right
        )

    def step(self, action):
        self.steps_taken += 1

        # Rabbit movement execution
        r, c = self.rabbit_pos
        next_pos = (r, c)
        if action == 0 and r > 0:
            next_pos = (r - 1, c)
        elif action == 1 and r < 10:
            next_pos = (r + 1, c)
        elif action == 2 and c > 0:
            next_pos = (r, c - 1)
        elif action == 3 and c < 10:
            next_pos = (r, c + 1)

        if next_pos not in self.walls:
            self.rabbit_pos = next_pos

        # TETHERED CHASE WOLF AI:
        # Wolf checks distance to the rabbit. If within a 5-step detection radius,
        # it actively takes a step towards the rabbit avoiding walls.
        dist_to_rabbit = abs(self.wolf_pos[0] - self.rabbit_pos[0]) + abs(self.wolf_pos[1] - self.rabbit_pos[1])

        if dist_to_rabbit <= 5:  # Detection tether range
            wr, wc = self.wolf_pos
            best_move = (wr, wc)
            min_dist = dist_to_rabbit

            # Look at all 4 possible steps the wolf can take
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = wr + dr, wc + dc
                if 0 <= nr < 11 and 0 <= nc < 11 and (nr, nc) not in self.walls:
                    d = abs(nr - self.rabbit_pos[0]) + abs(nc - self.rabbit_pos[1])
                    if d < min_dist:
                        min_dist = d
                        best_move = (nr, nc)
            self.wolf_pos = best_move

        # Carrot movement
        self.carrot_idx = (self.carrot_idx + 1) % len(self.carrot_path)
        self.carrot_pos = self.carrot_path[self.carrot_idx]

        reward = -1
        done = False

        if self.rabbit_pos == self.wolf_pos:
            reward = -150
            done = True
        elif self.rabbit_pos == self.carrot_pos:
            reward = 150
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True

        return self.get_state(), reward, done



# RENDERER
class Renderer:
    """Handles all pygame drawing: grid, entities, stats panel."""

    def __init__(self, level, cell_size, level_name):
        self.level      = level
        self.cell_size  = cell_size
        self.level_name = level_name
        gs              = level.grid_size
        self.grid_px    = gs * cell_size
        self.win_w      = self.grid_px + PANEL_WIDTH
        self.win_h      = max(self.grid_px, 520)
        pygame.display.set_caption(f"Q-Bunny — {level_name}")
        self.screen     = pygame.display.set_mode((self.win_w, self.win_h))
        self.font_big   = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_med   = pygame.font.SysFont("monospace", 14)
        self.font_small = pygame.font.SysFont("monospace", 11)
        # load carrot sprite
        try:
            _carrot_raw = pygame.image.load("carrot.PNG").convert_alpha()
            _size = int(cell_size * 1.2)
            self.carrot_img = pygame.transform.smoothscale(
                _carrot_raw, (_size, _size))
        except Exception:
            self.carrot_img = None   # falls back to drawn carrot if file missing

        # load bunny sprite
        try:
            _bunny_raw = pygame.image.load("bunny.PNG").convert_alpha()
            _size = int(cell_size * 1.2)
            self.bunny_img = pygame.transform.smoothscale(
                _bunny_raw, (_size, _size))
        except Exception:
            self.bunny_img = None    # falls back to drawn rabbit if file missing

        # load wolf sprite
        try:
            _wolf_raw = pygame.image.load("wolf.PNG").convert_alpha()
            _size = int(cell_size * 1.2)
            self.wolf_img = pygame.transform.smoothscale(
                _wolf_raw, (_size, _size))
        except Exception:
            self.wolf_img = None

        self.reward_history  = deque(maxlen=200)
        self.episode_history = []   # (episode, total_reward)
        self.episode         = 0
        self.total_reward    = 0.0
        self.epsilon         = 1.0
        self.mode_text       = "TRAINING"

    # ── grid ──────────────────────────────────

    def draw_grid(self):
        gs = self.level.grid_size
        cs = self.cell_size
        for r in range(gs):
            for c in range(gs):
                color = C_CELL_LIGHT if (r + c) % 2 == 0 else C_CELL_DARK
                rect  = (c * cs, r * cs, cs, cs)
                pygame.draw.rect(self.screen, color, rect)
                if (r, c) in self.level.walls:
                    pygame.draw.rect(self.screen, C_WALL, rect)
                    # brick lines
                    pygame.draw.line(self.screen, (70,58,42),
                                     (c*cs, r*cs + cs//2), (c*cs + cs, r*cs + cs//2), 1)
                    mid = cs // 2
                    pygame.draw.line(self.screen, (70,58,42),
                                     (c*cs + mid, r*cs), (c*cs + mid, r*cs + cs//2), 1)

        # grid lines
        for i in range(gs + 1):
            pygame.draw.line(self.screen, C_GRID_LINE,
                             (i * cs, 0), (i * cs, gs * cs), 1)
            pygame.draw.line(self.screen, C_GRID_LINE,
                             (0, i * cs), (gs * cs, i * cs), 1)

    def draw_carrot(self, pos):
        r, c   = pos
        cs     = self.cell_size
        _size  = int(cs * 1.2)
        offset = (cs - _size) // 2
        self.screen.blit(self.carrot_img, (c * cs + offset, r * cs + offset))

    def draw_rabbit(self, pos):
        r, c   = pos
        cs     = self.cell_size
        _size  = int(cs * 1.2)
        offset = (cs - _size) // 2
        self.screen.blit(self.bunny_img, (c * cs + offset, r * cs + offset))

    def draw_wolf(self, pos):
        if pos is None:
            return
        r, c   = pos
        cs     = self.cell_size
        _size  = int(cs * 1.2)
        offset = (cs - _size) // 2
        self.screen.blit(self.wolf_img, (c * cs + offset, r * cs + offset))

    # ── panel ──────────────────────────────────

    def draw_panel(self):
        px = self.grid_px
        pw = PANEL_WIDTH
        ph = self.win_h
        pygame.draw.rect(self.screen, C_PANEL, (px, 0, pw, ph))
        pygame.draw.line(self.screen, C_ACCENT, (px, 0), (px, ph), 2)

        y = 16
        def txt(text, color=C_TEXT, font=None, indent=12):
            nonlocal y
            f = font or self.font_med
            surf = f.render(text, True, color)
            self.screen.blit(surf, (px + indent, y))
            y += surf.get_height() + 4

        # Title
        txt(f"── Q-Bunny ──", C_ACCENT, self.font_big, indent=20)
        txt(f"Level: {self.level_name}", C_GOLD, self.font_big)
        txt(f"Mode:  {self.mode_text}", C_WHITE)
        y += 6

        # Stats
        txt(f"Episode:   {self.episode}", C_TEXT)
        r_color = C_REWARD_POS if self.total_reward >= 0 else C_REWARD_NEG
        txt(f"Reward:    {self.total_reward:.1f}", r_color)
        txt(f"Epsilon:   {self.epsilon:.3f}", C_EPSILON)

        if self.episode_history:
            last10 = [x[1] for x in self.episode_history[-10:]]
            txt(f"Avg(10):   {np.mean(last10):.1f}", C_TEXT)
            best = max(x[1] for x in self.episode_history)
            txt(f"Best:      {best:.1f}", C_GOLD)

        y += 8
        txt("── Legend ──", C_ACCENT, indent=20)
        txt("Rabbit  (agent)", (220, 210, 195))
        txt("Carrot  (+reward)", (230, 120, 30))
        if self.level.wolf_pos is not None:
            txt("Wolf    (danger)", (230, 50, 50))
        if self.level.walls:
            txt("Wall    (blocked)", C_WALL)

        y += 8
        txt("── Reward Chart ──", C_ACCENT, indent=20)
        self._draw_sparkline(px + 14, y, pw - 28, 80)
        y += 88

        y += 4
        txt("── Q-Learning ──", C_ACCENT, indent=20)
        txt(f"α (learn):  0.10", C_TEXT)
        txt(f"γ (discount):0.95", C_TEXT)
        txt(f"ε-decay:   0.995", C_TEXT)
        txt(f"Q states:  {len(self._agent.q_table) if hasattr(self, '_agent') else 0}", C_TEXT)

    def _draw_sparkline(self, x, y, w, h):
        """Mini reward chart in the panel."""
        pygame.draw.rect(self.screen, (30, 38, 30), (x, y, w, h))
        pygame.draw.rect(self.screen, C_GRID_LINE,  (x, y, w, h), 1)
        if len(self.episode_history) < 2:
            return
        rewards = [r for _, r in self.episode_history[-w:]]
        mn, mx  = min(rewards), max(rewards)
        span    = mx - mn if mx != mn else 1
        pts = []
        for i, r in enumerate(rewards):
            px_ = x + int(i / max(len(rewards) - 1, 1) * w)
            py_ = y + h - int((r - mn) / span * (h - 4)) - 2
            pts.append((px_, py_))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, C_ACCENT, False, pts, 2)
        # zero line
        if mn < 0 < mx:
            zy = y + h - int((0 - mn) / span * (h - 4)) - 2
            pygame.draw.line(self.screen, C_GRID_LINE, (x, zy), (x + w, zy), 1)

    # ── full frame ──────────────────────────────

    def draw(self, agent=None):
        self.screen.fill(C_BG)
        self.draw_grid()
        self.draw_carrot(self.level.carrot_pos)
        self.draw_rabbit(self.level.rabbit_pos)
        self.draw_wolf(self.level.wolf_pos)
        if agent:
            self._agent = agent
        self.draw_panel()
        pygame.display.flip()



# TRAINER / ORCHESTRATOR
class Trainer:

    def __init__(self, level, agent, renderer, level_name):
        self.level      = level
        self.agent      = agent
        self.renderer   = renderer
        self.level_name = level_name
        self.clock      = pygame.time.Clock()
        self.best_policy_reward = -999
        self.best_q_snapshot   = None
        self.successful_episodes = 0
        self.success_steps = []
        self.training_time = 0.0

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _snapshot_q(self):
        """Save a copy of the Q-table for replay."""
        return {k: v.copy() for k, v in self.agent.q_table.items()}

    def train(self):
        """Run per-level episodes, rendering live."""
        train_episodes = getattr(self.level, 'EPISODES', TRAIN_EPISODES_EASY)
        print(f"\n{'='*50}")
        print(f"  Training: {self.level_name}")
        print(f"  Episodes: {train_episodes}  |  Max steps: {MAX_STEPS}")
        print(f"{'='*50}")

        self.renderer.mode_text = "TRAINING"
        start_time = time.time()

        for ep in range(1, train_episodes + 1):
            state = self.level.reset()
            total_reward = 0.0

            for step in range(MAX_STEPS):
                if not self._handle_events():
                    return  # user closed window

                action          = self.agent.choose_action(state)
                next_state, reward, done = self.level.step(action)
                self.agent.update(state, action, reward, next_state, done)
                state        = next_state
                total_reward += reward

                # Render every few steps (speed up training)
                if step % 3 == 0:
                    self.renderer.episode      = ep
                    self.renderer.total_reward = total_reward
                    self.renderer.epsilon      = self.agent.epsilon
                    self.renderer.draw(self.agent)
                    self.clock.tick(FPS_TRAIN)


                # vizualis gareshe darunva da pirdapir resultebze gadasvlistvis
                # if ep % 20 == 0 and step % 4 == 0:
                #     self.renderer.episode = ep
                #     self.renderer.total_reward = total_reward
                #     self.renderer.epsilon = self.agent.epsilon
                #     self.renderer.draw(self.agent)

                if done:
                    if reward > 0:
                        self.successful_episodes += 1
                        self.success_steps.append(step + 1)
                    break

            self.agent.decay_epsilon()
            self.renderer.episode_history.append((ep, total_reward))

            if total_reward > self.best_policy_reward:
                self.best_policy_reward = total_reward
                self.best_q_snapshot    = self._snapshot_q()

            if ep % 100 == 0:
                avg = np.mean([r for _, r in self.renderer.episode_history[-100:]])
                print(f"  Ep {ep:4d} | Avg(100): {avg:7.2f} | ε: {self.agent.epsilon:.3f}")

        self.training_time = time.time() - start_time

        with open(f"{self.level_name.replace(' ','_')}_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode","Reward"])
            for ep_num, rew in self.renderer.episode_history:
                writer.writerow([ep_num, rew])

        print(f"\n  Training complete. Best reward: {self.best_policy_reward:.1f}")
        print(f"  Success Rate: {(self.successful_episodes/train_episodes)*100:.2f}%")
        if self.success_steps:
            print(f"  Avg Steps (successes): {np.mean(self.success_steps):.2f}")
        print(f"  Training Time: {self.training_time:.2f}s")

    def replay(self):
        """Replay best learned policy (no exploration)."""
        if self.best_q_snapshot is None:
            return

        print(f"\n  Replaying best policy for {self.level_name}...")
        self.renderer.mode_text = "REPLAY — BEST POLICY"

        # Load best Q-table snapshot
        saved_q       = self.agent.q_table
        saved_epsilon = self.agent.epsilon
        self.agent.q_table  = defaultdict(lambda: np.zeros(4), self.best_q_snapshot)
        self.agent.epsilon   = 0.0   # pure exploitation

        for run in range(REPLAY_EPISODES):
            state        = self.level.reset()
            total_reward = 0.0
            print(f"    Replay run {run+1}/{REPLAY_EPISODES}", end="", flush=True)

            for step in range(MAX_STEPS):
                if not self._handle_events():
                    break

                action = self.agent.get_best_action(state)
                state, reward, done = self.level.step(action)
                total_reward += reward

                self.renderer.episode      = f"REPLAY {run+1}"
                self.renderer.total_reward = total_reward
                self.renderer.epsilon      = 0.0
                self.renderer.draw(self.agent)
                self.clock.tick(FPS_REPLAY)

                if done:
                    break

            print(f"  →  reward: {total_reward:.1f}")
            time.sleep(0.5)

        # Restore
        self.agent.q_table  = saved_q
        self.agent.epsilon   = saved_epsilon



# RESULTS SCREEN
def show_results_screen(screen, results, font_big, font_med):
    """Summary screen shown after all 3 levels complete."""
    screen.fill(C_BG)
    w, h = screen.get_size()

    def center_text(text, y, font, color):
        surf = font.render(text, True, color)
        screen.blit(surf, (w // 2 - surf.get_width() // 2, y))

    center_text("══ Q-Bunny — RESULTS ══", 40,  font_big, C_GOLD)
    center_text("Complexity Comparison", 75, font_med, C_ACCENT)

    headers = ["Level", "Episodes", "Best Reward", "Avg(Last 100)"]
    col_x   = [60, 220, 380, 540]
    y       = 130
    for i, h_ in enumerate(headers):
        surf = font_big.render(h_, True, C_ACCENT)
        screen.blit(surf, (col_x[i], y))

    pygame.draw.line(screen, C_ACCENT, (50, y + 28), (w - 50, y + 28), 1)
    y += 42

    for name, hist, best in results:
        vals = [
            name,
            str(len(hist)),
            f"{best:.1f}",
            f"{np.mean([r for _,r in hist[-100:]]):.1f}",
        ]
        color = C_TEXT
        for i, v in enumerate(vals):
            surf = font_med.render(v, True, color)
            screen.blit(surf, (col_x[i], y))
        y += 34

    center_text("Press any key to exit", h - 60, font_med, C_GRID_LINE)
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False



# MAIN

def run_level(LevelClass, cell_size, name):
    level    = LevelClass()
    agent    = QLearningAgent()
    renderer = Renderer(level, cell_size, name)

    trainer = Trainer(level, agent, renderer, name)
    trainer.train()
    trainer.replay()

    return (name.split("—")[0].strip(),
            renderer.episode_history,
            trainer.best_policy_reward)

def main():
    pygame.init()

    print("\n=== Q-Bunny Menu ===")
    print("1. Easy only")
    print("2. Medium only")
    print("3. Hard only")
    print("4. Run all levels")
    print("5. Exit")

    choice = input("Select option: ").strip()

    level_configs = [
        (EasyLevel,   CELL_SIZE_EASY,   "EASY"),
        (MediumLevel, CELL_SIZE_MEDIUM, "MEDIUM"),
        (HardLevel,   CELL_SIZE_HARD,   "HARD"),
    ]

    if choice == "5":
        return

    results = []

    if choice in ("1", "2", "3"):
        idx = int(choice) - 1
        results.append(run_level(*level_configs[idx]))
    else:
        for cfg in level_configs:
            results.append(run_level(*cfg))

    font_big = pygame.font.SysFont("monospace", 18, bold=True)
    font_med = pygame.font.SysFont("monospace", 14)

    final_screen = pygame.display.set_mode((750, 420))
    pygame.display.set_caption("Q-Bunny — Results")
    show_results_screen(final_screen, results, font_big, font_med)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()