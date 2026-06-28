import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR  = os.path.join(SCRIPT_DIR, "..", "assets")
img_bunny  = mpimg.imread(os.path.join(ASSETS_DIR, "bunny.png"))
img_carrot = mpimg.imread(os.path.join(ASSETS_DIR, "carrot.PNG"))
img_wolf   = mpimg.imread(os.path.join(ASSETS_DIR, "wolf.PNG"))

REF_PX = img_bunny.shape[0]

def zoom_for(img, base):
    return base * (REF_PX / img.shape[0])

# ── Actual Medium level: 8x8, walls at {(3,2),(3,5),(4,2),(4,5)} ─────────────
ROWS, COLS = 8, 8
WALLS = {(3,2),(3,5),(4,2),(4,5)}

GRID = [[0]*8 for _ in range(8)]
for (r,c) in WALLS:
    GRID[r][c] = 1

# Positions chosen so the wolf is OUTSIDE radar distance 2 (dist = 4),
# so the figure honestly shows the radar-masking behaviour: wolf_dir = (0, 0)
# even though the wolf is not literally at the same position as the rabbit.
BUNNY_POS  = (2, 3)
WOLF_POS   = (2, 7)   # Manhattan dist = 4 → outside radar, masked to (0,0)
CARROT_POS = (6, 6)

WALL_COLOR   = "#AAAAAA"
FREE_COLOR   = "#FFFFFF"
GRID_COLOR   = "#BBBBBB"
ACCENT_GREEN = "#2D8A3E"
ACCENT_RED   = "#CC2222"
ACCENT_BLUE  = "#1A5FA8"
TEXT_DARK    = "#111111"
BOX_BG       = "#F5F5F5"

br, bc = BUNNY_POS
wr, wc = WOLF_POS
cr, cc = CARROT_POS

CARROT_DIR = (int((cr-br>0)-(cr-br<0)), int((cc-bc>0)-(cc-bc<0)))

# Radar gating, matching MediumLevel.get_state() exactly:
# wolf_dir is only the true sign-direction when within Manhattan distance 2;
# otherwise it is masked to (0, 0), regardless of the wolf's real position.
WOLF_DIST = abs(wr-br) + abs(wc-bc)
RADAR_RANGE = 2
if WOLF_DIST <= RADAR_RANGE:
    WOLF_DIR = (int((wr-br>0)-(wr-br<0)), int((wc-bc>0)-(wc-bc<0)))
    RADAR_ACTIVE = True
else:
    WOLF_DIR = (0, 0)
    RADAR_ACTIVE = False

# Wall sensor order matches get_state()'s actual return order: up, down, left, right
WALL_VEC = [
    1 if br-1 < 0 or (br-1,bc) in WALLS else 0,   # up
    1 if br+1 >= ROWS or (br+1,bc) in WALLS else 0,  # down
    1 if bc-1 < 0 or (br,bc-1) in WALLS else 0,   # left
    1 if bc+1 >= COLS or (br,bc+1) in WALLS else 0,  # right
]


def place(ax, img, zoom, x, y):
    ob = OffsetImage(img, zoom=zoom, interpolation="nearest")
    ob.image.axes = ax
    ax.add_artist(AnnotationBbox(ob, (x, y), frameon=False, zorder=5))


fig = plt.figure(figsize=(13, 9.5), facecolor="white")
fig.text(0.5, 0.972, "Medium / Hard Level Environment (Example)",
         ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT_DARK)

# ── Grid ──────────────────────────────────────────────────────────────────────
ax_grid = fig.add_axes([0.04, 0.40, 0.50, 0.54], aspect="equal")

for r in range(ROWS):
    for c in range(COLS):
        fc = WALL_COLOR if GRID[r][c] else FREE_COLOR
        ax_grid.add_patch(patches.Rectangle((c, ROWS-1-r), 1, 1,
                          lw=0.8, ec=GRID_COLOR, fc=fc))

gz = 0.045
place(ax_grid, img_bunny,  zoom_for(img_bunny,  gz), bc+0.5, ROWS-1-br+0.5)
place(ax_grid, img_wolf,   zoom_for(img_wolf,   gz), wc+0.5, ROWS-1-wr+0.5)
place(ax_grid, img_carrot, zoom_for(img_carrot, gz), cc+0.5, ROWS-1-cr+0.5)

ax_grid.set_xlim(0, COLS); ax_grid.set_ylim(0, ROWS)
ax_grid.axis("off")
for sp in ax_grid.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color("#555")

# ── Legend ────────────────────────────────────────────────────────────────────
ax_leg = fig.add_axes([0.62, 0.50, 0.34, 0.46])
ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1); ax_leg.axis("off")

ax_leg.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                  boxstyle="round,pad=0.03", lw=1.5, ec="#888",
                  fc="white", ls="--", transform=ax_leg.transAxes))
ax_leg.text(0.5, 0.93, "Legend", ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax_leg.transAxes)

lz = 0.038
for img, y, label in [
        (img_bunny,  0.75, "Rabbit (Agent)"),
        (img_wolf,   0.54, "Wolf (Enemy)"),
        (img_carrot, 0.33, "Carrot (Goal)")]:
    ob = OffsetImage(img, zoom=zoom_for(img, lz), interpolation="nearest")
    ax_leg.add_artist(AnnotationBbox(ob, (0.19, y),
                      xycoords="axes fraction", frameon=False))
    ax_leg.text(0.37, y, label, va="center", fontsize=10,
                transform=ax_leg.transAxes)

ax_leg.add_patch(patches.Rectangle((0.10, 0.12), 0.18, 0.10,
                  fc=WALL_COLOR, ec="#666", lw=1, transform=ax_leg.transAxes))
ax_leg.text(0.37, 0.17, "Wall (Obstacle)", va="center", fontsize=10,
            transform=ax_leg.transAxes)

# ── Bottom state diagram ──────────────────────────────────────────────────────
ax_bot = fig.add_axes([0.03, 0.01, 0.94, 0.36])
ax_bot.set_xlim(0, 1); ax_bot.set_ylim(0, 1); ax_bot.axis("off")

ax_bot.add_patch(FancyBboxPatch((0.005, 0.03), 0.99, 0.94,
                  boxstyle="round,pad=0.02", lw=1.2, ec="#CCCCCC",
                  fc=BOX_BG, transform=ax_bot.transAxes))

ax_bot.text(0.5, 0.93, "State Representation",
            ha="center", fontsize=11, fontweight="bold", va="top",
            transform=ax_bot.transAxes, color=TEXT_DARK)

for x in [0.345, 0.655]:
    ax_bot.plot([x, x], [0.08, 0.88], color="#CCCCCC", lw=1,
                transform=ax_bot.transAxes)

sz = 0.055
# ── Carrot Direction ──────────────────────────────────────────────────────────
ax_bot.text(0.17, 0.81, "Carrot Direction", ha="center", fontsize=10,
            fontweight="bold", transform=ax_bot.transAxes, color=TEXT_DARK)

place(ax_bot, img_carrot, zoom_for(img_carrot, sz), 0.10, 0.57)
place(ax_bot, img_bunny,  zoom_for(img_bunny,  sz), 0.24, 0.57)
ax_bot.annotate("", xy=(0.205, 0.57), xytext=(0.158, 0.57),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->, head_width=0.25", color="#555", lw=1.2))

ax_bot.text(0.175, 0.37, f"row: sign({cr}−{br}) = ", ha="center", fontsize=8.5,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.300, 0.37, f"{CARROT_DIR[0]:+d}", fontsize=9, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_GREEN)
ax_bot.text(0.175, 0.25, f"col:  sign({cc}−{bc}) = ", ha="center", fontsize=8.5,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.300, 0.25, f"{CARROT_DIR[1]:+d}", fontsize=9, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_GREEN)

ax_bot.add_patch(FancyBboxPatch((0.03, 0.06), 0.285, 0.16,
                  boxstyle="round,pad=0.01", lw=1, ec="#AAAAAA",
                  fc="white", transform=ax_bot.transAxes))
ax_bot.text(0.175, 0.145, f"({CARROT_DIR[0]:+d},  {CARROT_DIR[1]:+d})",
            ha="center", fontsize=13, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_GREEN)

# ── Wolf Direction ────────────────────────────────────────────────────────────
ax_bot.text(0.50, 0.81, "Wolf Direction", ha="center", fontsize=10,
            fontweight="bold", transform=ax_bot.transAxes, color=TEXT_DARK)
radar_label = (f"(Manhattan dist = {WOLF_DIST} ≤ {RADAR_RANGE} → radar active)"
               if RADAR_ACTIVE else
               f"(Manhattan dist = {WOLF_DIST} > {RADAR_RANGE} → out of range, masked to (0,0))")
ax_bot.text(0.50, 0.72, radar_label,
            ha="center", fontsize=7.5, transform=ax_bot.transAxes, color="#777")

place(ax_bot, img_wolf,  zoom_for(img_wolf,  sz), 0.43, 0.57)
place(ax_bot, img_bunny, zoom_for(img_bunny, sz), 0.57, 0.57)
ax_bot.annotate("", xy=(0.535, 0.57), xytext=(0.488, 0.57),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->, head_width=0.25", color="#555", lw=1.2))

if RADAR_ACTIVE:
    ax_bot.text(0.500, 0.37, f"row: sign({wr}−{br}) = ", ha="center", fontsize=8.5,
                transform=ax_bot.transAxes, color=TEXT_DARK)
    ax_bot.text(0.625, 0.37, f"{WOLF_DIR[0]:+d}", fontsize=9, fontweight="bold",
                transform=ax_bot.transAxes, color=ACCENT_RED)
    ax_bot.text(0.500, 0.25, f"col:  sign({wc}−{bc}) = ", ha="center", fontsize=8.5,
                transform=ax_bot.transAxes, color=TEXT_DARK)
    ax_bot.text(0.625, 0.25, f"{WOLF_DIR[1]:+d}", fontsize=9, fontweight="bold",
                transform=ax_bot.transAxes, color=ACCENT_RED)
else:
    ax_bot.text(0.500, 0.37, "dist > radar range", ha="center", fontsize=8.5,
                transform=ax_bot.transAxes, color=TEXT_DARK)
    ax_bot.text(0.500, 0.25, "→ direction masked", ha="center", fontsize=8.5,
                transform=ax_bot.transAxes, color=TEXT_DARK)

ax_bot.add_patch(FancyBboxPatch((0.365, 0.06), 0.27, 0.16,
                  boxstyle="round,pad=0.01", lw=1, ec="#AAAAAA",
                  fc="white", transform=ax_bot.transAxes))
ax_bot.text(0.50, 0.145, f"({WOLF_DIR[0]:+d},  {WOLF_DIR[1]:+d})",
            ha="center", fontsize=13, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_RED)

# ── Wall Sensor ───────────────────────────────────────────────────────────────
ax_bot.text(0.83, 0.81, "Wall Sensor", ha="center", fontsize=10,
            fontweight="bold", transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.83, 0.72, "[Up,  Down,  Left,  Right]",
            ha="center", fontsize=8, transform=ax_bot.transAxes, color="#777")

MINI = 0.088
cx0, cy0 = 0.786, 0.43
ax_bot.add_patch(patches.Rectangle((cx0, cy0), MINI, MINI,
                  lw=1, ec="#888", fc=FREE_COLOR, transform=ax_bot.transAxes))
place(ax_bot, img_bunny, 0.028, cx0+MINI/2, cy0+MINI/2)

# WALL_VEC is [up, down, left, right]; map each to its compass offset on screen
for dx, dy, is_wall, arrow in [
        (0,  1, WALL_VEC[0], "↑"),   # up
        (1,  0, WALL_VEC[3], "→"),   # right
        (0, -1, WALL_VEC[1], "↓"),   # down
        (-1, 0, WALL_VEC[2], "←")]:  # left
    nx, ny = cx0+dx*MINI, cy0+dy*MINI
    ax_bot.add_patch(patches.Rectangle((nx, ny), MINI, MINI,
                      lw=1, ec="#888",
                      fc=WALL_COLOR if is_wall else FREE_COLOR,
                      transform=ax_bot.transAxes))
    ax_bot.text(nx+MINI/2, ny+MINI/2, arrow, ha="center", va="center",
                fontsize=9, color="white" if is_wall else "#888",
                transform=ax_bot.transAxes)

ax_bot.add_patch(FancyBboxPatch((0.695, 0.06), 0.27, 0.16,
                  boxstyle="round,pad=0.01", lw=1, ec="#AAAAAA",
                  fc="white", transform=ax_bot.transAxes))
wv = WALL_VEC
ax_bot.text(0.83, 0.145, f"[{wv[0]},  {wv[1]},  {wv[2]},  {wv[3]}]",
            ha="center", fontsize=12, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_BLUE)

cd = CARROT_DIR; wd = WOLF_DIR
ax_bot.text(0.5, -0.05,
            f"State  =  ( carrot_dir, wolf_dir, walls )  =  "
            f"( ({cd[0]:+d},{cd[1]:+d}),  ({wd[0]:+d},{wd[1]:+d}),  [{wv[0]},{wv[1]},{wv[2]},{wv[3]}] )",
            ha="center", fontsize=10, fontweight="bold",
            transform=ax_bot.transAxes, color=TEXT_DARK)

fig.text(0.5, 0.005,
         " ",
         ha="center", va="bottom", fontsize=11, fontstyle="italic", color=TEXT_DARK)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, "figure 2.png"), dpi=200, bbox_inches="tight", facecolor="white")
print(f"Carrot dir: {CARROT_DIR}, Wolf dir: {WOLF_DIR}, Wolf dist: {WOLF_DIST}, "
      f"Radar active: {RADAR_ACTIVE}, Walls [up,down,left,right]: {WALL_VEC}")
print(f"Saved {os.path.join(OUTPUT_DIR, 'figure 2.png')}")
plt.show()