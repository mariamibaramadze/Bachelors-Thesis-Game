import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR  = os.path.join(SCRIPT_DIR, "..", "assets")    # figures/ -> ../assets/
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "generated")        # figures/generated/
os.makedirs(OUTPUT_DIR, exist_ok=True)

img_bunny  = mpimg.imread(os.path.join(ASSETS_DIR, "bunny.png"))
img_carrot = mpimg.imread(os.path.join(ASSETS_DIR, "carrot.PNG"))

REF_PX = img_bunny.shape[0]

# ── Actual Easy level: 5x5, no walls ─────────────────────────────────────────
ROWS, COLS = 5, 5
GRID = [[0]*5 for _ in range(5)]   # all free, no walls in Easy

BUNNY_POS  = (1, 0)
CARROT_POS = (3, 3)

WALL_COLOR   = "#AAAAAA"
FREE_COLOR   = "#FFFFFF"
GRID_COLOR   = "#BBBBBB"
ACCENT_GREEN = "#2D8A3E"
TEXT_DARK    = "#111111"
BOX_BG       = "#F5F5F5"

br, bc = BUNNY_POS
cr, cc = CARROT_POS
CARROT_DIR = (int((cr-br>0)-(cr-br<0)), int((cc-bc>0)-(cc-bc<0)))


def place(ax, img, zoom, x, y):
    ob = OffsetImage(img, zoom=zoom, interpolation="nearest")
    ob.image.axes = ax
    ax.add_artist(AnnotationBbox(ob, (x, y), frameon=False, zorder=5))


fig = plt.figure(figsize=(12, 9), facecolor="white")
fig.text(0.5, 0.968, "Easy Level Environment (Example)",
         ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT_DARK)

# ── Grid ──────────────────────────────────────────────────────────────────────
ax_grid = fig.add_axes([0.08, 0.38, 0.45, 0.56], aspect="equal")

for r in range(ROWS):
    for c in range(COLS):
        fc = WALL_COLOR if GRID[r][c] else FREE_COLOR
        ax_grid.add_patch(patches.Rectangle((c, ROWS-1-r), 1, 1,
                          lw=0.8, ec=GRID_COLOR, fc=fc))

gz = 0.050
zb = gz
zc = gz * (REF_PX / img_carrot.shape[0])
place(ax_grid, img_bunny,  zb, bc+0.5, ROWS-1-br+0.5)
place(ax_grid, img_carrot, zc, cc+0.5, ROWS-1-cr+0.5)

ax_grid.set_xlim(0, COLS); ax_grid.set_ylim(0, ROWS)
ax_grid.axis("off")
for sp in ax_grid.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color("#555")

# ── Legend ────────────────────────────────────────────────────────────────────
ax_leg = fig.add_axes([0.62, 0.54, 0.32, 0.38])
ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1); ax_leg.axis("off")

ax_leg.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                  boxstyle="round,pad=0.03", lw=1.5, ec="#888",
                  fc="white", ls="--", transform=ax_leg.transAxes))
ax_leg.text(0.5, 0.93, "Legend", ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax_leg.transAxes)

leg_zoom = 0.038
for img, zoom, y, label in [
        (img_bunny,  leg_zoom, 0.65, "Rabbit (Agent)"),
        (img_carrot, leg_zoom*(REF_PX/img_carrot.shape[0]), 0.35, "Carrot (Goal)")]:
    ob = OffsetImage(img, zoom=zoom, interpolation="nearest")
    ax_leg.add_artist(AnnotationBbox(ob, (0.19, y),
                      xycoords="axes fraction", frameon=False))
    ax_leg.text(0.37, y, label, va="center", fontsize=10,
                transform=ax_leg.transAxes)

# ── Bottom diagram ────────────────────────────────────────────────────────────
ax_bot = fig.add_axes([0.03, 0.01, 0.94, 0.34])
ax_bot.set_xlim(0, 1); ax_bot.set_ylim(0, 1); ax_bot.axis("off")

ax_bot.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.92,
                  boxstyle="round,pad=0.02", lw=1.2, ec="#CCCCCC",
                  fc=BOX_BG, transform=ax_bot.transAxes))

ax_bot.text(0.02, 0.93, "Step 1: Relative Direction Calculation",
            fontsize=10, fontweight="bold", va="top",
            transform=ax_bot.transAxes, color=TEXT_DARK)

zbs = 0.060
zcs = zbs * (REF_PX / img_carrot.shape[0])

place(ax_bot, img_bunny,  zbs, 0.08, 0.57)
ax_bot.text(0.08, 0.82, "Rabbit Position", ha="center", fontsize=9,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.08, 0.30, f"({br}, {bc})", ha="center", fontsize=10,
            transform=ax_bot.transAxes, color=TEXT_DARK)

place(ax_bot, img_carrot, zcs, 0.26, 0.57)
ax_bot.text(0.26, 0.82, "Carrot Position", ha="center", fontsize=9,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.26, 0.30, f"({cr}, {cc})", ha="center", fontsize=10,
            transform=ax_bot.transAxes, color=TEXT_DARK)

ax_bot.annotate("", xy=(0.41, 0.57), xytext=(0.35, 0.57),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->, head_width=0.3",
                                color="#555", lw=1.5))

ax_bot.text(0.44, 0.80, "Row Difference", fontsize=9,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.44, 0.63, f"{cr} - {br} = {cr-br}  \u2192  sign = ", fontsize=9,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.658, 0.63, f"{CARROT_DIR[0]:+d}", fontsize=9, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_GREEN)

ax_bot.text(0.44, 0.44, "Column Difference", fontsize=9,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.44, 0.27, f"{cc} - {bc} = {cc-bc}  \u2192  sign = ", fontsize=9,
            transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.658, 0.27, f"{CARROT_DIR[1]:+d}", fontsize=9, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_GREEN)

ax_bot.annotate("", xy=(0.725, 0.57), xytext=(0.695, 0.57),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->, head_width=0.3",
                                color="#555", lw=1.5))

ax_bot.text(0.74, 0.93, "Step 2: State Encoding",
            fontsize=10, fontweight="bold", va="top",
            transform=ax_bot.transAxes, color=TEXT_DARK)

ax_bot.add_patch(FancyBboxPatch((0.735, 0.15), 0.245, 0.72,
                  boxstyle="round,pad=0.02", lw=1.0, ec="#AAAAAA",
                  fc="white", transform=ax_bot.transAxes))

ax_bot.text(0.858, 0.80, "Relative Direction (Row, Column)",
            ha="center", fontsize=8.5, transform=ax_bot.transAxes, color=TEXT_DARK)
ax_bot.text(0.858, 0.60, f"({CARROT_DIR[0]:+d}, {CARROT_DIR[1]:+d})",
            ha="center", fontsize=14, fontweight="bold",
            transform=ax_bot.transAxes, color=ACCENT_GREEN)

ax_bot.annotate("", xy=(0.858, 0.40), xytext=(0.858, 0.50),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->, head_width=0.3",
                                color="#555", lw=1.5))

ax_bot.text(0.858, 0.29, f"State = ({CARROT_DIR[0]}, {CARROT_DIR[1]})",
            ha="center", fontsize=11, fontweight="bold",
            transform=ax_bot.transAxes, color=TEXT_DARK)

fig.text(0.5, 0.005,
         " ",
         ha="center", va="bottom", fontsize=11, fontstyle="italic", color=TEXT_DARK)

out_path = os.path.join(OUTPUT_DIR, "figure 1.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
plt.show()