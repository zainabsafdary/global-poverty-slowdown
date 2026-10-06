"""
Makes three PNG charts for the website from the poverty analysis.
Run from the folder containing complete_series.csv and poverty_stall_analysis.py:
    python3 make_charts.py
Needs: pip install matplotlib (plus pandas and numpy)
"""
import contextlib, io
import matplotlib.pyplot as plt

# Run the analysis quietly and reuse its results (reg, world, share, sp)
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("poverty_stall_analysis.py").read())

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#9aa5b1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white"})
SOURCE = ("Source: World Bank Poverty and Inequality Platform via Our World in Data. "
          "Extreme poverty = below $3.00/day (2021 PPP).")

def titles(fig, title, subtitle):
    fig.text(0.0, 1.06, title, fontsize=17, fontweight="bold", ha="left")
    fig.text(0.0, 1.0, subtitle, fontsize=12, color=MUTED, ha="left")
    fig.text(0.0, -0.06, SOURCE, fontsize=8.5, color=MUTED, ha="left")

# --- Chart 1: world extreme poverty rate -----------------------------------
w = world[(world.year >= 1990) & (world.year <= 2025)].sort_values("year")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(w.year, w.headcount_ratio, color=BLUE, lw=3)
for y in (1990, 2015, 2025):
    v = w.set_index("year").headcount_ratio[y]
    ax.scatter(y, v, color=BLUE, zorder=3, s=40)
    ax.annotate(f"{v:.0f}%", (y, v), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=12, fontweight="bold", color=BLUE)
ax.axvspan(2015, 2025, color=GREY, alpha=0.12, lw=0)
ax.text(2020, 36, "Since 2015:\n−0.35 points/yr", ha="center", color=INK, fontsize=11)
ax.text(2002.5, 12, "1990–2015:\n−1.2 points/yr", ha="center", color=INK, fontsize=11)
ax.set_ylim(0, 50); ax.set_xlim(1989, 2026)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
titles(fig, "The fall in extreme poverty",
       "Share of the world's population living in extreme poverty from 1990 to 2025")
fig.savefig("chart1_world_poverty.png"); plt.close(fig)

# --- Chart 2: Sub-Saharan Africa's share of the world's poor ---------------
s = share.loc[1990:2025]
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(s.index, s.values, color=ORANGE, alpha=0.15, lw=0)
ax.plot(s.index, s.values, color=ORANGE, lw=3)
for y in (1990, 2015, 2025):
    ax.scatter(y, s[y], color=ORANGE, zorder=3, s=40)
    ax.annotate(f"{s[y]:.0f}%", (y, s[y]), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=12, fontweight="bold", color=ORANGE)
ax.set_ylim(0, 100); ax.set_xlim(1989, 2026)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
titles(fig, "Concentrated of Extreme Poverty in Sub-Saharan Africa",
       "Sub-Saharan Africa's share of all people in extreme poverty worldwide")
fig.savefig("chart2_ssa_share.png"); plt.close(fig)

# --- Chart 3: growth vs redistribution effects -----------------------------
rel = sp[sp.H0 >= 5]
groups = [("Sub-Saharan\nAfrica", True, False), ("Sub-Saharan\nAfrica", True, True),
          ("Rest of\nworld", False, False), ("Rest of\nworld", False, True)]
labels, gro, red = [], [], []
for name, is_ssa, late in groups:
    sub = rel[(rel.ssa == is_ssa) & ((rel.y1 > 2015) == late)]
    labels.append(f"{name}\n{'after 2015' if late else 'up to 2015'}")
    gro.append(sub.growth.sum() / sub.yrs.sum())
    red.append(sub.redist.sum() / sub.yrs.sum())
x = range(len(labels)); bw = 0.36
fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar([i - bw/2 for i in x], gro, bw, color=BLUE, label="Income growth")
b2 = ax.bar([i + bw/2 for i in x], red, bw, color=GREY, label="Change in inequality")
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.annotate(f"{v:+.2f}", (b.get_x() + b.get_width()/2, v),
                    xytext=(0, 5 if v >= 0 else -14), textcoords="offset points",
                    ha="center", fontsize=10, color=INK)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("Change in poverty rate (points per year)")
ax.set_ylim(-1.15, 0.45)
ax.legend(frameon=False, loc="lower right")
ax.text(1 - bw/2, 0.33, "Growth stalled:\npoverty pushed up", ha="center",
        fontsize=10, color=BLUE)
titles(fig, "Income Growth",
       "In Africa, income growth stopped reducing poverty after 2015.")
fig.savefig("chart3_growth_vs_inequality.png"); plt.close(fig)

print("Saved chart1_world_poverty.png, chart2_ssa_share.png, chart3_growth_vs_inequality.png")
