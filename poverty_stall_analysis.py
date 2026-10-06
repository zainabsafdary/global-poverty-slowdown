"""
Why has the fall in global extreme poverty stalled since ~2015?
Data: World Bank PIP via Our World in Data (complete_series.csv)
Convention throughout: 2021 PPPs, $3.00/day international poverty line.
"""
import numpy as np
import pandas as pd

raw = pd.read_csv("complete_series.csv", low_memory=False)
df = raw[raw.ppp_version == 2021]
# Inequality measures are PPP-invariant: their rows have no ppp_version
ineq = raw[raw.ppp_version.isna() & raw.gini.notna()]
LINE = "300"   # poverty lines are stored in cents
Z = 3.00

# ---------------------------------------------------------------------------
# STEP 1. Regional decomposition of the change in the world poverty rate
# ---------------------------------------------------------------------------
# World rate = sum over regions of (population share x regional poverty rate).
# The change is split with a Shapley decomposition into
#   (a) changes in regional poverty rates ("within"), and
#   (b) changes in where people live ("population shift").
REGIONS = ["East Asia and Pacific (WB)", "Europe and Central Asia (WB)",
           "Latin America and Caribbean (WB)",
           "Middle East, North Africa, Afghanistan and Pakistan (WB)",
           "North America (WB)", "South Asia (WB)", "Sub-Saharan Africa (WB)"]

reg = df[df.country.isin(REGIONS) & (df.poverty_line == LINE) & df.decile.isna()]
reg = reg[["country", "year", "headcount", "headcount_ratio"]].copy()
reg["pop"] = reg.headcount / (reg.headcount_ratio / 100)
reg["s"] = reg["pop"] / reg.groupby("year")["pop"].transform("sum")
reg["p"] = reg.headcount_ratio

world = df[(df.country == "World") & (df.poverty_line == LINE) & df.decile.isna()]
check = reg.groupby("year").headcount.sum() / world.set_index("year").headcount
print(f"Regions sum to world: ratio {check.min():.3f}-{check.max():.3f}\n")

def decompose(y0, y1):
    a = reg[reg.year == y0].set_index("country")
    b = reg[reg.year == y1].set_index("country")
    within = ((b.p - a.p) * (a.s + b.s) / 2)   # Shapley: average weights
    shift = ((b.s - a.s) * (a.p + b.p) / 2)
    out = pd.DataFrame({"within": within, "pop_shift": shift})
    out["total"] = out.sum(axis=1)
    return out, (a.s * a.p).sum(), (b.s * b.p).sum()

for y0, y1 in [(1990, 2015), (2015, 2025)]:
    out, p0, p1 = decompose(y0, y1)
    yrs = y1 - y0
    print(f"=== {y0}-{y1}: world rate {p0:.1f}% -> {p1:.1f}% "
          f"({(p1 - p0) / yrs:+.2f} pts/yr) ===")
    out.index = out.index.str.replace(" (WB)", "")
    print((out / yrs).round(3).rename(columns=lambda c: c + "/yr").to_string())
    print(f"Total population-shift effect: {out.pop_shift.sum():+.2f} pts\n")

# Share of the world's extreme poor living in Sub-Saharan Africa
ssa = reg[reg.country == "Sub-Saharan Africa (WB)"].set_index("year").headcount
share = (ssa / reg.groupby("year").headcount.sum() * 100).round(1)
print("SSA share of world's extreme poor:",
      share.loc[[1990, 2000, 2010, 2015, 2020, 2025]].to_dict(), "\n")

# ---------------------------------------------------------------------------
# STEP 2. Growth vs redistribution (Datt-Ravallion), country survey spells
# ---------------------------------------------------------------------------
# Rebuild each survey's distribution from its decile thresholds, as a
# piecewise-linear quantile function Q(p). Headcount H(z) = share with Q(p) < z.
# Growth effect: scale year-0 distribution to year-1 mean.
# Redistribution effect: year-1 shape at year-0 mean. Shapley average of both.
SSA = ["Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
       "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo",
       "Cote d'Ivoire", "Democratic Republic of Congo", "Equatorial Guinea",
       "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
       "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Madagascar", "Malawi",
       "Mali", "Mauritania", "Mauritius", "Mozambique", "Namibia", "Niger",
       "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles",
       "Sierra Leone", "South Africa", "South Sudan", "Sudan", "Tanzania",
       "Togo", "Uganda", "Zambia", "Zimbabwe"]

ctry = df[~df.country.str.contains(r"\(|World", regex=True)]
dec = ctry[ctry.decile.notna()].pivot_table(
    index=["country", "year", "welfare_type"], columns="decile",
    values=["thr", "avg"])
dec.columns = [f"{v}{int(d)}" for v, d in dec.columns]
head = ctry[ctry["mean"].notna()].set_index(
    ["country", "year", "welfare_type"])[["mean", "survey_comparability"]]
pov = ctry[(ctry.poverty_line == LINE) & ctry.decile.isna()].set_index(
    ["country", "year", "welfare_type"])["headcount_ratio"]
gini = ineq.set_index(
    ["country", "year", "welfare_type"])["gini"]
surv = head.join(pov).join(gini).join(dec, how="inner")

def quantile_points(row):
    thr = np.array([row[f"thr{d}"] for d in range(1, 10)])
    a1, a10 = row["avg1"], row["avg10"]
    q0 = max(0.0, 2 * a1 - thr[0])          # bottom decile mean matches avg1
    q1 = 2 * a10 - thr[-1]                  # top decile mean matches avg10
    return np.r_[0, np.arange(0.1, 1.0, 0.1), 1], np.r_[q0, thr, q1]

def H(p, q, z):
    """Share of population below z; Q is increasing so invert by interpolation."""
    if z <= q[0]:
        return 0.0
    if z >= q[-1]:
        return 100.0
    return float(np.interp(z, q, p) * 100)

recs = []
for (c, wt), g in surv.reset_index().groupby(["country", "welfare_type"]):
    g = g.sort_values("year")
    for (_, r0), (_, r1) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
        if r0.survey_comparability != r1.survey_comparability:
            continue                        # never compare across a break
        p0, q0 = quantile_points(r0)
        p1, q1 = quantile_points(r1)
        m0, m1 = r0["mean"], r1["mean"]
        H00, H11 = H(p0, q0, Z), H(p1, q1, Z)
        H10 = H(p0, q0 * m1 / m0, Z)       # year-0 shape, year-1 mean
        H01 = H(p1, q1 * m0 / m1, Z)       # year-1 shape, year-0 mean
        growth = 0.5 * ((H10 - H00) + (H11 - H01))
        redist = 0.5 * ((H01 - H00) + (H11 - H10))
        recs.append(dict(country=c, welfare=wt, y0=r0.year, y1=r1.year,
                         H0=r0.headcount_ratio, H1=r1.headcount_ratio,
                         H0_rebuilt=H00, mean0=m0, mean1=m1, gini0=r0.gini,
                         growth=growth, redist=redist,
                         ssa=c in SSA))
sp = pd.DataFrame(recs)
sp["yrs"] = sp.y1 - sp.y0

err = (sp.H0_rebuilt - sp.H0).abs()
print(f"Rebuilt vs published headcount: median abs error {err.median():.2f} pts, "
      f"90th pct {err.quantile(.9):.2f} pts (n={len(sp)} spells)\n")

# Focus on spells where extreme poverty actually matters (H0 >= 5%)
rel = sp[sp.H0 >= 5]
for label, sub in [("Sub-Saharan Africa", rel[rel.ssa]),
                   ("Rest of world", rel[~rel.ssa])]:
    for period, s in [("ends <=2015", sub[sub.y1 <= 2015]),
                      ("ends >2015", sub[sub.y1 > 2015])]:
        if len(s) == 0:
            continue
        yrs = s.yrs.sum()
        print(f"{label:20s} {period:11s} n={len(s):3d}  "
              f"growth {s.growth.sum() / yrs:+.2f}  "
              f"redistribution {s.redist.sum() / yrs:+.2f}  pts/yr")
print()

# ---------------------------------------------------------------------------
# STEP 3. Growth elasticity of poverty
# ---------------------------------------------------------------------------
# How much does the poverty rate fall (in %) for each 1% of mean growth,
# and is the payoff smaller where inequality or poverty starts high?
def wls(y, X, w, names):
    """Weighted least squares with HC1 robust standard errors."""
    X = np.column_stack([np.ones(len(y))] + X)
    W = np.sqrt(w)
    Xw, yw = X * W[:, None], y * W
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    b = XtX_inv @ Xw.T @ yw
    u = yw - Xw @ b
    n, k = Xw.shape
    V = XtX_inv @ (Xw.T * u**2) @ Xw @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"coef": b, "se": se, "t": b / se},
                        index=["const"] + names).round(3)

e = sp[(sp.H0 >= 5) & (sp.H1 > 0.5)].copy()
e["dlnH"] = np.log(e.H1 / e.H0) / e.yrs
e["dlnM"] = np.log(e.mean1 / e.mean0) / e.yrs
e = e[e.gini0.notna()]
e["gini_c"] = e.gini0 - e.gini0.mean()     # centred, so dlnM = average
e["H0_c"] = e.H0 - e.H0.mean()
y, w, g = e.dlnH.values, e.yrs.values, e.dlnM.values
print(f"n = {len(e)} spells, mean initial Gini {e.gini0.mean():.2f}, "
      f"mean initial H {e.H0.mean():.0f}%")
print("\n--- Simple ---")
print(wls(y, [g], w, ["dlnM"]))
print("\n--- + inequality & initial poverty ---")
print(wls(y, [g, g * e.gini_c.values, g * e.H0_c.values], w,
          ["dlnM", "dlnM x Gini", "dlnM x H0"]))
print("\n--- + SSA interaction ---")
print(wls(y, [g, g * e.ssa.values], w, ["dlnM", "dlnM x SSA"]))

# Growth of the mean: how fast have the poor countries been growing?
print("\nAnnual growth in survey mean, median spell:")
for label, s in [("SSA", e[e.ssa]), ("Rest", e[~e.ssa])]:
    print(f"  {label}: <=2015 {s[s.y1 <= 2015].dlnM.median()*100:+.1f}%  "
          f">2015 {s[s.y1 > 2015].dlnM.median()*100:+.1f}%")
