"""RQ4 — part-load performance and flexibility scoring."""
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import save, COLORS, MARKERS, LABELS
from co2whr.offdesign import sweep, marchionni, LOADS
from co2whr.sites import SITES, SITE_ORDER, CYCLES


def run(fig_dir):
    print("RQ4 ...")
    res = sweep(); sc = marchionni(res)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9)); axes = axes.flatten()
    for a, sk in zip(axes[:3], SITE_ORDER):
        for cyc in CYCLES:
            a.plot([f * 100 for f in LOADS], res[sk][cyc]["retention"], color=COLORS[cyc], marker=MARKERS[cyc], label=LABELS[cyc])
        a.axhline(1.0, color="grey", lw=0.8); a.axhline(0.9, color="grey", ls="--", lw=0.8); a.text(22, 0.905, "90 % retention", fontsize=7.5, color="grey")
        a.set_xlim(15, 105); a.set_ylim(0.4, 1.12); a.set_xlabel("Load (% of design mass flow)"); a.set_ylabel("Efficiency retention η / η_design")
        a.set_title(f"{SITES[sk]['name']}  (design η: " + ", ".join(f"{LABELS[c]} {res[sk][c]['eta'][0]:.1f} %" for c in CYCLES) + ")", fontsize=8.5); a.legend(fontsize=8)
    a = axes[3]; crit = ["Turndown", "η at 60 %", "η at 40 %", "Start-up", "TOTAL"]; x = np.arange(5); w = 0.26
    for i, cyc in enumerate(CYCLES):
        v = [sc[cyc]["turndown"], sc[cyc]["r60"], sc[cyc]["r40"], sc[cyc]["startup"], sc[cyc]["total"]]
        b = a.bar(x + (i - 1) * w, v, w, color=COLORS[cyc], label=LABELS[cyc])
        for bb, vv in zip(b, v): a.text(bb.get_x() + w / 2, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=7)
    a.set_xticks(x); a.set_xticklabels(crit); a.set_ylim(0, 1.15); a.set_ylabel("Score (0–1)"); a.legend(); a.set_title("Marchionni flexibility scoring (weights 0.30 / 0.30 / 0.25 / 0.15)", fontsize=9)
    fig.suptitle("Part-load performance: Stodola correction η_od = η₀[1 − 0.3(1−f)²] with sliding-pressure control", fontsize=10)
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ4_OFFDESIGN_PARTLOAD.png")
    return dict(loads=LOADS, sweep=res, scores=sc)
