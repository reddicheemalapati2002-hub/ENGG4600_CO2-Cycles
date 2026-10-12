"""RQ3 — techno-economics on a fixed-source basis, with Monte Carlo."""
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import save, COLORS, MARKERS, LABELS, SITE_SHORT
from co2whr.costs import tec_breakdown
from co2whr.economics import cashflow_metrics, monte_carlo, hx_cost_reduction_to_match
from co2whr.sites import SITES, SITE_ORDER, CYCLES


def run(designs, fig_dir):
    print("RQ3 ...")
    out = {}
    for sk in SITE_ORDER:
        s = SITES[sk]; out[sk] = {}
        for cyc in CYCLES:
            d = designs[sk][cyc]; W = d["W_site_kW"]
            b = tec_breakdown(d, W)
            m = cashflow_metrics(b["TEC_kAUD"], W, s["price"])
            mc = monte_carlo(b["TEC_kAUD"], W, s["price"])
            m.pop("TEC_kAUD"); m.pop("W_kW")
            out[sk][cyc] = dict(W_kW=W, TEC_kAUD=b["TEC_kAUD"], TEC_per_kW=b["TEC_per_kW"], items=b["items"], **m, mc=mc,
                                T_gas_out=d["T_gas_out"], eta=d["c"]["eta_th"] * 100, Q_recovered_kW=d["Q_recovered_kW"])
        best_lcoe = min(CYCLES, key=lambda c: out[sk][c]["LCOE"]); best_npv = max(CYCLES, key=lambda c: out[sk][c]["NPV_kAUD"])
        out[sk]["best_lcoe"] = best_lcoe; out[sk]["best_npv"] = best_npv
        # heat-exchanger cost reduction needed for each non-best cycle to match the best LCOE
        for cyc in CYCLES:
            if cyc == best_lcoe:
                out[sk][cyc]["hx_reduction_to_match"] = 0.0; continue
            # LCOE-matching TEC for this cycle: scale target by generation ratio
            target_tec = out[sk][best_lcoe]["TEC_kAUD"] * out[sk][cyc]["E_MWh"] / out[sk][best_lcoe]["E_MWh"]
            out[sk][cyc]["hx_reduction_to_match"] = hx_cost_reduction_to_match(out[sk][cyc]["items"], target_tec)
    # ---- figures ----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5)); x = np.arange(3); w = 0.26
    a = axes[0, 0]
    for i, cyc in enumerate(CYCLES):
        v = [out[sk][cyc]["W_kW"] / 1e3 for sk in SITE_ORDER]; b = a.bar(x + (i - 1) * w, v, w, color=COLORS[cyc], label=LABELS[cyc])
        for bb, vv in zip(b, v): a.text(bb.get_x() + w / 2, vv + 0.04, f"{vv:.2f}", ha="center", fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a.set_ylabel("Net electrical output (MWe)"); a.legend()
    a.set_title("Net power recovered from each site's waste-gas stream")
    a = axes[0, 1]
    for i, cyc in enumerate(CYCLES):
        v = [out[sk][cyc]["LCOE"] for sk in SITE_ORDER]; b = a.bar(x + (i - 1) * w, v, w, color=COLORS[cyc], label=LABELS[cyc])
        for bb, vv in zip(b, v): a.text(bb.get_x() + w / 2, vv + 2, f"{vv:.0f}", ha="center", fontsize=7.5)
    for i, sk in enumerate(SITE_ORDER): a.hlines(SITES[sk]["price"], i - 0.45, i + 0.45, color="#CC0000", ls="--", lw=1.2)
    a.text(2.45, SITES["CS3"]["price"] + 3, "site electricity price", color="#CC0000", fontsize=7.5, ha="right")
    a.set_xticks(x); a.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a.set_ylabel("LCOE (AUD/MWh)"); a.legend(); a.set_title("Levelised cost of electricity (7 % real, 20 yr, CF 85 %)")
    a = axes[1, 0]
    for i, cyc in enumerate(CYCLES):
        v = [out[sk][cyc]["NPV_kAUD"] / 1e3 for sk in SITE_ORDER]; b = a.bar(x + (i - 1) * w, v, w, color=COLORS[cyc], label=LABELS[cyc])
        for bb, vv in zip(b, v): a.text(bb.get_x() + w / 2, vv + 0.2 if vv >= 0 else vv - 0.25, f"{vv:.1f}", ha="center", va="bottom" if vv >= 0 else "top", fontsize=7.5)
    a.axhline(0, color="k", lw=0.8); a.set_ylim(-4.5, 14); a.set_xticks(x); a.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a.set_ylabel("NPV (MAUD)"); a.legend(); a.set_title("Net present value of the site project")
    a = axes[1, 1]
    for i, cyc in enumerate(CYCLES):
        v = [out[sk][cyc]["mc"]["p_viable"] for sk in SITE_ORDER]; b = a.bar(x + (i - 1) * w, v, w, color=COLORS[cyc], label=LABELS[cyc])
        for bb, vv in zip(b, v): a.text(bb.get_x() + w / 2, vv + 1, f"{vv:.0f} %", ha="center", fontsize=7.5)
    a.axhline(50, color="grey", ls=":"); a.set_ylim(0, 112); a.set_xticks(x); a.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a.set_ylabel("P(NPV > 0) (%)"); a.legend(loc="lower right")
    a.set_title("Monte Carlo viability — 10,000 samples\ncapital ±30 %, price ±40 %, capacity factor 60–95 %")
    fig.suptitle("Techno-economic results on a fixed-source basis (each cycle recovers what it can from the same stream; 2026 AUD)", fontsize=10)
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ3_LCOE_NPV_MONTE_CARLO.png")
    return out
