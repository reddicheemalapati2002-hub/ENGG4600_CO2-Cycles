"""
rq3_economics.py
────────────────
LCOE, NPV, and payback period for all three cycles at the three
Australian case study sites. Uses RQ1 efficiency results and
RQ2 capital costs as inputs.

Parameters
──────────
  Discount rate  : 7% real (Australian Treasury guidelines)
  Project life   : 20 years
  Capacity factor: 85% (cement/steel operate continuously)
  O&M            : 2.5% of capital per year
  Plant size     : 1 MW net (representative for Australian WHR)

Electricity prices (AEMO 2024/25 data, AUD/MWh):
  CS1: Gladstone QLD  — AUD 90/MWh (QLD spot average)
  CS2: Pilbara WA     — AUD 130/MWh (WA wholesale, isolated grid)
  CS3: Sydney NSW     — AUD 100/MWh (NSW spot average)

Monte Carlo: 10,000 samples varying:
  - Electricity price ±40%
  - Capital cost ±30%
  - Capacity factor 60-95%
  - Heat source temperature ±25°C
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Economic parameters ────────────────────────────────────────────────────
DISCOUNT_RATE  = 0.07   # 7% real
PROJECT_LIFE   = 20     # years
CAPACITY_FACTOR = 0.85
OM_FRACTION    = 0.025  # 2.5% of capital per year
PLANT_SIZE_KW  = 1000   # 1 MW net

# CRF = r(1+r)^n / ((1+r)^n - 1)
CRF = (DISCOUNT_RATE * (1+DISCOUNT_RATE)**PROJECT_LIFE) / \
      ((1+DISCOUNT_RATE)**PROJECT_LIFE - 1)

HOURS_PER_YEAR = 8760

# ── Case study parameters ──────────────────────────────────────────────────
CASE_STUDIES = {
    "CS2: Gas 200°C\n(Pilbara WA)": {
        "T_src":        200,
        "elec_price":   130,     # AUD/MWh — WA isolated grid premium
        "cap_cost": {
            "SRC":  4391,        # AUD/kW at 1 MW from RQ2
            "sCO₂": 2167,
            "tCO₂": 4429,
        },
        "eta_th": {
            "SRC":  24.45,       # % from RQ1 parametric sweep
            "sCO₂": 14.54,
            "tCO₂": 24.70,
        },
    },
    "CS1: Cement 350°C\n(Gladstone QLD)": {
        "T_src":        350,
        "elec_price":   90,
        "cap_cost": {
            "SRC":  3683,
            "sCO₂": 2565,
            "tCO₂": 3584,
        },
        "eta_th": {
            "SRC":  32.13,
            "sCO₂": 31.90,
            "tCO₂": 34.84,
        },
    },
    "CS3: Steel 500°C\n(Sydney NSW)": {
        "T_src":        500,
        "elec_price":   100,
        "cap_cost": {
            "SRC":   822,
            "sCO₂": 2617,
            "tCO₂": 3491,
        },
        "eta_th": {
            "SRC":  34.06,
            "sCO₂": 38.62,
            "tCO₂": 38.70,
        },
    },
}
CYCLES = ["SRC", "sCO₂", "tCO₂"]


def compute_lcoe_npv(cap_cost_aud_kw, eta_th_pct, elec_price_aud_mwh,
                     plant_kw=PLANT_SIZE_KW):
    """
    Returns LCOE (AUD/MWh), NPV (AUD), and payback period (years).
    Assumes waste heat is free (WHR application — no fuel cost).
    """
    cap_total   = cap_cost_aud_kw * plant_kw          # AUD
    om_annual   = OM_FRACTION * cap_total              # AUD/yr
    e_gen_mwh   = plant_kw * HOURS_PER_YEAR * CAPACITY_FACTOR / 1000  # MWh/yr
    revenue_yr  = e_gen_mwh * elec_price_aud_mwh       # AUD/yr
    net_revenue = revenue_yr - om_annual               # AUD/yr

    # LCOE (no fuel cost for WHR)
    lcoe = (cap_total * CRF + om_annual) / e_gen_mwh   # AUD/MWh

    # NPV: sum of discounted net revenues minus initial capital
    npv = sum(net_revenue / (1+DISCOUNT_RATE)**t
              for t in range(1, PROJECT_LIFE+1)) - cap_total

    # Simple payback (undiscounted)
    payback = cap_total / net_revenue if net_revenue > 0 else float("inf")

    return lcoe, npv, payback


def run_deterministic():
    """Deterministic LCOE/NPV at baseline parameters."""
    print("\n" + "="*76)
    print(f"RQ3 — LCOE AND NPV  (deterministic, {PLANT_SIZE_KW} kW plant)")
    print(f"  Discount rate: {DISCOUNT_RATE*100:.0f}%  |  "
          f"Life: {PROJECT_LIFE} yr  |  CF: {CAPACITY_FACTOR*100:.0f}%  |  "
          f"CRF: {CRF:.4f}")
    print("="*76)

    results = {}
    for cs_label, cs in CASE_STUDIES.items():
        print(f"\n  {cs_label.replace(chr(10), ' ')}")
        print(f"  Electricity price: AUD {cs['elec_price']}/MWh")
        print(f"  {'─'*60}")
        print(f"  {'Cycle':>6}  {'Cap(AUD/kW)':>12}  {'LCOE':>12}  "
              f"{'NPV':>14}  {'Payback':>9}  {'Viable'}")
        print(f"  {'':>6}  {'':>12}  {'(AUD/MWh)':>12}  "
              f"{'(kAUD)':>14}  {'(yr)':>9}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*14}  {'─'*9}  {'─'*8}")
        results[cs_label] = {}
        for cycle in CYCLES:
            lcoe, npv, pb = compute_lcoe_npv(
                cs["cap_cost"][cycle],
                cs["eta_th"][cycle],
                cs["elec_price"],
            )
            viable = "✓ YES" if npv > 0 else "✗ NO"
            pb_str = f"{pb:.1f}" if pb < 50 else ">50"
            print(f"  {cycle:>6}  {cs['cap_cost'][cycle]:>12,}  "
                  f"{lcoe:>12.1f}  {npv/1000:>14.0f}  "
                  f"{pb_str:>9}  {viable}")
            results[cs_label][cycle] = {
                "lcoe": lcoe, "npv": npv, "payback": pb,
                "cap_cost": cs["cap_cost"][cycle],
            }
    return results


def run_monte_carlo(n_samples=10000):
    """
    Monte Carlo uncertainty analysis.
    Returns distribution statistics for LCOE and NPV.
    """
    print(f"\n{'='*60}")
    print(f"RQ3 — MONTE CARLO  (n={n_samples:,})")
    print(f"{'='*60}")

    np.random.seed(42)
    mc_results = {}

    for cs_label, cs in CASE_STUDIES.items():
        mc_results[cs_label] = {}
        for cycle in CYCLES:
            base_cap   = cs["cap_cost"][cycle]
            base_elec  = cs["elec_price"]
            base_T     = cs["T_src"]

            # ── Sample uncertain parameters ───────────────────────────
            cap_samples  = base_cap  * np.random.uniform(0.70, 1.30, n_samples)
            elec_samples = base_elec * np.random.uniform(0.60, 1.40, n_samples)
            cf_samples   = np.random.uniform(0.60, 0.95, n_samples)

            lcoe_s, npv_s = [], []
            for cap, elec, cf in zip(cap_samples, elec_samples, cf_samples):
                cap_t   = cap * PLANT_SIZE_KW
                om_a    = OM_FRACTION * cap_t
                e_gen   = PLANT_SIZE_KW * HOURS_PER_YEAR * cf / 1000
                rev_yr  = e_gen * elec
                net_rev = rev_yr - om_a
                lc = (cap_t * CRF + om_a) / e_gen
                npv_val = sum(net_rev/(1+DISCOUNT_RATE)**t
                              for t in range(1, PROJECT_LIFE+1)) - cap_t
                lcoe_s.append(lc)
                npv_s.append(npv_val)

            lcoe_arr = np.array(lcoe_s)
            npv_arr  = np.array(npv_s)
            p_viable = np.mean(npv_arr > 0) * 100

            mc_results[cs_label][cycle] = {
                "lcoe_p10": np.percentile(lcoe_arr, 10),
                "lcoe_p50": np.percentile(lcoe_arr, 50),
                "lcoe_p90": np.percentile(lcoe_arr, 90),
                "npv_p10":  np.percentile(npv_arr, 10),
                "npv_p50":  np.percentile(npv_arr, 50),
                "npv_p90":  np.percentile(npv_arr, 90),
                "p_viable": p_viable,
            }

            cs_short = cs_label.split("\n")[0]
            print(f"\n  {cs_short} — {cycle}")
            print(f"    LCOE P10/P50/P90: "
                  f"{mc_results[cs_label][cycle]['lcoe_p10']:.0f} / "
                  f"{mc_results[cs_label][cycle]['lcoe_p50']:.0f} / "
                  f"{mc_results[cs_label][cycle]['lcoe_p90']:.0f}  AUD/MWh")
            print(f"    NPV  P10/P50/P90: "
                  f"{mc_results[cs_label][cycle]['npv_p10']/1000:.0f} / "
                  f"{mc_results[cs_label][cycle]['npv_p50']/1000:.0f} / "
                  f"{mc_results[cs_label][cycle]['npv_p90']/1000:.0f}  kAUD")
            print(f"    P(NPV > 0):       {p_viable:.0f}%")

    return mc_results


def plot_results(det_results, mc_results):
    """Four-panel summary figure for RQ3."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {"SRC": "#51247A", "sCO₂": "#0064B4", "tCO₂": "#BB9B1C"}
    cs_labels_short = [k.split("\n")[0] for k in CASE_STUDIES.keys()]
    cs_keys = list(CASE_STUDIES.keys())
    x = np.arange(len(cs_keys))
    width = 0.25

    # ── Panel 1: Deterministic LCOE ───────────────────────────────────
    ax = axes[0, 0]
    for i, cycle in enumerate(CYCLES):
        lcoes = [det_results[cs][cycle]["lcoe"] for cs in cs_keys]
        bars = ax.bar(x + i*width, lcoes, width,
                      color=colors[cycle], label=cycle, alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{bar.get_height():.0f}",
                    ha="center", va="bottom", fontsize=7.5)
    for i, (cs_key, cs) in enumerate(CASE_STUDIES.items()):
        ax.axhline(cs["elec_price"], color="red", linestyle="--",
                   alpha=0.4, linewidth=1)
    ax.set_xticks(x + width)
    ax.set_xticklabels(cs_labels_short, fontsize=9)
    ax.set_ylabel("LCOE (AUD/MWh)", fontsize=10)
    ax.set_title("Deterministic LCOE\n(red dashes = site electricity price)",
                 fontsize=9)
    ax.legend(fontsize=8.5); ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: Deterministic NPV ────────────────────────────────────
    ax = axes[0, 1]
    for i, cycle in enumerate(CYCLES):
        npvs = [det_results[cs][cycle]["npv"]/1000 for cs in cs_keys]
        bars = ax.bar(x + i*width, npvs, width,
                      color=colors[cycle], label=cycle, alpha=0.88)
        for bar in bars:
            va = "bottom" if bar.get_height() >= 0 else "top"
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + (20 if bar.get_height() >= 0 else -20),
                    f"{bar.get_height():.0f}",
                    ha="center", va=va, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(cs_labels_short, fontsize=9)
    ax.set_ylabel("NPV (kAUD, 1 MW plant)", fontsize=10)
    ax.set_title("Deterministic NPV\n(above zero = viable investment)",
                 fontsize=9)
    ax.legend(fontsize=8.5); ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: Monte Carlo P(NPV > 0) ──────────────────────────────
    ax = axes[1, 0]
    for i, cycle in enumerate(CYCLES):
        pv = [mc_results[cs][cycle]["p_viable"] for cs in cs_keys]
        bars = ax.bar(x + i*width, pv, width,
                      color=colors[cycle], label=cycle, alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.0f}%",
                    ha="center", va="bottom", fontsize=7.5)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(cs_labels_short, fontsize=9)
    ax.set_ylabel("P(NPV > 0)  (%)", fontsize=10)
    ax.set_title("Monte Carlo: Probability of Positive NPV\n"
                 "10,000 samples — capital ±30%, price ±40%, CF 60–95%",
                 fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8.5); ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: LCOE P10/P50/P90 for best cycle per site ────────────
    ax = axes[1, 1]
    best_cycles = []
    for cs_key, cs in CASE_STUDIES.items():
        best = min(CYCLES, key=lambda c: det_results[cs_key][c]["lcoe"])
        best_cycles.append(best)
    for j, (cs_key, best) in enumerate(zip(cs_keys, best_cycles)):
        mc = mc_results[cs_key][best]
        p10, p50, p90 = mc["lcoe_p10"], mc["lcoe_p50"], mc["lcoe_p90"]
        elec = CASE_STUDIES[cs_key]["elec_price"]
        ax.barh(j, p90-p10, left=p10, height=0.4,
                color=colors[best], alpha=0.5,
                label=f"{cs_labels_short[j]}: {best}")
        ax.plot([p50], [j], "o", color=colors[best], markersize=8)
        ax.axvline(elec, color="red", linestyle="--", alpha=0.4)
    ax.set_yticks(range(len(cs_keys)))
    ax.set_yticklabels([f"{s}\n(best: {b})"
                        for s, b in zip(cs_labels_short, best_cycles)],
                       fontsize=9)
    ax.set_xlabel("LCOE (AUD/MWh)", fontsize=10)
    ax.set_title("Best Cycle LCOE Range  (P10–P90)\n"
                 "Dot = median  |  Red dashes = site electricity price",
                 fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle(
        f"RQ3 — Techno-Economic Analysis: Australian WHR Sites  |  "
        f"{PLANT_SIZE_KW} kW plant  |  2026 AUD\n"
        f"Discount: {DISCOUNT_RATE*100:.0f}%  |  Life: {PROJECT_LIFE} yr  |  "
        f"CF: {CAPACITY_FACTOR*100:.0f}%  |  O&M: {OM_FRACTION*100:.1f}% capex/yr",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("RQ3_LCOE_NPV_MONTE_CARLO.png", dpi=150)
    plt.show()
    print("Saved: RQ3_LCOE_NPV_MONTE_CARLO.png")


if __name__ == "__main__":
    det  = run_deterministic()
    mc   = run_monte_carlo(n_samples=10000)
    plot_results(det, mc)
