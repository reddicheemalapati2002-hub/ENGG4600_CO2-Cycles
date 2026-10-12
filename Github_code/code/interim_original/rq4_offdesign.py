"""
rq4_offdesign.py
────────────────
Off-design and part-load performance for all three cycles using
Stodola ellipse law for turbine off-design and sliding pressure
control for CO2 cycles.

Evaluates each cycle at 100%, 80%, 60%, 40%, 20% of design
mass flow, recording how efficiency and net power degrade.

Stodola ellipse law:
  (m_dot / m_dot_0)^2 = (P_in^2 - P_out^2) / (P_in0^2 - P_out0^2)
  Used here in simplified form as efficiency correction:
  η_od = η_0 × f_load  where f_load accounts for incidence losses

Flexibility scoring (Marchionni et al. 2020):
  Score = Σ w_i × s_i  across: turndown ratio, ramp rate, 
  part-load efficiency retention, startup complexity
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from tco2_model import tco2_rankine_cycle

# ── Design point parameters ────────────────────────────────────────────────
T_CRIT_CO2  = CP.PropsSI("Tcrit", "CO2")
ETA_T_0     = 0.85
ETA_P_0     = 0.80
LOAD_LEVELS = [1.0, 0.80, 0.60, 0.40, 0.20]

CASE_STUDIES = [
    ("CS2: Gas 200°C",    200, 12.8e6, 5e6,  0.70),
    ("CS1: Cement 350°C", 350, 20.0e6, 5e6,  0.70),
    ("CS3: Steel 500°C",  500, 20.0e6, 5e6,  0.70),
]


def stodola_eta_correction(load_fraction):
    """
    Turbine isentropic efficiency correction for off-design operation.
    Simplified Stodola/Wilson model with incidence loss correction.
    η_od/η_0 = 1 - 0.3*(1-f)^2  (parabolic, typical radial turbine)
    At f=0.4: η_od/η_0 ≈ 0.892 (10.8% degradation)
    At f=0.2: η_od/η_0 ≈ 0.808 (19.2% degradation)
    """
    f = load_fraction
    return 1.0 - 0.3 * (1.0 - f)**2


def src_offdesign(T_src_C, load_fraction):
    """SRC part-load via sliding pressure control."""
    eta_t = ETA_T_0 * stodola_eta_correction(load_fraction)
    eta_p = ETA_P_0
    fluid = "Water"
    T_K   = T_src_C + 273.15
    T_CRIT_W = CP.PropsSI("Tcrit", fluid)

    P_hi0 = min(10e6, 0.95 * CP.PropsSI("P","T",T_K,"Q",1,fluid)) \
            if T_K < T_CRIT_W else 20e6
    P_hi  = P_hi0 * load_fraction         # sliding pressure
    P_lo  = 10000

    try:
        h3 = CP.PropsSI("H","T",T_K,"P",P_hi,fluid)
        s3 = CP.PropsSI("S","T",T_K,"P",P_hi,fluid)
        h4s = CP.PropsSI("H","P",P_lo,"S",s3,fluid)
        h1  = CP.PropsSI("H","P",P_lo,"Q",0,fluid)
        s1  = CP.PropsSI("S","P",P_lo,"Q",0,fluid)
        h2s = CP.PropsSI("H","P",P_hi,"S",s1,fluid)
        wp  = (h2s - h1) / eta_p / 1000
        wt  = (h3 - h4s) * eta_t / 1000
        qin = (h3 - (h1 + wp*1000)) / 1000
        wnet = wt - wp
        return max(wnet / qin * 100, 0), wnet, wt, wp
    except Exception:
        return 0, 0, 0, 0


def sco2_offdesign(T_src_C, load_fraction):
    """sCO₂ part-load via sliding pressure."""
    eta_t = ETA_T_0 * stodola_eta_correction(load_fraction)
    eta_c = ETA_P_0 * stodola_eta_correction(load_fraction)
    fluid = "CO2"
    T_K   = T_src_C + 273.15
    T1_K  = T_CRIT_CO2 + 5
    P_hi0, P_lo0 = 20e6, 8e6
    P_hi  = P_hi0 * (0.5 + 0.5 * load_fraction)  # partial sliding
    P_lo  = P_lo0

    try:
        h1  = CP.PropsSI("H","T",T1_K,"P",P_lo,fluid)
        s1  = CP.PropsSI("S","T",T1_K,"P",P_lo,fluid)
        h2s = CP.PropsSI("H","P",P_hi,"S",s1,fluid)
        wc  = (h2s - h1) / eta_c / 1000
        h3  = CP.PropsSI("H","T",T_K,"P",P_hi,fluid)
        s3  = CP.PropsSI("S","T",T_K,"P",P_hi,fluid)
        h4s = CP.PropsSI("H","P",P_lo,"S",s3,fluid)
        wt  = (h3 - h4s) * eta_t / 1000
        h4  = h3 - wt * 1000
        T4K = CP.PropsSI("T","P",P_lo,"H",h4,fluid)
        hcm = CP.PropsSI("H","T",T4K,"P",P_hi,fluid)
        h2a = h1 + wc * 1000
        Qr  = 0.85 * (hcm - h2a) / 1000
        qin = (h3 - (h2a + Qr*1000)) / 1000
        wnet = wt - wc
        return max(wnet / qin * 100, 0), wnet, wt, wc
    except Exception:
        return 0, 0, 0, 0


def tco2_offdesign(T_src_C, P_hi, P_lo, eps, load_fraction):
    """tCO₂ part-load via sliding high-side pressure."""
    eta_t = ETA_T_0 * stodola_eta_correction(load_fraction)
    eta_p = ETA_P_0

    P_hi_od = P_hi * (0.6 + 0.4 * load_fraction)  # partial sliding
    try:
        r = tco2_rankine_cycle(
            T_heater_in_C             = T_src_C,
            P_high_Pa                 = P_hi_od,
            P_low_Pa                  = P_lo,
            eta_turbine               = eta_t,
            eta_pump                  = eta_p,
            recuperator_effectiveness = eps,
        )
        return r["eta_th_pct"], r["w_net_kJ_kg"], \
               r["w_turbine_kJ_kg"], r["w_pump_kJ_kg"]
    except Exception:
        return 0, 0, 0, 0


def run_offdesign_sweep():
    """Run part-load sweep for all cycles at all case studies."""
    print("\n" + "="*72)
    print("RQ4 — OFF-DESIGN PART-LOAD PERFORMANCE (Stodola ellipse law)")
    print("="*72)

    all_results = {}

    for label, T_src, P_hi, P_lo, eps in CASE_STUDIES:
        print(f"\n{'─'*72}")
        print(f"  {label}")
        print(f"{'─'*72}")
        print(f"  {'Load':>6}  {'SRC η':>8}  {'sCO₂ η':>8}  {'tCO₂ η':>8}  "
              f"{'SRC/100%':>10}  {'sCO₂/100%':>10}  {'tCO₂/100%':>10}")
        print(f"  {'(%)':>6}  {'(%)':>8}  {'(%)':>8}  {'(%)':>8}  "
              f"{'(η ratio)':>10}  {'(η ratio)':>10}  {'(η ratio)':>10}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  "
              f"{'─'*10}  {'─'*10}  {'─'*10}")

        res = {"load": [], "SRC": [], "sCO₂": [], "tCO₂": [],
               "SRC_r": [], "sCO₂_r": [], "tCO₂_r": []}

        # Get design-point values first
        eta100_src,  _, _, _ = src_offdesign(T_src, 1.0)
        eta100_sco2, _, _, _ = sco2_offdesign(T_src, 1.0)
        eta100_tco2, _, _, _ = tco2_offdesign(T_src, P_hi, P_lo, eps, 1.0)

        for load in LOAD_LEVELS:
            eta_src,  _, _, _ = src_offdesign(T_src, load)
            eta_sco2, _, _, _ = sco2_offdesign(T_src, load)
            eta_tco2, _, _, _ = tco2_offdesign(T_src, P_hi, P_lo, eps, load)

            r_src  = eta_src  / eta100_src  if eta100_src  > 0 else 0
            r_sco2 = eta_sco2 / eta100_sco2 if eta100_sco2 > 0 else 0
            r_tco2 = eta_tco2 / eta100_tco2 if eta100_tco2 > 0 else 0

            print(f"  {load*100:>5.0f}%  {eta_src:>8.2f}  {eta_sco2:>8.2f}  "
                  f"{eta_tco2:>8.2f}  {r_src:>10.3f}  {r_sco2:>10.3f}  "
                  f"{r_tco2:>10.3f}")

            res["load"].append(load * 100)
            res["SRC"].append(eta_src)
            res["sCO₂"].append(eta_sco2)
            res["tCO₂"].append(eta_tco2)
            res["SRC_r"].append(r_src)
            res["sCO₂_r"].append(r_sco2)
            res["tCO₂_r"].append(r_tco2)

        all_results[label] = res

    return all_results


def marchionni_flexibility_score(results):
    """
    Score each cycle on operational flexibility using a simplified
    version of the Marchionni et al. (2020) framework.

    Criteria and weights:
      1. Turndown ratio (minimum viable load)     w=0.30
      2. Efficiency retention at 60% load         w=0.30
      3. Efficiency retention at 40% load         w=0.25
      4. Startup complexity (qualitative)         w=0.15
    """
    scores = {}
    labels = list(results.keys())
    cycles = ["SRC", "sCO₂", "tCO₂"]

    # Startup complexity scores (qualitative, from literature)
    startup = {"SRC": 0.55, "sCO₂": 0.75, "tCO₂": 0.80}
    # Higher = simpler startup (CO2 cycles start faster, no steam drum hazards)

    print("\n" + "="*60)
    print("RQ4 — MARCHIONNI FLEXIBILITY SCORES")
    print("="*60)
    print(f"\n  {'Cycle':>6}  {'Turndown':>10}  {'η@60%':>8}  "
          f"{'η@40%':>8}  {'Startup':>8}  {'TOTAL':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for cycle in cycles:
        # Average retention across all case studies
        r60 = np.mean([results[l][f"{cycle}_r"][LOAD_LEVELS.index(0.60)]
                       for l in labels])
        r40 = np.mean([results[l][f"{cycle}_r"][LOAD_LEVELS.index(0.40)]
                       for l in labels])

        # Turndown: min load where η_ratio > 0.7 → score proportional
        turndown_score = 0.85 if r40 > 0.80 else (0.70 if r40 > 0.70 else 0.55)

        total = (0.30 * turndown_score +
                 0.30 * r60 +
                 0.25 * r40 +
                 0.15 * startup[cycle])

        scores[cycle] = {
            "turndown": turndown_score,
            "r60": r60, "r40": r40,
            "startup": startup[cycle],
            "total": total,
        }
        print(f"  {cycle:>6}  {turndown_score:>10.2f}  {r60:>8.3f}  "
              f"{r40:>8.3f}  {startup[cycle]:>8.2f}  {total:>8.3f}")

    return scores


def plot_offdesign(results, flex_scores):
    """Plot part-load curves and flexibility scores."""
    colors = {"SRC": "#51247A", "sCO₂": "#0064B4", "tCO₂": "#BB9B1C"}
    markers = {"SRC": "o", "sCO₂": "s", "tCO₂": "^"}
    labels = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1-3: η_retention vs load for each case study ───────────
    for idx, (label, ax) in enumerate(zip(labels, axes.flatten()[:3])):
        res = results[label]
        for cycle in ["SRC", "sCO₂", "tCO₂"]:
            ax.plot(res["load"], res[f"{cycle}_r"],
                    color=colors[cycle], marker=markers[cycle],
                    linewidth=2.2, markersize=7, label=cycle)
        ax.axhline(0.90, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(22, 0.91, "90% retention", fontsize=8, color="gray")
        ax.axhline(0.80, color="gray", linestyle=":", alpha=0.4, linewidth=1)
        ax.set_xlabel("Load (%)", fontsize=10)
        ax.set_ylabel("Efficiency Retention (η / η_100%)", fontsize=9)
        ax.set_title(f"{label}\nEfficiency Retention vs Part-Load", fontsize=9)
        ax.set_xlim(15, 105)
        ax.set_ylim(0.6, 1.05)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)

    # ── Panel 4: Flexibility scores ────────────────────────────────────
    ax4 = axes[1, 1]
    criteria = ["Turndown", "η@60%", "η@40%", "Startup", "TOTAL"]
    x = np.arange(len(criteria))
    width = 0.25
    for i, cycle in enumerate(["SRC", "sCO₂", "tCO₂"]):
        s = flex_scores[cycle]
        vals = [s["turndown"], s["r60"], s["r40"],
                s["startup"], s["total"]]
        bars = ax4.bar(x + i*width, vals, width,
                       color=colors[cycle], label=cycle, alpha=0.88)
        for bar in bars:
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01,
                     f"{bar.get_height():.2f}",
                     ha="center", va="bottom", fontsize=7)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(criteria, fontsize=9)
    ax4.set_ylabel("Score (0–1)", fontsize=10)
    ax4.set_title("Marchionni Flexibility Scoring\n"
                  "(higher = more flexible)", fontsize=9)
    ax4.set_ylim(0, 1.1)
    ax4.legend(fontsize=8.5)
    ax4.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "RQ4 — Off-Design Part-Load Performance\n"
        "Stodola ellipse law  |  Sliding pressure control  |  "
        "η_od = η_0 × (1 − 0.3(1−f)²)",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("RQ4_OFFDESIGN_PARTLOAD.png", dpi=150)
    plt.show()
    print("Saved: RQ4_OFFDESIGN_PARTLOAD.png")


if __name__ == "__main__":
    results = run_offdesign_sweep()
    flex    = marchionni_flexibility_score(results)
    plot_offdesign(results, flex)
