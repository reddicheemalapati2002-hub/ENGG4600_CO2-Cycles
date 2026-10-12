import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt

# ── Turbine Size Parameter and Volumetric Flow Ratio ─────────────────────
# Macchi & Astolfi (2017): SP = V_out^0.5 / dh_is^0.25
# Small SP → compact turbine (CO2 advantage over steam)

def turbine_size_parameter(V_dot_out_m3s, dh_is_kJkg):
    """
    SP = sqrt(V_dot_out) / dh_is^0.25
    Units: V_dot in m3/s, dh_is in kJ/kg → SP in m·(kJ/kg)^-0.25
    Lower SP = more compact turbine.
    """
    return (V_dot_out_m3s ** 0.5) / (dh_is_kJkg ** 0.25)

def volumetric_flow_ratio(v_out, v_in):
    """VFR = specific volume at outlet / specific volume at inlet."""
    return v_out / v_in

def compute_turbine_sp_all_cycles(mass_flow_kg_s=1.0):
    """
    Compute SP and VFR for all three cycles at all three case study
    temperatures. mass_flow_kg_s=1.0 gives specific values (per kg/s).
    """
    fluid_co2   = "CO2"
    fluid_water = "Water"

    case_studies = [
        ("CS2: Gas 200°C",    200, 12.8e6, 5e6),
        ("CS1: Cement 350°C", 350, 20.0e6, 5e6),
        ("CS3: Steel 500°C",  500, 20.0e6, 5e6),
    ]

    print("\n" + "="*80)
    print("RQ2 — TURBINE SIZE PARAMETER AND VOLUMETRIC FLOW RATIO")
    print("="*80)
    print(f"\n  {'Case':>22}  {'Cycle':>8}  {'SP (m)':>10}  "
          f"{'VFR':>8}  {'Notes'}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*30}")

    results = []

    for label, T_src, P_hi_co2, P_lo_co2 in case_studies:
        T_K = T_src + 273.15
        eta_t = 0.85
        eta_p = 0.80

        # ── SRC ───────────────────────────────────────────────────────
        P_src_hi = min(10e6, 0.95 * CP.PropsSI("P","T",T_K,"Q",1,fluid_water)) \
                   if T_K < CP.PropsSI("Tcrit","Water") else 20e6
        P_src_lo = 10000  # 10 kPa condenser

        h3_src = CP.PropsSI("H","T",T_K,"P",P_src_hi,fluid_water)
        s3_src = CP.PropsSI("S","T",T_K,"P",P_src_hi,fluid_water)
        h4s_src = CP.PropsSI("H","P",P_src_lo,"S",s3_src,fluid_water)
        dh_is_src = (h3_src - h4s_src) / 1000  # kJ/kg
        h4_src = h3_src - eta_t * (h3_src - h4s_src)
        v4_src = 1 / CP.PropsSI("D","P",P_src_lo,"H",h4_src,fluid_water)
        v3_src = 1 / CP.PropsSI("D","T",T_K,"P",P_src_hi,fluid_water)
        V_dot_out_src = mass_flow_kg_s * v4_src
        sp_src  = turbine_size_parameter(V_dot_out_src, dh_is_src)
        vfr_src = volumetric_flow_ratio(v4_src, v3_src)

        # ── sCO2 Brayton ──────────────────────────────────────────────
        P_sco2_hi = 20e6
        P_sco2_lo = 8e6
        T1_sco2   = CP.PropsSI("Tcrit","CO2") + 5
        h3_sco2 = CP.PropsSI("H","T",T_K,"P",P_sco2_hi,fluid_co2)
        s3_sco2 = CP.PropsSI("S","T",T_K,"P",P_sco2_hi,fluid_co2)
        h4s_sco2 = CP.PropsSI("H","P",P_sco2_lo,"S",s3_sco2,fluid_co2)
        dh_is_sco2 = (h3_sco2 - h4s_sco2) / 1000
        h4_sco2 = h3_sco2 - eta_t * (h3_sco2 - h4s_sco2)
        v4_sco2 = 1 / CP.PropsSI("D","P",P_sco2_lo,"H",h4_sco2,fluid_co2)
        v3_sco2 = 1 / CP.PropsSI("D","T",T_K,"P",P_sco2_hi,fluid_co2)
        V_dot_out_sco2 = mass_flow_kg_s * v4_sco2
        sp_sco2  = turbine_size_parameter(V_dot_out_sco2, dh_is_sco2)
        vfr_sco2 = volumetric_flow_ratio(v4_sco2, v3_sco2)

        # ── tCO2 Rankine ──────────────────────────────────────────────
        h3_tco2 = CP.PropsSI("H","T",T_K,"P",P_hi_co2,fluid_co2)
        s3_tco2 = CP.PropsSI("S","T",T_K,"P",P_hi_co2,fluid_co2)
        h4s_tco2 = CP.PropsSI("H","P",P_lo_co2,"S",s3_tco2,fluid_co2)
        dh_is_tco2 = (h3_tco2 - h4s_tco2) / 1000
        h4_tco2 = h3_tco2 - eta_t * (h3_tco2 - h4s_tco2)
        v4_tco2 = 1 / CP.PropsSI("D","P",P_lo_co2,"H",h4_tco2,fluid_co2)
        v3_tco2 = 1 / CP.PropsSI("D","T",T_K,"P",P_hi_co2,fluid_co2)
        V_dot_out_tco2 = mass_flow_kg_s * v4_tco2
        sp_tco2  = turbine_size_parameter(V_dot_out_tco2, dh_is_tco2)
        vfr_tco2 = volumetric_flow_ratio(v4_tco2, v3_tco2)

        for cycle, sp, vfr in [("SRC", sp_src, vfr_src),
                                 ("sCO₂", sp_sco2, vfr_sco2),
                                 ("tCO₂", sp_tco2, vfr_tco2)]:
            print(f"  {label:>22}  {cycle:>8}  {sp:>10.4f}  "
                  f"{vfr:>8.2f}")
            results.append((label, cycle, sp, vfr))

    return results

def plot_sp_vfr_comparison():
    """Bar chart comparing SP and VFR for all three cycles and case studies."""

    cases  = ["CS2\nGas 200°C", "CS1\nCement 350°C", "CS3\nSteel 500°C"]
    cycles = ["SRC", "sCO₂", "tCO₂"]
    colors = {"SRC": "#51247A", "sCO₂": "#0064B4", "tCO₂": "#BB9B1C"}

    sp_data = {
        "SRC":   [0.6634, 0.5921, 0.5759],
        "sCO₂":  [0.0315, 0.0352, 0.0377],
        "tCO₂":  [0.0404, 0.0391, 0.0419],
    }
    vfr_data = {
        "SRC":   [89.54,  504.49, 807.29],
        "sCO₂":  [2.02,   2.09,   2.11],
        "tCO₂":  [2.14,   3.08,   3.14],
    }

    x     = np.arange(len(cases))
    width = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── SP panel ──────────────────────────────────────────────────────
    for i, cycle in enumerate(cycles):
        bars = ax1.bar(x + i*width, sp_data[cycle], width,
                       label=cycle, color=colors[cycle],
                       alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(cases, fontsize=10)
    ax1.set_ylabel("Size Parameter SP (m·(kJ/kg)⁻⁰·²⁵)", fontsize=10)
    ax1.set_title("Turbine Size Parameter\n"
                  "Smaller SP = More Compact Turbine", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, 0.80)

    # Annotation showing CO2 compactness
    ax1.annotate("CO₂ cycles\n14–21× more\ncompact than SRC",
                 xy=(0.75, 0.52), xytext=(1.5, 0.62),
                 arrowprops=dict(arrowstyle="->", color="#51247A"),
                 fontsize=8.5, color="#51247A",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#51247A", alpha=0.9))

    # ── VFR panel — log scale ─────────────────────────────────────────
    for i, cycle in enumerate(cycles):
        bars = ax2.bar(x + i*width, vfr_data[cycle], width,
                       label=cycle, color=colors[cycle],
                       alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar in bars:
            val = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2,
                     val * 1.08,
                     f"{val:.1f}",
                     ha="center", va="bottom", fontsize=7.5)

    ax2.set_yscale("log")
    ax2.axhline(5, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
    ax2.text(2.75, 5.5, "VFR=5\nSingle-stage\nthreshold",
             fontsize=7.5, color="red", alpha=0.8)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(cases, fontsize=10)
    ax2.set_ylabel("Volumetric Flow Ratio VFR (log scale)", fontsize=10)
    ax2.set_title("Volumetric Flow Ratio\n"
                  "VFR < 5 → Single-Stage Radial Turbine Feasible", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "RQ2 — Turbine Compactness Comparison: SRC vs sCO₂ vs tCO₂\n"
        "Mass flow = 1 kg/s  |  η_turbine = 0.85  |  "
        "P_high scaled with TIT  |  SRC P_low = 10 kPa",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("RQ2_TURBINE_SP_VFR.png", dpi=150)
    plt.show()
    print("Saved: RQ2_TURBINE_SP_VFR.png")

# ═══════════════════════════════════════════════════════════════════════════
# NTU-EFFECTIVENESS HEAT EXCHANGER SIZING
# Uses 50-point T-H profile — mandatory due to pseudo-critical cp spike
# Reference: Dostal (2004) PCHE correlations for CO2 side
# ═══════════════════════════════════════════════════════════════════════════

def ntu_hx_sizing():
    """
    Size the tCO2 supercritical heater using NTU-effectiveness method
    with 50-point T-H discretisation.

    Key correction vs naive implementation:
      - Run at ε=0 so pseudo-critical cp spike is inside the heater
      - Source profile is linear in CO2 ENTHALPY (constant cp source),
        not in CO2 temperature.  This gives a physically correct
        non-uniform ΔT profile that reveals the NTU advantage.
    """
    fluid = "CO2"
    MIN_APPROACH = 10.0
    n_pts = 50

    h_co2_side = 2000.0   # W/m²K  — PCHE CO2 supercritical side
    h_gas_side  =  500.0  # W/m²K  — exhaust gas side
    t_wall      = 0.002   # m
    k_wall      = 15.0    # W/mK
    U_overall   = 1 / (1/h_co2_side + t_wall/k_wall + 1/h_gas_side)

    # Run at eps=0 so cp spike is inside the heater, not the recuperator
    case_studies = [
        ("CS2: Gas 200°C",    200, 12.8e6, 5e6, 0.0),
        ("CS1: Cement 350°C", 350, 20.0e6, 5e6, 0.0),
        ("CS3: Steel 500°C",  500, 20.0e6, 5e6, 0.0),
    ]

    print("\n" + "="*76)
    print("RQ2 — NTU-EFFECTIVENESS HX SIZING  (ε=0, cp spike inside heater)")
    print(f"  U_overall = {U_overall:.1f} W/m²K")
    print("="*76)

    results = {}

    for label, T_src, P_hi, P_lo, eps in case_studies:

        # ── Cycle: get pump exit temperature ──────────────────────────
        from tco2_model import tco2_rankine_cycle
        r = tco2_rankine_cycle(
            T_heater_in_C             = T_src,
            P_high_Pa                 = P_hi,
            P_low_Pa                  = P_lo,
            eta_turbine               = 0.85,
            eta_pump                  = 0.80,
            recuperator_effectiveness = eps,    # 0.0
        )
        T_heater_inlet_C = r["state_2"]["T_C"]   # pump exit at eps=0
        T_sink_C         = T_heater_inlet_C + MIN_APPROACH

        # ── 50-point CO2 T-H profile ──────────────────────────────────
        T_turbine_K   = T_src + 273.15 - 10.0
        T_heater_in_K = T_heater_inlet_C + 273.15
        T_co2_range   = np.linspace(T_heater_in_K, T_turbine_K, n_pts)
        h_co2         = [CP.PropsSI("H","T",T,"P",P_hi,fluid)/1000
                         for T in T_co2_range]
        T_co2_C       = [T - 273.15 for T in T_co2_range]
        h_base        = h_co2[0]
        h_co2_norm    = [h - h_base for h in h_co2]
        Q_total       = h_co2_norm[-1]

        # ── Source profile: linear in CO2 ENTHALPY (constant cp source)
        # At cold end (i=0): source exits at T_sink (counter-flow)
        # At hot  end (i=n): source enters at T_src
        # As CO2 absorbs fraction f=h_norm/Q_total of total Q,
        # source has delivered the same fraction of its temperature drop.
        T_src_profile = [T_sink_C + (T_src - T_sink_C) *
                         (h_co2_norm[i] / Q_total)
                         for i in range(n_pts)]

        # ── Segment-by-segment NTU area ───────────────────────────────
        UA_total = 0.0
        A_ntu    = 0.0

        for i in range(n_pts - 1):
            dQ_kJkg = h_co2[i+1] - h_co2[i]
            dQ_W    = dQ_kJkg * 1000

            dT_cold = T_src_profile[i]   - T_co2_C[i]
            dT_hot  = T_src_profile[i+1] - T_co2_C[i+1]

            dT_cold = max(dT_cold, 0.1)
            dT_hot  = max(dT_hot,  0.1)

            if abs(dT_hot - dT_cold) < 0.01:
                LMTD_seg = dT_hot
            else:
                LMTD_seg = (dT_hot - dT_cold) / np.log(dT_hot / dT_cold)

            UA_total += dQ_W / LMTD_seg
            A_ntu    += (dQ_W / LMTD_seg) / U_overall

        # ── Simple LMTD (2-point only) ────────────────────────────────
        dT_cold_simple = T_sink_C        - T_heater_inlet_C
        dT_hot_simple  = T_src           - (T_turbine_K - 273.15)
        dT_cold_simple = max(dT_cold_simple, 0.1)
        dT_hot_simple  = max(dT_hot_simple,  0.1)

        if abs(dT_hot_simple - dT_cold_simple) < 0.01:
            LMTD_simple = dT_hot_simple
        else:
            LMTD_simple = (dT_hot_simple - dT_cold_simple) / \
                           np.log(dT_hot_simple / dT_cold_simple)

        Q_total_W  = Q_total * 1000
        A_simple   = (Q_total_W / LMTD_simple) / U_overall
        under_pct  = (A_ntu - A_simple) / A_ntu * 100

        results[label] = {
            "A_ntu":             round(A_ntu, 3),
            "A_simple":          round(A_simple, 3),
            "UA_total":          round(UA_total, 1),
            "LMTD_simple":       round(LMTD_simple, 1),
            "Q_total_kJkg":      round(Q_total, 1),
            "underestimate_pct": round(under_pct, 1),
            "T_sink_C":          round(T_sink_C, 1),
            "T_heater_inlet":    round(T_heater_inlet_C, 1),
        }

        print(f"\n  {label}")
        print(f"  {'─'*60}")
        print(f"  Heater inlet T_CO2:    {T_heater_inlet_C:.1f} °C  (pump exit, ε=0)")
        print(f"  T_sink (waste gas):    {T_sink_C:.1f} °C")
        print(f"  Total HX duty:         {Q_total:.1f} kJ/kg")
        print(f"  NTU area (50-point):   {A_ntu:.3f} m² per kg/s CO₂")
        print(f"  Simple LMTD area:      {A_simple:.3f} m² per kg/s CO₂")
        print(f"  LMTD underestimates:   {under_pct:.1f}%")
        print(f"  Simple LMTD value:     {LMTD_simple:.1f} K")

    _plot_hx_area_comparison(results)
    return results

def _plot_hx_area_comparison(results):
    labels  = list(results.keys())
    a_ntu   = [results[l]["A_ntu"]    for l in labels]
    a_simp  = [results[l]["A_simple"] for l in labels]

    # Correct sign: LMTD overestimates (positive = bigger than NTU)
    over    = [(results[l]["A_simple"] - results[l]["A_ntu"]) /
               results[l]["A_ntu"] * 100 for l in labels]

    x     = np.arange(len(labels))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Area comparison ───────────────────────────────────────────────
    b1 = ax1.bar(x - width/2, a_ntu,  width,
                 label="NTU-effectiveness\n(50-point T-H)",
                 color="#51247A", alpha=0.88)
    b2 = ax1.bar(x + width/2, a_simp, width,
                 label="Simple LMTD\n(2-point terminals)",
                 color="#BB9B1C", alpha=0.88)

    for bar in b1:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=9)
    for bar in b2:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace(": ",":\n") for l in labels], fontsize=9)
    ax1.set_ylabel("HX Area (m² per kg/s CO₂)", fontsize=10)
    ax1.set_title(
        "tCO₂ Heater Area: NTU-Effectiveness vs Simple LMTD\n"
        "ε=0, pump exit → turbine inlet  |  PCHE U = 380 W/m²K",
        fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # ── Overestimation panel ──────────────────────────────────────────
    bar_colors = ["#BB9B1C", "#51247A", "#0064B4"]
    bars = ax2.bar(x, over, color=bar_colors, alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, over):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 f"+{val:.0f}%", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(": ",":\n") for l in labels], fontsize=9)
    ax2.set_ylabel("Simple LMTD Overestimation of HX Area (%)", fontsize=10)
    ax2.set_title(
        "Simple LMTD OVERestimates Required Area\n"
        "Because Both Terminal ΔTs = 10 K (Mid-Exchanger Pinch Effect)",
        fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, max(over) * 1.25)

    plt.suptitle(
        "RQ2 — tCO₂ Supercritical Heater Sizing  (ε = 0, P_low = 5 MPa)\n"
        "NTU-effectiveness with 50-point T-H correctly captures non-uniform "
        "cₚ profile; simple LMTD uses only 2-point terminals",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("RQ2_HX_AREA_NTU_vs_LMTD.png", dpi=150)
    plt.show()
    print("Saved: RQ2_HX_AREA_NTU_vs_LMTD.png")
if __name__ == "__main__":
    results = compute_turbine_sp_all_cycles(mass_flow_kg_s=1.0)
    plot_sp_vfr_comparison()
    hx_results = ntu_hx_sizing()

