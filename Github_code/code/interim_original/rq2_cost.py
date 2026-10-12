"""
rq2_cost.py
───────────
Capital cost estimation for all three cycles at the three Australian case
study conditions. Converts SP/VFR and HX area results from rq2_sizing.py
into equipment cost (AUD/kW) at four plant sizes.

Cost basis and escalation
─────────────────────────
  Turbines  : Weiland & Thimsen (2016) correlation, 2016 USD
              C_turb = a × W_net^b  (power-law, per-unit equipment)
  HX / PCHE : Weiland, Lance & Pidaparti (NETL GT2019-90493), 2017 USD
              C_HX = 49.45 × UA^0.7544 × f_T
  CEPCI escalation : 2016→2026 factor = 830/541.7 = 1.532
                     2017→2026 factor = 830/567.5 = 1.463
  Exchange rate    : 1 USD = 1.395 AUD  (RBA, 17 April 2026)
  Installation     : +35% on all equipment (Turton et al. 4th ed.)

References
──────────
  Weiland & Thimsen (2016) Proc. 5th Int. Symp. sCO2 Power Cycles
  Weiland, Lance & Pidaparti (2019) ASME GT2019-90493
  Turton et al. (2012) Analysis, Synthesis and Design of Chemical Processes
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

# ── Escalation and exchange ────────────────────────────────────────────────
CEPCI_2016   = 541.7
CEPCI_2017   = 567.5
CEPCI_2026   = 830.0      # best estimate, April 2026
ESCL_2016    = CEPCI_2026 / CEPCI_2016   # 1.532 — for turbine correlation
ESCL_2017    = CEPCI_2026 / CEPCI_2017   # 1.463 — for HX/PCHE correlation
USD_TO_AUD   = 1.395      # RBA spot, 17 April 2026
INSTALL_FACT = 1.35       # +35% installation on all equipment


# ══════════════════════════════════════════════════════════════════════════
# TURBINE COST — Weiland & Thimsen (2016)
# C_turb [2016 USD] = a × W_kW^b
# Coefficients differ by cycle type:
#   SRC   : a=1200, b=0.70  (small industrial steam turbine)
#   sCO₂  : a=2500, b=0.67  (radial turbine, sub-5 MWe range)
#   tCO₂  : a=2200, b=0.68  (similar to ORC radial at this scale)
# ══════════════════════════════════════════════════════════════════════════

TURB_COEFFS = {
    "SRC":  {"a": 1200, "b": 0.70},
    "sCO₂": {"a": 2500, "b": 0.67},
    "tCO₂": {"a": 2200, "b": 0.68},
}

def turbine_cost_usd_2026(W_kW, cycle):
    """
    Turbine equipment cost in 2026 USD at net output W_kW.
    Weiland & Thimsen (2016), escalated from 2016 USD.
    """
    c = TURB_COEFFS[cycle]
    cost_2016 = c["a"] * (W_kW ** c["b"])
    return cost_2016 * ESCL_2016


# ══════════════════════════════════════════════════════════════════════════
# HEAT EXCHANGER COST — Weiland, Lance & Pidaparti (NETL, 2019)
# C_HX [2017 USD] = 49.45 × UA^0.7544 × f_T
# UA in W/K
# f_T = 1.0          for T_max ≤ 550°C
#     = 1 + 0.02141 × (T_max - 550)   for T_max > 550°C
# ══════════════════════════════════════════════════════════════════════════

def hx_temperature_factor(T_max_C):
    if T_max_C <= 550:
        return 1.0
    return 1.0 + 0.02141 * (T_max_C - 550)

def hx_cost_usd_2026(UA_W_per_K, T_max_C):
    """
    PCHE/HX equipment cost in 2026 USD.
    Weiland, Lance & Pidaparti (2019), escalated from 2017 USD.
    UA in W/K.
    """
    f_T = hx_temperature_factor(T_max_C)
    cost_2017 = 49.45 * (UA_W_per_K ** 0.7544) * f_T
    return cost_2017 * ESCL_2017


# ══════════════════════════════════════════════════════════════════════════
# PUMP / COMPRESSOR COST
# Simple power-law based on shaft power (Turton et al. 2012)
# C_pump [2001 USD] = 3500 × W_kW^0.45  (centrifugal pump/compressor)
# ══════════════════════════════════════════════════════════════════════════

CEPCI_2001 = 394.3
ESCL_2001  = CEPCI_2026 / CEPCI_2001   # 2.105

def pump_cost_usd_2026(W_pump_kW):
    cost_2001 = 3500 * (W_pump_kW ** 0.45)
    return cost_2001 * ESCL_2001


# ══════════════════════════════════════════════════════════════════════════
# CONDENSER / COOLER COST
# Shell-and-tube (SRC condenser, CO2 gas cooler)
# C_cond [2001 USD] = 1200 × A_m2^0.6   (Turton et al.)
# Area estimated from duty: A = Q / (U × LMTD)
# U_condenser ≈ 1500 W/m²K (water-cooled shell-and-tube)
# LMTD_condenser ≈ 10 K (conservative)
# ══════════════════════════════════════════════════════════════════════════

U_COND   = 1500.0   # W/m²K
LMTD_COND = 10.0    # K

def condenser_cost_usd_2026(Q_kJ_per_kg, mass_flow_kg_s):
    Q_W = Q_kJ_per_kg * 1000 * mass_flow_kg_s
    A_m2 = Q_W / (U_COND * LMTD_COND)
    cost_2001 = 1200 * (A_m2 ** 0.6)
    return cost_2001 * ESCL_2001


# ══════════════════════════════════════════════════════════════════════════
# MAIN COST ESTIMATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def estimate_capital_costs():
    """
    Estimate total erected cost (TEC) in AUD/kW for each cycle
    at each case study temperature and four plant sizes.

    Plant sizes: 100 kW, 500 kW, 1 MW, 5 MW net output.
    Mass flow is derived from net specific work and plant size.
    """
    from rq2_sizing import compute_turbine_sp_all_cycles
    from tco2_model import tco2_rankine_cycle, src_eff_bwr, sco2_eff_bwr

    fluid_co2   = "CO2"
    fluid_water = "Water"

    case_studies = [
        ("CS2: Gas 200°C",    200, 12.8e6, 5e6),
        ("CS1: Cement 350°C", 350, 20.0e6, 5e6),
        ("CS3: Steel 500°C",  500, 20.0e6, 5e6),
    ]
    plant_sizes_kW = [100, 500, 1000, 5000]
    eps_tco2  = 0.70
    eps_sco2  = 0.85
    eta_t     = 0.85
    eta_p     = 0.80

    all_results = {}

    print("\n" + "="*80)
    print("RQ2 — CAPITAL COST ESTIMATION  (AUD/kW, 2026)")
    print(f"  CEPCI 2026 ≈ {CEPCI_2026:.0f}  |  Escl 2016→2026 = {ESCL_2016:.3f}×  "
          f"|  Escl 2017→2026 = {ESCL_2017:.3f}×")
    print(f"  USD→AUD = {USD_TO_AUD}  |  Installation factor = {INSTALL_FACT}×")
    print("="*80)

    for label, T_src, P_hi_co2, P_lo_co2 in case_studies:
        T_K = T_src + 273.15
        print(f"\n{'─'*80}")
        print(f"  {label}")
        print(f"{'─'*80}")

        all_results[label] = {}

        # ── Specific work for each cycle ──────────────────────────────
        # SRC
        P_src_hi = min(10e6, 0.95 * CP.PropsSI("P","T",T_K,"Q",1,fluid_water)) \
                   if T_K < CP.PropsSI("Tcrit","Water") else 20e6
        P_src_lo = 10000
        h3w = CP.PropsSI("H","T",T_K,"P",P_src_hi,fluid_water)
        s3w = CP.PropsSI("S","T",T_K,"P",P_src_hi,fluid_water)
        h4sw = CP.PropsSI("H","P",P_src_lo,"S",s3w,fluid_water)
        h1w  = CP.PropsSI("H","P",P_src_lo,"Q",0,fluid_water)
        s1w  = CP.PropsSI("S","P",P_src_lo,"Q",0,fluid_water)
        h2sw = CP.PropsSI("H","P",P_src_hi,"S",s1w,fluid_water)
        w_pump_src = (h2sw - h1w) / eta_p / 1000
        w_turb_src = (h3w - h4sw) * eta_t / 1000
        w_net_src  = w_turb_src - w_pump_src   # kJ/kg
        q_in_src   = (h3w - (h1w + w_pump_src*1000)) / 1000
        q_cond_src = (h4sw*eta_t + (1-eta_t)*h3w - h1w) / 1000  # approx

        # sCO2
        T1_sco2 = CP.PropsSI("Tcrit","CO2") + 5
        P_sco2_hi, P_sco2_lo = 20e6, 8e6
        h1s = CP.PropsSI("H","T",T1_sco2,"P",P_sco2_lo,fluid_co2)
        s1s = CP.PropsSI("S","T",T1_sco2,"P",P_sco2_lo,fluid_co2)
        h2ss = CP.PropsSI("H","P",P_sco2_hi,"S",s1s,fluid_co2)
        w_comp_sco2 = (h2ss - h1s) / eta_p / 1000
        h3s = CP.PropsSI("H","T",T_K,"P",P_sco2_hi,fluid_co2)
        s3s = CP.PropsSI("S","T",T_K,"P",P_sco2_hi,fluid_co2)
        h4ss = CP.PropsSI("H","P",P_sco2_lo,"S",s3s,fluid_co2)
        w_turb_sco2 = (h3s - h4ss) * eta_t / 1000
        w_net_sco2  = w_turb_sco2 - w_comp_sco2
        h4s_actual  = h3s - w_turb_sco2*1000
        T4_sco2_K   = CP.PropsSI("T","P",P_sco2_lo,"H",h4s_actual,fluid_co2)
        h_cold_max  = CP.PropsSI("H","T",T4_sco2_K,"P",P_sco2_hi,fluid_co2)
        h2s_actual  = h1s + w_comp_sco2*1000
        Q_recup_sco2= eps_sco2 * (h_cold_max - h2s_actual) / 1000
        q_in_sco2   = (h3s - (h2s_actual + Q_recup_sco2*1000)) / 1000

        # tCO2
        r_tco2 = tco2_rankine_cycle(T_src, P_hi_co2, P_lo_co2,
                                    eta_t, eta_p, eps_tco2)
        w_net_tco2   = r_tco2["w_net_kJ_kg"]
        w_pump_tco2  = r_tco2["w_pump_kJ_kg"]
        w_turb_tco2  = r_tco2["w_turbine_kJ_kg"]
        q_in_tco2    = r_tco2["q_in_kJ_kg"]
        Q_recup_tco2 = r_tco2["Q_recup_kJ_kg"]

        # UA for HX (per kg/s) — use NTU result from sizing
        # Approximate UA = Q_total [W/kg/s] / LMTD_effective
        # Use the corrected NTU LMTD from the heater (eps=0) as reference
        # Full heater UA at eps=0.70 is smaller; T_sink constraint matters
        U_pche = 379.7  # W/m²K from rq2_sizing
        T_heater_inlet_tco2 = r_tco2["state_2"]["T_C"]
        T_sink_tco2 = T_heater_inlet_tco2 + 10.0

        # Build 50-pt heater T-H for UA
        T_turb_K = T_src + 273.15 - 10
        T_hx_in  = T_heater_inlet_tco2 + 273.15
        T_range  = np.linspace(T_hx_in, T_turb_K, 50)
        h_co2    = [CP.PropsSI("H","T",T,"P",P_hi_co2,fluid_co2)/1000
                    for T in T_range]
        T_co2_C  = [T-273.15 for T in T_range]
        h_norm   = [h - h_co2[0] for h in h_co2]
        Q_tot    = h_norm[-1]
        T_src_p  = [T_sink_tco2+(T_src-T_sink_tco2)*i/49 for i in range(50)]
        UA_per_kgs = 0.0
        for i in range(49):
            dQ_W = (h_co2[i+1]-h_co2[i])*1000
            dT_c = max(T_src_p[i]  - T_co2_C[i],   0.1)
            dT_h = max(T_src_p[i+1]- T_co2_C[i+1], 0.1)
            LMTD = dT_h if abs(dT_h-dT_c)<0.01 \
                   else (dT_h-dT_c)/np.log(dT_h/dT_c)
            UA_per_kgs += dQ_W / LMTD

        print(f"\n  {'Plant':>8}  {'Cycle':>6}  {'ṁ':>8}  {'C_turb':>10}  "
              f"{'C_HX':>10}  {'C_pump':>10}  {'C_total':>10}  "
              f"{'TEC':>12}")
        print(f"  {'(kW)':>8}  {'':>6}  {'(kg/s)':>8}  {'(kAUD)':>10}  "
              f"{'(kAUD)':>10}  {'(kAUD)':>10}  {'(kAUD)':>10}  "
              f"{'(AUD/kW)':>12}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}  "
              f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")

        for W_kW in plant_sizes_kW:
            row = {}
            for cycle, w_net, w_pump, q_in, q_recup in [
                ("SRC",  w_net_src,  w_pump_src,  q_in_src,  0),
                ("sCO₂", w_net_sco2, w_comp_sco2, q_in_sco2, Q_recup_sco2),
                ("tCO₂", w_net_tco2, w_pump_tco2, q_in_tco2, Q_recup_tco2),
            ]:
                m_dot = W_kW / w_net        # kg/s

                # Turbine cost
                C_turb = turbine_cost_usd_2026(W_kW, cycle) * USD_TO_AUD

                # HX (primary heater) cost
                if cycle == "SRC":
                    # Shell-and-tube boiler — simpler correlation
                    Q_boiler_W = q_in * 1000 * m_dot
                    dT1 = T_src - (CP.PropsSI("T","P",P_src_hi,"Q",1,fluid_water)-273.15)
                    dT2 = T_src * 0.1 + 10   # crude approach on cold end
                    LMTD_b = max((dT1-dT2)/np.log(max(dT1/max(dT2,1),1.01)), 5)
                    U_boiler = 800  # W/m²K shell-and-tube steam boiler
                    A_boiler = Q_boiler_W / (U_boiler * LMTD_b)
                    UA_boiler = U_boiler * A_boiler
                    f_T = hx_temperature_factor(T_src)
                    C_hx = hx_cost_usd_2026(UA_boiler, T_src) * USD_TO_AUD
                elif cycle == "sCO₂":
                    # Primary heater UA — approximate from Q/LMTD at terminal ΔTs
                    Q_heat_W = q_in * 1000 * m_dot
                    dT1s = T_src - T_K + 273.15 - 10
                    dT2s = T_src * 0.12
                    LMTD_s = max(abs(dT1s-dT2s)/np.log(max(abs(dT1s/max(dT2s,1)),1.01)),5)
                    UA_sco2 = Q_heat_W / max(LMTD_s, 10)
                    C_hx = hx_cost_usd_2026(UA_sco2, T_src) * USD_TO_AUD
                    # Add recuperator
                    Q_recup_W = q_recup * 1000 * m_dot
                    dT_rec = 20   # typical mid-HX ΔT for recuperator
                    UA_recup = Q_recup_W / dT_rec
                    C_hx += hx_cost_usd_2026(UA_recup, min(T_src, 549)) \
                            * USD_TO_AUD
                else:  # tCO₂
                    UA_heater = UA_per_kgs * m_dot
                    C_hx = hx_cost_usd_2026(UA_heater, T_src) * USD_TO_AUD
                    # Add recuperator at lower temperature
                    Q_recup_W = q_recup * 1000 * m_dot
                    dT_rec = 20
                    UA_recup = Q_recup_W / dT_rec
                    C_hx += hx_cost_usd_2026(
                        UA_recup, r_tco2["state_4"]["T_C"]) * USD_TO_AUD

                # Pump/compressor cost
                W_pump_kW = w_pump * m_dot
                C_pump = pump_cost_usd_2026(max(W_pump_kW, 1)) * USD_TO_AUD

                # Total equipment + installation
                C_equip = C_turb + C_hx + C_pump
                C_total = C_equip * INSTALL_FACT   # kAUD
                TEC_per_kW = C_total * 1000 / W_kW # AUD/kW

                print(f"  {W_kW:>8}  {cycle:>6}  {m_dot:>8.2f}  "
                      f"{C_turb/1000:>10.1f}  {C_hx/1000:>10.1f}  "
                      f"{C_pump/1000:>10.1f}  {C_total/1000:>10.1f}  "
                      f"{TEC_per_kW:>12.0f}")

                row[cycle] = TEC_per_kW

            all_results[label][W_kW] = row

    _plot_capital_costs(all_results, plant_sizes_kW, case_studies)
    return all_results


def _plot_capital_costs(results, plant_sizes_kW, case_studies):
    colors = {"SRC": "#51247A", "sCO₂": "#0064B4", "tCO₂": "#BB9B1C"}
    markers = {"SRC": "o", "sCO₂": "s", "tCO₂": "^"}
    cycles = ["SRC", "sCO₂", "tCO₂"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    for ax, (label, T_src, _, _) in zip(axes, case_studies):
        for cycle in cycles:
            tec = [results[label][W][cycle] for W in plant_sizes_kW]
            ax.plot(plant_sizes_kW, tec,
                    color=colors[cycle], marker=markers[cycle],
                    linewidth=2.2, markersize=8, label=cycle)
            for W, v in zip(plant_sizes_kW, tec):
                ax.annotate(f"{v/1000:.1f}k",
                            xy=(W, v), xytext=(0, 7),
                            textcoords="offset points",
                            ha="center", fontsize=7,
                            color=colors[cycle])

        ax.set_xscale("log")
        ax.set_xticks(plant_sizes_kW)
        ax.set_xticklabels(["100\nkW", "500\nkW", "1\nMW", "5\nMW"], fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Net Output (kW)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5)

    axes[0].set_ylabel("Total Erected Cost (AUD/kW)", fontsize=10)

    plt.suptitle(
        "RQ2 — Capital Cost (AUD/kW) vs Plant Size  |  2026 AUD\n"
        "CEPCI 2026 ≈ 830  |  1 USD = 1.395 AUD  |  Installation +35%\n"
        "SRC: Turton et al.  |  CO₂: Weiland & Thimsen 2016 + "
        "Weiland et al. 2019",
        fontsize=9)
    plt.tight_layout()
    plt.savefig("RQ2_CAPITAL_COST_AUD_per_kW.png", dpi=150)
    plt.show()
    print("\nSaved: RQ2_CAPITAL_COST_AUD_per_kW.png")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from rq2_sizing import compute_turbine_sp_all_cycles, plot_sp_vfr_comparison
    from rq2_sizing import ntu_hx_sizing

    print("Running SP/VFR sizing...")
    compute_turbine_sp_all_cycles()
    plot_sp_vfr_comparison()

    print("\nRunning NTU HX sizing...")
    ntu_hx_sizing()

    print("\nRunning capital cost estimation...")
    cost_results = estimate_capital_costs()
