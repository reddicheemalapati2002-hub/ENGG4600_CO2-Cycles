import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

# ── CO2 critical point ────────────────────────────────────────────────────
T_CRIT_CO2 = CP.PropsSI("Tcrit", "CO2")   # 304.13 K = 30.98°C
P_CRIT_CO2 = CP.PropsSI("Pcrit", "CO2")   # 7.377 MPa

# ═══════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════

def get_props_co2(P_Pa, T_K=None, h_Jkg=None, s_Jkg=None, Q=None):
    """Return T_C, h (kJ/kg), s (kJ/kg·K) for CO2 at a given state."""
    fluid = "CO2"
    if T_K is not None:
        h = CP.PropsSI("H", "T", T_K, "P", P_Pa, fluid)
        s = CP.PropsSI("S", "T", T_K, "P", P_Pa, fluid)
        T_out = T_K
    elif Q is not None:
        h = CP.PropsSI("H", "P", P_Pa, "Q", Q, fluid)
        s = CP.PropsSI("S", "P", P_Pa, "Q", Q, fluid)
        T_out = CP.PropsSI("T", "P", P_Pa, "Q", Q, fluid)
    elif s_Jkg is not None:
        h = CP.PropsSI("H", "P", P_Pa, "S", s_Jkg, fluid)
        s = s_Jkg
        T_out = CP.PropsSI("T", "P", P_Pa, "S", s_Jkg, fluid)
    elif h_Jkg is not None:
        s = CP.PropsSI("S", "P", P_Pa, "H", h_Jkg, fluid)
        h = h_Jkg
        T_out = CP.PropsSI("T", "P", P_Pa, "H", h_Jkg, fluid)
    return {"T_C": T_out - 273.15, "h": h / 1000, "s": s / 1000}


# ═══════════════════════════════════════════════════════════════════════════
# tCO2 RANKINE CYCLE MODEL
# ═══════════════════════════════════════════════════════════════════════════

def tco2_rankine_cycle(T_heater_in_C, P_high_Pa, P_low_Pa,
                       eta_turbine=0.85, eta_pump=0.80,
                       recuperator_effectiveness=0.0):
    fluid = "CO2"
    assert P_low_Pa < P_CRIT_CO2
    assert P_high_Pa > P_CRIT_CO2

    T_heater_K = T_heater_in_C + 273.15
    s1 = get_props_co2(P_low_Pa, Q=0)

    h2s_J         = CP.PropsSI("H", "P", P_high_Pa, "S", s1["s"] * 1000, fluid)
    w_pump_ideal  = (h2s_J / 1000) - s1["h"]
    w_pump_actual = w_pump_ideal / eta_pump
    h2_actual_J   = (s1["h"] + w_pump_actual) * 1000
    s2 = get_props_co2(P_high_Pa, h_Jkg=h2_actual_J)

    s3 = get_props_co2(P_high_Pa, T_K=T_heater_K)

    h4s_J         = CP.PropsSI("H", "P", P_low_Pa, "S", s3["s"] * 1000, fluid)
    w_turb_ideal  = s3["h"] - h4s_J / 1000
    w_turb_actual = w_turb_ideal * eta_turbine
    h4_actual_J   = (s3["h"] - w_turb_actual) * 1000
    s4 = get_props_co2(P_low_Pa, h_Jkg=h4_actual_J)

    Q_recup  = 0.0
    s2_final = s2
    s4_final = s4

    if recuperator_effectiveness > 0:
        T4_K         = CP.PropsSI("T", "P", P_low_Pa, "H", h4_actual_J, fluid)
        h_cold_max_J = CP.PropsSI("H", "T", T4_K, "P", P_high_Pa, fluid)
        Q_max        = (h_cold_max_J / 1000) - s2["h"]
        Q_recup      = recuperator_effectiveness * Q_max
        h2r_J        = (s2["h"] + Q_recup) * 1000
        s2_final     = get_props_co2(P_high_Pa, h_Jkg=h2r_J)
        h4r_J        = (s4["h"] - Q_recup) * 1000
        s4_final     = get_props_co2(P_low_Pa, h_Jkg=h4r_J)

    q_in   = s3["h"] - s2_final["h"]
    w_net  = w_turb_actual - w_pump_actual
    bwr    = w_pump_actual / w_turb_actual
    eta_th = w_net / q_in
    T_cond = CP.PropsSI("T", "P", P_low_Pa, "Q", 0, fluid) - 273.15

    return {
        "state_1":         s1,
        "state_2":         s2_final,
        "state_3":         s3,
        "state_4":         s4_final,
        "T_condenser_C":   round(T_cond, 2),
        "Q_recup_kJ_kg":   round(Q_recup, 2),
        "q_in_kJ_kg":      round(q_in, 2),
        "w_turbine_kJ_kg": round(w_turb_actual, 2),
        "w_pump_kJ_kg":    round(w_pump_actual, 2),
        "w_net_kJ_kg":     round(w_net, 2),
        "bwr_pct":         round(bwr * 100, 2),
        "eta_th_pct":      round(eta_th * 100, 2),
        "P_high_MPa":      round(P_high_Pa / 1e6, 2),
        "P_low_MPa":       round(P_low_Pa / 1e6, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR SRC AND sCO2
# ═══════════════════════════════════════════════════════════════════════════

def src_eff_bwr(T_C, P_high=10e6, P_low=10000, eta_t=0.85, eta_p=0.80):
    fluid  = "Water"
    T_K    = T_C + 273.15
    T_CRIT_W = CP.PropsSI("Tcrit", "Water")
    if T_K < T_CRIT_W:
        Psat = CP.PropsSI("P", "T", T_K, "Q", 1, fluid)
        if P_high >= Psat:
            P_high = 0.95 * Psat
    else:
        P_high = min(P_high, 20e6)
    h1  = CP.PropsSI("H", "P", P_low, "Q", 0, fluid)
    s1  = CP.PropsSI("S", "P", P_low, "Q", 0, fluid)
    h2s = CP.PropsSI("H", "P", P_high, "S", s1, fluid)
    w_p = (h2s - h1) / eta_p
    h2  = h1 + w_p
    h3  = CP.PropsSI("H", "T", T_K, "P", P_high, fluid)
    s3  = CP.PropsSI("S", "T", T_K, "P", P_high, fluid)
    h4s = CP.PropsSI("H", "P", P_low, "S", s3, fluid)
    w_t = (h3 - h4s) * eta_t
    q_in  = h3 - h2
    w_net = w_t - w_p
    return round(w_net / q_in * 100, 2), round(w_p / w_t * 100, 2)


def sco2_eff_bwr(T_C, P_high=20e6, P_low=8e6,
                 eta_t=0.85, eta_c=0.80, eps=0.85):
    fluid = "CO2"
    T_K   = T_C + 273.15
    T1_K  = T_CRIT_CO2 + 5
    h1  = CP.PropsSI("H", "T", T1_K, "P", P_low, fluid)
    s1  = CP.PropsSI("S", "T", T1_K, "P", P_low, fluid)
    h2s = CP.PropsSI("H", "P", P_high, "S", s1, fluid)
    w_c = (h2s - h1) / eta_c
    h2  = h1 + w_c
    h3h = CP.PropsSI("H", "T", T_K, "P", P_high, fluid)
    s3h = CP.PropsSI("S", "T", T_K, "P", P_high, fluid)
    h4s = CP.PropsSI("H", "P", P_low, "S", s3h, fluid)
    w_t = (h3h - h4s) * eta_t
    h4  = h3h - w_t
    T4_K       = CP.PropsSI("T", "P", P_low, "H", h4, fluid)
    h_cold_max = CP.PropsSI("H", "T", T4_K, "P", P_high, fluid)
    Q_recup    = eps * (h_cold_max - h2)
    h3         = h2 + Q_recup
    q_in  = (h3h - h3) / 1000
    w_net = (w_t - w_c) / 1000
    bwr   = w_c / w_t * 100
    return round(w_net / q_in * 100, 2), round(bwr, 2)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-POINT TEST
# ═══════════════════════════════════════════════════════════════════════════

def run_single_point_test():
    print("=" * 55)
    print("tCO2 RANKINE CYCLE — Single Point Test (200°C, no recuperator)")
    print("=" * 55)
    r = tco2_rankine_cycle(
        T_heater_in_C=200, P_high_Pa=12e6, P_low_Pa=5e6,
        eta_turbine=0.85, eta_pump=0.80, recuperator_effectiveness=0.0)
    labels = {
        "state_1": "Condenser out (subcritical liquid)",
        "state_2": "Pump out (supercritical liquid)",
        "state_3": "Turbine in (supercritical CO2)",
        "state_4": "Turbine out (subcritical CO2)",
    }
    for key, label in labels.items():
        sp = r[key]
        print(f"\n{key} — {label}")
        print(f"  T = {sp['T_C']:.1f} °C")
        print(f"  h = {sp['h']:.2f} kJ/kg")
        print(f"  s = {sp['s']:.4f} kJ/kg·K")
    print("\n" + "=" * 55)
    print("PERFORMANCE SUMMARY")
    print("=" * 55)
    print(f"  Condenser temp:    {r['T_condenser_C']} °C")
    print(f"  Heat input:        {r['q_in_kJ_kg']} kJ/kg")
    print(f"  Turbine work:      {r['w_turbine_kJ_kg']} kJ/kg")
    print(f"  Pump work:         {r['w_pump_kJ_kg']} kJ/kg")
    print(f"  Net work:          {r['w_net_kJ_kg']} kJ/kg")
    print(f"  Back-work ratio:   {r['bwr_pct']} %")
    print(f"  Thermal efficiency:{r['eta_th_pct']} %")
    print(f"\n  P_low  = {r['P_low_MPa']} MPa  "
          f"({'✓ subcritical' if r['P_low_MPa'] < 7.377 else '✗ ERROR'})")
    print(f"  P_high = {r['P_high_MPa']} MPa  "
          f"({'✓ supercritical' if r['P_high_MPa'] > 7.377 else '✗ ERROR'})")


# ═══════════════════════════════════════════════════════════════════════════
# THREE-CYCLE PARAMETRIC SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def run_three_cycle_comparison():
    temperatures = range(150, 625, 25)
    src_effs, src_bwrs   = [], []
    sco2_effs, sco2_bwrs = [], []
    tco2_effs, tco2_bwrs = [], []
    tco2_pressures       = []

    print(f"\n{'TIT':>5}  {'P_hi_tco2':>10}  {'SRC η':>7}  "
          f"{'sCO2 η':>7}  {'tCO2 η':>7}  "
          f"{'SRC BWR':>8}  {'sCO2 BWR':>9}  {'tCO2 BWR':>9}")
    print("-" * 80)

    for T in temperatures:
        try:
            e, b = src_eff_bwr(T)
            src_effs.append(e); src_bwrs.append(b)
        except:
            src_effs.append(None); src_bwrs.append(None)
            e, b = None, None

        try:
            e2, b2 = sco2_eff_bwr(T)
            sco2_effs.append(e2); sco2_bwrs.append(b2)
        except:
            sco2_effs.append(None); sco2_bwrs.append(None)
            e2, b2 = None, None

        P_high_tco2 = min(20e6, max(12e6, 8e6 + (T - 100) * 48000))
        tco2_pressures.append(P_high_tco2 / 1e6)

        try:
            r = tco2_rankine_cycle(
                T_heater_in_C=T, P_high_Pa=P_high_tco2, P_low_Pa=5e6,
                eta_turbine=0.85, eta_pump=0.80,
                recuperator_effectiveness=0.70)
            tco2_effs.append(r["eta_th_pct"])
            tco2_bwrs.append(r["bwr_pct"])
            e3, b3 = r["eta_th_pct"], r["bwr_pct"]
        except Exception as ex:
            print(f"  tCO2 error at {T}°C: {ex}")
            tco2_effs.append(None); tco2_bwrs.append(None)
            e3, b3 = None, None

        print(f"{T:>4}°C  {P_high_tco2/1e6:>8.1f}MPa  "
              f"{str(e):>6}%  {str(e2):>6}%  {str(e3):>6}%  "
              f"{str(b):>7}%  {str(b2):>8}%  {str(b3):>8}%")

    temps = list(temperatures)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = {"SRC": "#51247A", "sCO2": "#0064B4", "tCO2": "#BB9B1C"}

    ax1.plot(temps, src_effs,  color=colors["SRC"],  lw=2.5, marker="o", ms=4,
             label="Steam Rankine Cycle (SRC)")
    ax1.plot(temps, sco2_effs, color=colors["sCO2"], lw=2.5, marker="s", ms=4,
             label="sCO₂ Brayton (recuperated, ε=0.85)")
    ax1.plot(temps, tco2_effs, color=colors["tCO2"], lw=2.5, marker="^", ms=5,
             label="tCO₂ Rankine (recuperated, ε=0.70, P_high scaled)")

    t_arr    = np.array(temps, dtype=float)
    src_arr  = np.array([e if e else np.nan for e in src_effs])
    sco2_arr = np.array([e if e else np.nan for e in sco2_effs])
    tco2_arr = np.array([e if e else np.nan for e in tco2_effs])

    ax1.fill_between(t_arr, src_arr, tco2_arr, where=(tco2_arr > src_arr),
                     alpha=0.15, color=colors["tCO2"], label="tCO₂ beats SRC")
    ax1.fill_between(t_arr, src_arr, sco2_arr, where=(sco2_arr > src_arr),
                     alpha=0.12, color=colors["sCO2"], label="sCO₂ beats SRC")

    cases = [(200,"CS2: Gas\n200°C"), (350,"CS1: Cement\n350°C"), (500,"CS3: Steel\n500°C")]
    for T_cs, label in cases:
        ax1.axvline(T_cs, color="gray", linestyle="--", alpha=0.45, lw=1.2)
        ax1.text(T_cs + 3, 1.5, label, fontsize=7.5, color="gray",
                 rotation=90, va="bottom")

    ax1.axvline(374, color="red", linestyle="--", alpha=0.4, lw=1)
    ax1.text(376, 48, "H₂O\ncritical\n374°C", fontsize=7, color="red", alpha=0.7)
    ax1.set_ylabel("Thermal Efficiency (%)", fontsize=12)
    ax1.set_title(
        "RQ1 — Three-Cycle Comparison: SRC vs sCO₂ Brayton vs tCO₂ Rankine\n"
        "WHR Temperature 150–600°C  |  η_turbine=0.85, η_pump/comp=0.80  |  "
        "sCO₂: P=8–20 MPa, ε=0.85  |  tCO₂: P_low=5 MPa, ε=0.70", fontsize=10)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 55)

    ax2.plot(temps, src_bwrs,  color=colors["SRC"],  lw=2.5, marker="o", ms=4, label="SRC BWR")
    ax2.plot(temps, sco2_bwrs, color=colors["sCO2"], lw=2.5, marker="s", ms=4, label="sCO₂ Brayton BWR")
    ax2.plot(temps, tco2_bwrs, color=colors["tCO2"], lw=2.5, marker="^", ms=5, label="tCO₂ Rankine BWR")
    for T_cs, _ in cases:
        ax2.axvline(T_cs, color="gray", linestyle="--", alpha=0.45, lw=1.2)
    ax2.axhline(20, color="#0064B4", linestyle=":", alpha=0.4, lw=1)
    ax2.text(155, 21, "20% reference", fontsize=7.5, color="#0064B4", alpha=0.7)
    ax2.set_xlabel("Turbine Inlet / Heat Source Temperature (°C)", fontsize=12)
    ax2.set_ylabel("Back-Work Ratio (%)", fontsize=12)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 90)
    ax2.set_xlim(148, 602)

    plt.tight_layout()
    plt.savefig("RQ1_THREE_CYCLE_FINAL.png", dpi=150)
    plt.show()
    print("\nSaved: RQ1_THREE_CYCLE_FINAL.png")


# ═══════════════════════════════════════════════════════════════════════════
# PINCH DIAGRAM — CORRECTED FOR TRANSITIONAL EFFECTS
# ═══════════════════════════════════════════════════════════════════════════

def tco2_pinch_diagram(T_source_C, P_high_Pa, P_low_Pa,
                       eta_pump=0.80, T_sink_C=None, n_points=50):
    """
    T-H diagram for the tCO2 supercritical heater.
    T_sink_C: waste gas minimum exit temperature.
              If None, auto-set to pump exit + 10 K (minimum feasible).
    The CO2 pump exit temperature sets a hard lower bound on T_sink.
    """
    fluid = "CO2"
    MIN_APPROACH = 10.0   # minimum ΔT at cold end (K)

    # ── Actual pump outlet temperature ───────────────────────────────
    h1   = CP.PropsSI("H", "P", P_low_Pa,  "Q", 0, fluid)
    s1   = CP.PropsSI("S", "P", P_low_Pa,  "Q", 0, fluid)
    h2s  = CP.PropsSI("H", "P", P_high_Pa, "S", s1, fluid)
    h2_actual    = h1 + (h2s - h1) / eta_pump
    T_pump_out_K = CP.PropsSI("T", "P", P_high_Pa, "H", h2_actual, fluid)
    T_pump_out_C = T_pump_out_K - 273.15

    # ── Enforce feasible T_sink ───────────────────────────────────────
    T_sink_min = T_pump_out_C + MIN_APPROACH
    if T_sink_C is None:
        T_sink_C = T_sink_min
        auto_note = f" (auto: pump exit {T_pump_out_C:.1f}°C + {MIN_APPROACH:.0f} K)"
    elif T_sink_C < T_sink_min:
        print(f"  ⚠ T_sink={T_sink_C:.1f}°C < pump exit {T_pump_out_C:.1f}°C "
              f"+ {MIN_APPROACH:.0f} K — raising to {T_sink_min:.1f}°C")
        T_sink_C = T_sink_min
        auto_note = " (corrected)"
    else:
        auto_note = ""

    # ── CO2 temperature profile: pump exit → turbine inlet ───────────
    T_turbine_K = T_source_C + 273.15 - 10   # 10 K hot-end approach
    T_co2_range = np.linspace(T_pump_out_K, T_turbine_K, n_points)
    h_co2  = [CP.PropsSI("H",      "T", T, "P", P_high_Pa, fluid) / 1000
              for T in T_co2_range]
    cp_co2 = [CP.PropsSI("CPMASS", "T", T, "P", P_high_Pa, fluid) / 1000
              for T in T_co2_range]
    T_co2_C = T_co2_range - 273.15

    # Normalise enthalpy from cold end (x=0 is pump exit)
    h_base     = h_co2[0]
    h_co2_norm = [h - h_base for h in h_co2]
    Q_total    = h_co2_norm[-1]

    # ── Source profile: counter-flow from cold end ────────────────────
    # x=0 → source exits at T_sink (has given up all its heat)
    # x=Q_total → source enters at T_source (has given up nothing yet)
    h_source_norm    = [Q_total * i / (n_points - 1) for i in range(n_points)]
    T_source_profile = [T_sink_C + (T_source_C - T_sink_C) * i / (n_points - 1)
                        for i in range(n_points)]

    # ── Pinch: minimum ΔT = source − CO2 at each cross-section ───────
    temp_diffs = [Ts - Tc for Ts, Tc in zip(T_source_profile, T_co2_C)]
    min_diff   = min(temp_diffs)
    pinch_idx  = temp_diffs.index(min_diff)

    # Pseudo-critical temperature (cp peak)
    T_pc_idx = cp_co2.index(max(cp_co2))
    T_pc     = T_co2_C[T_pc_idx]

    # ── Plot ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # Shade any infeasible region
    t_arr = np.array(h_co2_norm)
    src   = np.array(T_source_profile)
    co2   = np.array(T_co2_C)
    if np.any(src < co2):
        ax1.fill_between(t_arr, co2, src, where=(src < co2),
                         alpha=0.25, color="red",
                         label="Infeasible zone (CO₂ > source)")

    ax1.plot(h_co2_norm, T_co2_C,
             color="#BB9B1C", linewidth=2.5, label="tCO₂ (supercritical side)")
    ax1.plot(h_source_norm, T_source_profile,
             color="#CC0000", linewidth=2.5, linestyle="--",
             label=f"Waste gas source ({T_source_C}°C → {T_sink_C:.0f}°C{auto_note})")

    # Pinch marker
    p_color = "green" if min_diff >= 10 else ("orange" if min_diff >= 5 else "red")
    ax1.scatter([h_co2_norm[pinch_idx]], [T_co2_C[pinch_idx]],
                color=p_color, s=140, zorder=7)
    ax1.scatter([h_source_norm[pinch_idx]], [T_source_profile[pinch_idx]],
                color=p_color, s=140, zorder=7, marker="^")
    offset_x = Q_total * 0.10
    offset_y = 18
    ax1.annotate(f"Pinch ΔT = {min_diff:.1f} K\nT_CO₂ = {T_co2_C[pinch_idx]:.1f}°C",
                 xy=(h_co2_norm[pinch_idx], T_co2_C[pinch_idx]),
                 xytext=(h_co2_norm[pinch_idx] + offset_x,
                         T_co2_C[pinch_idx] + offset_y),
                 arrowprops=dict(arrowstyle="->", color="#51247A"),
                 fontsize=9, color="#51247A",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#51247A", alpha=0.9))

    # Pseudo-critical marker
    ax1.axvline(h_co2_norm[T_pc_idx], color="green",
                linestyle=":", alpha=0.4, linewidth=1.2)
    ax1.scatter([h_co2_norm[T_pc_idx]], [T_pc],
                color="green", s=100, marker="D", zorder=6)
    ax1.annotate(f"Pseudo-critical\nT_pc = {T_pc:.1f}°C",
                 xy=(h_co2_norm[T_pc_idx], T_pc),
                 xytext=(h_co2_norm[T_pc_idx] + Q_total * 0.06, T_pc + 18),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=8.5, color="green")

    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.set_title(
        f"tCO₂ Heater T-H Diagram — Transitional Effects\n"
        f"Source: {T_source_C}°C  |  P_high: {P_high_Pa/1e6:.0f} MPa  |  "
        f"P_low: {P_low_Pa/1e6:.0f} MPa  |  Pinch ΔT: {min_diff:.1f} K",
        fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # cp profile
    ax2.plot(h_co2_norm, cp_co2, color="#BB9B1C", linewidth=2.5)
    ax2.axvline(h_co2_norm[T_pc_idx], color="green",
                linestyle="--", alpha=0.6, linewidth=1.2,
                label=f"Pseudo-critical peak T_pc = {T_pc:.1f}°C")
    ax2.fill_between(h_co2_norm, cp_co2, alpha=0.15, color="#BB9B1C")
    ax2.set_xlabel("Specific Enthalpy Transfer (kJ/kg CO₂)", fontsize=11)
    ax2.set_ylabel("$c_p$ (kJ/kg·K)", fontsize=11)
    ax2.set_title("CO₂ Specific Heat Profile Along Heater — "
                  "Pseudo-Critical Spike Creates Non-Uniform HX Load",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"tco2_pinch_{T_source_C}C.png"
    plt.savefig(fname, dpi=150)
    plt.show()

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"tCO2 PINCH ANALYSIS — Source: {T_source_C}°C")
    print(f"{'='*58}")
    print(f"  CO2 pump exit temperature:  {T_pump_out_C:.1f} °C")
    print(f"  CO2 turbine inlet:          {T_turbine_K - 273.15:.1f} °C")
    print(f"  Waste gas exit (T_sink):    {T_sink_C:.1f} °C{auto_note}")
    print(f"  Pseudo-critical temp:       {T_pc:.1f} °C  (cp peak)")
    print(f"  Pinch point ΔT:             {min_diff:.1f} K")
    print(f"  Pinch location (T_CO2):     {T_co2_C[pinch_idx]:.1f} °C")
    print(f"  Total HX duty:              {Q_total:.1f} kJ/kg CO₂")
    print(f"  Heat NOT recovered below T_sink: waste gas exits at {T_sink_C:.1f}°C")
    if min_diff < 0:
        print(f"  ✗ INFEASIBLE — CO2 hotter than source at pinch")
    elif min_diff < 5:
        print(f"  ⚠ Very tight pinch — extremely large HX area needed")
    elif min_diff < 10:
        print(f"  ⚠ Acceptable but HX area will be substantial")
    else:
        print(f"  ✓ Feasible pinch for preliminary sizing")
    print(f"  Saved: {fname}")
    return min_diff, T_pc, T_sink_C


# ═══════════════════════════════════════════════════════════════════════════
# EXERGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

T0_K  = 298.15
P0_Pa = 101325

def specific_exergy_co2(h_kJkg, s_kJkgK):
    h0 = CP.PropsSI("H", "T", T0_K, "P", P0_Pa, "CO2") / 1000
    s0 = CP.PropsSI("S", "T", T0_K, "P", P0_Pa, "CO2") / 1000
    return (h_kJkg - h0) - T0_K * (s_kJkgK - s0)

def tco2_exergy_analysis(T_heater_C=200, P_high=12e6, P_low=5e6,
                          eta_t=0.85, eta_p=0.80, eps=0.0):
    r = tco2_rankine_cycle(T_heater_C, P_high, P_low, eta_t, eta_p, eps)
    s1, s2, s3, s4 = r["state_1"], r["state_2"], r["state_3"], r["state_4"]

    ex1 = specific_exergy_co2(s1["h"], s1["s"])
    ex2 = specific_exergy_co2(s2["h"], s2["s"])
    ex3 = specific_exergy_co2(s3["h"], s3["s"])
    ex4 = specific_exergy_co2(s4["h"], s4["s"])

    ex_dest_pump = T0_K * (s2["s"] - s1["s"])
    T_source_K   = T_heater_C + 273.15
    q_in         = r["q_in_kJ_kg"]
    ex_heat_in   = q_in * (1 - T0_K / T_source_K)
    ex_dest_hx   = ex_heat_in - (ex3 - ex2)
    ex_dest_turb = T0_K * (s4["s"] - s3["s"])
    ex_dest_cond = ex4 - ex1

    total_dest = ex_dest_pump + ex_dest_hx + ex_dest_turb + ex_dest_cond
    w_net      = r["w_net_kJ_kg"]
    ex_eff     = w_net / ex_heat_in * 100

    return {
        "ex_heat_in":      round(ex_heat_in, 2),
        "ex_dest_pump":    round(ex_dest_pump, 2),
        "ex_dest_hx":      round(ex_dest_hx, 2),
        "ex_dest_turb":    round(ex_dest_turb, 2),
        "ex_dest_cond":    round(ex_dest_cond, 2),
        "total_dest":      round(total_dest, 2),
        "w_net":           round(w_net, 2),
        "eta_th_pct":      r["eta_th_pct"],
        "ex_eff_pct":      round(ex_eff, 2),
    }
# ═══════════════════════════════════════════════════════════════════════════
# RECUPERATOR SENSITIVITY SWEEP
# RQ1 extension — how does ε affect η_th, BWR, T_sink, and HX duty?
# ═══════════════════════════════════════════════════════════════════════════

def recuperator_sensitivity_sweep():
    """
    Sweep recuperator effectiveness ε = 0, 0.50, 0.70, 0.85 for the
    tCO2 Rankine cycle at each Australian case study temperature.

    Also runs the T-H pinch analysis at each ε to show how the pinch
    location and T_sink constraint shift with recuperation.

    Records:
      - Thermal efficiency
      - BWR
      - Net work
      - Heat input from external source
      - Recuperator duty Q_recup
      - T_sink lower bound (from pinch analysis)
      - Pinch location T_CO2
    """
    fluid = "CO2"
    MIN_APPROACH = 10.0

    # Case studies: (label, T_source_C, P_high_Pa)
    case_studies = [
        ("CS2: Gas 200°C",    200, 12.8e6),
        ("CS1: Cement 350°C", 350, 20.0e6),
        ("CS3: Steel 500°C",  500, 20.0e6),
    ]
    eps_values = [0.0, 0.50, 0.70, 0.85]
    P_low_Pa   = 5e6
    eta_t      = 0.85
    eta_p      = 0.80

    # ── Storage for plotting ──────────────────────────────────────────
    results = {cs[0]: {
        "eps":      [],
        "eta_th":   [],
        "bwr":      [],
        "w_net":    [],
        "q_in":     [],
        "q_recup":  [],
        "T_sink":   [],
        "pinch_T":  [],
        "pinch_dT": [],
    } for cs in case_studies}

    print("\n" + "="*72)
    print("RECUPERATOR SENSITIVITY SWEEP — tCO2 Rankine")
    print("="*72)

    for label, T_src, P_hi in case_studies:
        print(f"\n{'─'*72}")
        print(f"  {label}   P_high = {P_hi/1e6:.1f} MPa")
        print(f"{'─'*72}")
        print(f"  {'ε':>6}  {'η_th':>7}  {'BWR':>7}  {'w_net':>8}  "
              f"{'q_in':>8}  {'Q_rec':>8}  {'T_sink':>8}  "
              f"{'Pinch T_CO2':>12}  {'Pinch ΔT':>9}")
        print(f"  {'':>6}  {'(%)':>7}  {'(%)':>7}  {'(kJ/kg)':>8}  "
              f"{'(kJ/kg)':>8}  {'(kJ/kg)':>8}  {'(°C)':>8}  "
              f"{'(°C)':>12}  {'(K)':>9}")
        print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*8}  "
              f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*9}")

        for eps in eps_values:
            # ── Cycle performance ─────────────────────────────────────
            r = tco2_rankine_cycle(
                T_heater_in_C             = T_src,
                P_high_Pa                 = P_hi,
                P_low_Pa                  = P_low_Pa,
                eta_turbine               = eta_t,
                eta_pump                  = eta_p,
                recuperator_effectiveness = eps,
            )

            # ── T_sink lower bound from pump exit ─────────────────────
            h1 = CP.PropsSI("H", "P", P_low_Pa, "Q", 0, fluid)
            s1 = CP.PropsSI("S", "P", P_low_Pa, "Q", 0, fluid)
            h2s = CP.PropsSI("H", "P", P_hi, "S", s1, fluid)
            h2a = h1 + (h2s - h1) / eta_p
            T_pump_out_C = CP.PropsSI("T", "P", P_hi, "H", h2a, fluid) - 273.15
            T_sink_C = T_pump_out_C + MIN_APPROACH

            # ── Multi-point T-H pinch ──────────────────────────────────
            # CO2 side: after recuperation (heater inlet = s2_final)
            T_heater_in_K = r["state_2"]["T_C"] + 273.15
            T_turbine_K   = T_src + 273.15 - 10.0
            n_pts = 50
            T_co2_range = np.linspace(T_heater_in_K, T_turbine_K, n_pts)
            h_co2 = [CP.PropsSI("H","T",T,"P",P_hi,fluid)/1000
                     for T in T_co2_range]
            T_co2_C = T_co2_range - 273.15
            h_base = h_co2[0]
            h_co2_norm = [h - h_base for h in h_co2]
            Q_total = h_co2_norm[-1]

            # Source side: counter-flow from cold end
            T_heater_inlet_C = r["state_2"]["T_C"]
            T_sink_adjusted  = T_heater_inlet_C + MIN_APPROACH
            h_src = [Q_total * i/(n_pts-1) for i in range(n_pts)]
            T_src_profile = [T_sink_adjusted + (T_src - T_sink_adjusted) * i/(n_pts-1)
                 for i in range(n_pts)]

            diffs    = [Ts - Tc for Ts, Tc in zip(T_src_profile, T_co2_C)]
            min_diff = min(diffs)
            pinch_idx = diffs.index(min_diff)
            pinch_T   = T_co2_C[pinch_idx]

            # ── Store ─────────────────────────────────────────────────
            d = results[label]
            d["eps"].append(eps)
            d["eta_th"].append(r["eta_th_pct"])
            d["bwr"].append(r["bwr_pct"])
            d["w_net"].append(r["w_net_kJ_kg"])
            d["q_in"].append(r["q_in_kJ_kg"])
            d["q_recup"].append(r["Q_recup_kJ_kg"])
            d["T_sink"].append(round(T_sink_C, 1))
            d["pinch_T"].append(round(pinch_T, 1))
            d["pinch_dT"].append(round(min_diff, 1))

            print(f"  {eps:>6.2f}  {r['eta_th_pct']:>7.2f}  "
                  f"{r['bwr_pct']:>7.2f}  {r['w_net_kJ_kg']:>8.2f}  "
                  f"{r['q_in_kJ_kg']:>8.2f}  {r['Q_recup_kJ_kg']:>8.2f}  "
                  f"{T_sink_C:>8.1f}  {pinch_T:>12.1f}  {min_diff:>9.1f}")

    # ── Plot ──────────────────────────────────────────────────────────
    _plot_recuperator_sensitivity(results, eps_values, case_studies)
    return results


def _plot_recuperator_sensitivity(results, eps_values, case_studies):
    """
    Four-panel figure: η_th, BWR, Q_recup, and pinch location
    vs recuperator effectiveness, for all three case studies.
    """
    colors = {
        "CS2: Gas 200°C":    "#BB9B1C",  # gold
        "CS1: Cement 350°C": "#51247A",  # purple
        "CS3: Steel 500°C":  "#0064B4",  # blue
    }
    markers = {"CS2: Gas 200°C": "o",
               "CS1: Cement 350°C": "s",
               "CS3: Steel 500°C": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    panels = [
        ("eta_th",  "Thermal Efficiency (%)",           "η_th vs Recuperator Effectiveness"),
        ("bwr",     "Back-Work Ratio (%)",               "BWR vs Recuperator Effectiveness"),
        ("q_recup", "Recuperator Duty (kJ/kg CO₂)",     "Recuperator Heat Transfer"),
        ("pinch_T", "Pinch Location T_CO₂ (°C)",        "Pinch Location Along Heater"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for label, T_src, _ in case_studies:
            d = results[label]
            ax.plot(d["eps"], d[key],
                    color=colors[label],
                    marker=markers[label],
                    linewidth=2.2, markersize=7,
                    label=label)
        ax.set_xlabel("Recuperator Effectiveness ε", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(eps_values)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5)

    # Annotate baseline (ε=0.70) on η_th panel
    ax0 = axes[0]
    for label, T_src, _ in case_studies:
        d  = results[label]
        i  = d["eps"].index(0.70)
        ax0.annotate(f"  ε=0.70\n  {d['eta_th'][i]:.2f}%",
                     xy=(0.70, d["eta_th"][i]),
                     fontsize=7.5, color=colors[label])

    plt.suptitle(
        "tCO₂ Rankine — Recuperator Effectiveness Sensitivity\n"
        "P_low = 5 MPa  |  η_turbine = 0.85  |  η_pump = 0.80  |"
        "  P_high scaled with TIT",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("RQ1_RECUPERATOR_SENSITIVITY.png", dpi=150)
    plt.show()
    print("\nSaved: RQ1_RECUPERATOR_SENSITIVITY.png")
def plow_sensitivity_sweep():
    """
    Sweep low-side pressure P_low = 3.5, 4.0, 5.0 MPa for the tCO2
    Rankine cycle at each Australian case study temperature.
    Fixed ε = 0.70 (baseline from interim report).

    P_low controls:
      - Condenser temperature (and therefore cooling requirement)
      - BWR (lower P_low → lower pump work → lower BWR)
      - T_sink lower bound
      - η_th (lower condensing pressure → more expansion → higher efficiency)
    """
    fluid = "CO2"
    MIN_APPROACH = 10.0

    case_studies = [
        ("CS2: Gas 200°C",    200, 12.8e6),
        ("CS1: Cement 350°C", 350, 20.0e6),
        ("CS3: Steel 500°C",  500, 20.0e6),
    ]
    plow_values = [3.5e6, 4.0e6, 5.0e6]
    eps  = 0.70
    eta_t = 0.85
    eta_p = 0.80

    results = {cs[0]: {
        "P_low_MPa":  [],
        "T_cond":     [],
        "eta_th":     [],
        "bwr":        [],
        "w_net":      [],
        "q_in":       [],
        "T_sink":     [],
        "pinch_T":    [],
        "pinch_dT":   [],
    } for cs in case_studies}

    print("\n" + "="*76)
    print("P_LOW SENSITIVITY SWEEP — tCO2 Rankine  (ε = 0.70 fixed)")
    print("="*76)

    for label, T_src, P_hi in case_studies:
        print(f"\n{'─'*76}")
        print(f"  {label}   P_high = {P_hi/1e6:.1f} MPa   ε = {eps}")
        print(f"{'─'*76}")
        print(f"  {'P_low':>8}  {'T_cond':>8}  {'η_th':>7}  {'BWR':>7}  "
              f"{'w_net':>8}  {'q_in':>8}  {'T_sink':>8}  "
              f"{'Pinch T':>9}  {'Pinch ΔT':>9}")
        print(f"  {'(MPa)':>8}  {'(°C)':>8}  {'(%)':>7}  {'(%)':>7}  "
              f"{'(kJ/kg)':>8}  {'(kJ/kg)':>8}  {'(°C)':>8}  "
              f"{'(°C)':>9}  {'(K)':>9}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  "
              f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*9}")

        for P_lo in plow_values:

            # ── Check feasibility: P_low must be subcritical ──────────
            if P_lo >= P_CRIT_CO2:
                print(f"  {P_lo/1e6:>8.1f}  SKIP — P_low >= Pcrit")
                continue

            # ── Cycle performance ─────────────────────────────────────
            r = tco2_rankine_cycle(
                T_heater_in_C             = T_src,
                P_high_Pa                 = P_hi,
                P_low_Pa                  = P_lo,
                eta_turbine               = eta_t,
                eta_pump                  = eta_p,
                recuperator_effectiveness = eps,
            )

            # ── T_sink from heater inlet temperature ──────────────────
            T_heater_inlet_C = r["state_2"]["T_C"]
            T_sink_C         = T_heater_inlet_C + MIN_APPROACH

            # ── Multi-point T-H pinch ──────────────────────────────────
            T_heater_in_K = T_heater_inlet_C + 273.15
            T_turbine_K   = T_src + 273.15 - 10.0
            n_pts = 50
            T_co2_range = np.linspace(T_heater_in_K, T_turbine_K, n_pts)
            h_co2 = [CP.PropsSI("H","T",T,"P",P_hi,fluid)/1000
                     for T in T_co2_range]
            T_co2_C = T_co2_range - 273.15
            h_base     = h_co2[0]
            h_co2_norm = [h - h_base for h in h_co2]
            Q_total    = h_co2_norm[-1]

            h_src_norm  = [Q_total * i/(n_pts-1) for i in range(n_pts)]
            T_src_prof  = [T_sink_C + (T_src - T_sink_C) * i/(n_pts-1)
                           for i in range(n_pts)]

            diffs     = [Ts - Tc for Ts, Tc in zip(T_src_prof, T_co2_C)]
            min_diff  = min(diffs)
            pinch_idx = diffs.index(min_diff)
            pinch_T   = T_co2_C[pinch_idx]

            # ── Store ─────────────────────────────────────────────────
            d = results[label]
            d["P_low_MPa"].append(round(P_lo/1e6, 1))
            d["T_cond"].append(r["T_condenser_C"])
            d["eta_th"].append(r["eta_th_pct"])
            d["bwr"].append(r["bwr_pct"])
            d["w_net"].append(r["w_net_kJ_kg"])
            d["q_in"].append(r["q_in_kJ_kg"])
            d["T_sink"].append(round(T_sink_C, 1))
            d["pinch_T"].append(round(pinch_T, 1))
            d["pinch_dT"].append(round(min_diff, 1))

            print(f"  {P_lo/1e6:>8.1f}  {r['T_condenser_C']:>8.2f}  "
                  f"{r['eta_th_pct']:>7.2f}  {r['bwr_pct']:>7.2f}  "
                  f"{r['w_net_kJ_kg']:>8.2f}  {r['q_in_kJ_kg']:>8.2f}  "
                  f"{T_sink_C:>8.1f}  {pinch_T:>9.1f}  {min_diff:>9.1f}")

    _plot_plow_sensitivity(results, plow_values, case_studies)
    return results


def _plot_plow_sensitivity(results, plow_values, case_studies):
    colors  = {"CS2: Gas 200°C":    "#BB9B1C",
               "CS1: Cement 350°C": "#51247A",
               "CS3: Steel 500°C":  "#0064B4"}
    markers = {"CS2: Gas 200°C": "o",
               "CS1: Cement 350°C": "s",
               "CS3: Steel 500°C": "^"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    panels = [
        ("T_cond",   "Condenser Temp (°C)",        "Condensing Temperature vs P_low"),
        ("eta_th",   "Thermal Efficiency (%)",      "η_th vs P_low"),
        ("bwr",      "Back-Work Ratio (%)",          "BWR vs P_low"),
        ("w_net",    "Net Work (kJ/kg)",             "Net Specific Work vs P_low"),
        ("T_sink",   "Waste Gas Exit T_sink (°C)",  "T_sink Lower Bound vs P_low"),
        ("pinch_dT", "Pinch ΔT (K)",                "Pinch ΔT vs P_low"),
    ]

    plow_labels = [f"{p/1e6:.1f}" for p in plow_values]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for label, _, _ in case_studies:
            d = results[label]
            ax.plot(d["P_low_MPa"], d[key],
                    color=colors[label], marker=markers[label],
                    linewidth=2.2, markersize=8, label=label)
        ax.set_xlabel("P_low (MPa)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([3.5, 4.0, 5.0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        "tCO₂ Rankine — Low-Side Pressure Sensitivity  (ε = 0.70 fixed)\n"
        "P_high scaled with TIT  |  η_turbine = 0.85  |  η_pump = 0.80",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("RQ1_PLOW_SENSITIVITY.png", dpi=150)
    plt.show()
    print("\nSaved: RQ1_PLOW_SENSITIVITY.png")
if __name__ == "__main__":

    # ── 1. Single point test ──────────────────────────────────────────
    run_single_point_test()

    # ── 2. Pinch analysis — transitional effects ──────────────────────
    print("\n" + "="*58)
    print("TRANSITIONAL EFFECTS — PINCH ANALYSIS")
    print("="*58)
    for T_src in [200, 350, 500]:
        P_hi = min(20e6, max(12e6, 8e6 + (T_src - 100) * 48000))
        tco2_pinch_diagram(
            T_source_C = T_src,
            P_high_Pa  = P_hi,
            P_low_Pa   = 5e6,
            eta_pump   = 0.80,
            T_sink_C   = None,
        )

    # ── 3. Recuperator sensitivity sweep  ◄─── ADD THIS ──────────────
    recup_results = recuperator_sensitivity_sweep()
# ── 4. P_low sensitivity sweep ────────────────────────────────────
    plow_results = plow_sensitivity_sweep()
    # ── 4. Three-cycle parametric sweep ───────────────────────────────
    print("\n" + "=" * 55)
    print("THREE-CYCLE PARAMETRIC SWEEP")
    print("=" * 55)
    run_three_cycle_comparison()

    # ── 5. Exergy analysis ────────────────────────────────────────────
    ex = tco2_exergy_analysis(T_heater_C=200, P_high=12e6, P_low=5e6)
    print("\n" + "=" * 58)
    print("tCO2 RANKINE EXERGY ANALYSIS — 200°C")
    print("=" * 58)
    print(f"  Exergy input from heat source: {ex['ex_heat_in']:.2f} kJ/kg")
    print(f"\n  Exergy destruction by component:")
    print(f"    Pump:      {ex['ex_dest_pump']:.2f} kJ/kg  "
          f"({ex['ex_dest_pump']/ex['total_dest']*100:.1f}%)")
    print(f"    HX:        {ex['ex_dest_hx']:.2f} kJ/kg  "
          f"({ex['ex_dest_hx']/ex['total_dest']*100:.1f}%)")
    print(f"    Turbine:   {ex['ex_dest_turb']:.2f} kJ/kg  "
          f"({ex['ex_dest_turb']/ex['total_dest']*100:.1f}%)")
    print(f"    Condenser: {ex['ex_dest_cond']:.2f} kJ/kg  "
          f"({ex['ex_dest_cond']/ex['total_dest']*100:.1f}%)")
    print(f"    ─────────────────────────────────────")
    print(f"    Total:     {ex['total_dest']:.2f} kJ/kg")
    print(f"\n  Net work:           {ex['w_net']:.2f} kJ/kg")
    print(f"  Thermal efficiency: {ex['eta_th_pct']} %")
    print(f"  Exergy efficiency:  {ex['ex_eff_pct']} %")
    print(f"\n  Check: W_net + Ex_dest = "
          f"{ex['w_net'] + ex['total_dest']:.2f} kJ/kg "
          f"(should equal {ex['ex_heat_in']:.2f} kJ/kg)")
