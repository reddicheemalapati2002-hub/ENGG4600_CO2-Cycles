import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

# CO2 critical point
T_CRIT_CO2 = CP.PropsSI("Tcrit", "CO2")   # 304.13 K = 30.98°C
P_CRIT_CO2 = CP.PropsSI("Pcrit", "CO2")   # 7.377 MPa

def get_props_co2(T_K, P_Pa):
    """Get h and s for CO2 at given T and P."""
    h = CP.PropsSI("H", "T", T_K, "P", P_Pa, "CO2")
    s = CP.PropsSI("S", "T", T_K, "P", P_Pa, "CO2")
    return {"T_C": T_K - 273.15, "h": h/1000, "s": s/1000}


def sco2_brayton_cycle(T_heater_in_C, P_high_Pa, P_low_Pa,
                       eta_turbine=0.85, eta_compressor=0.80,
                       recuperator_effectiveness=0.85):
    """
    Recuperated sCO2 Brayton cycle.

    All states stay ABOVE critical pressure (supercritical throughout).
    No phase change occurs anywhere.

    State points (matching your proposal schematic):
      1: Cooler outlet (near critical point, high density)
      2: Compressor outlet
      3: Recuperator hot-side outlet → Primary heater inlet
      3h: Primary heater outlet = Turbine inlet  (called state 3 in proposal)
      4: Turbine outlet (hot, high entropy)
      5: Recuperator cold-side outlet → Cooler inlet

    Returns dict of results.
    """
    fluid = "CO2"

    # ── Safety checks ────────────────────────────────────────────────
    # Both pressures must be supercritical
    assert P_low_Pa  > P_CRIT_CO2, \
        f"P_low={P_low_Pa/1e6:.2f} MPa must be > Pcrit={P_CRIT_CO2/1e6:.3f} MPa"
    assert P_high_Pa > P_low_Pa, \
        "P_high must be greater than P_low"

    T_heater_in_K = T_heater_in_C + 273.15

    # ── State 1: Cooler outlet ───────────────────────────────────────
    # Operating just above critical point for maximum density benefit
    # Typically T1 ~ 32-35°C, P1 = P_low
    T1_K = T_CRIT_CO2 + 5   # 5 K above critical = ~36°C
    s1 = get_props_co2(T1_K, P_low_Pa)

    # ── State 2: Compressor outlet (isentropic + efficiency) ─────────
    # Isentropic compression from P_low to P_high
    h2s = CP.PropsSI("H", "P", P_high_Pa, "S", s1["s"]*1000, fluid) / 1000
    w_comp_ideal  = h2s - s1["h"]           # kJ/kg, always positive
    w_comp_actual = w_comp_ideal / eta_compressor
    h2_actual = s1["h"] + w_comp_actual
    T2_K = CP.PropsSI("T", "P", P_high_Pa, "H", h2_actual*1000, fluid)
    s2 = {"T_C": T2_K - 273.15, "h": h2_actual,
          "s": CP.PropsSI("S","P",P_high_Pa,"H",h2_actual*1000,fluid)/1000}

    # ── State 3h: Turbine inlet = Primary heater outlet ──────────────
    s3h = get_props_co2(T_heater_in_K, P_high_Pa)

    # ── State 4: Turbine outlet (isentropic + efficiency) ────────────
    h4s = CP.PropsSI("H", "P", P_low_Pa, "S", s3h["s"]*1000, fluid) / 1000
    w_turb_ideal  = s3h["h"] - h4s
    w_turb_actual = w_turb_ideal * eta_turbine
    h4_actual = s3h["h"] - w_turb_actual
    T4_K = CP.PropsSI("T", "P", P_low_Pa, "H", h4_actual*1000, fluid)
    s4 = {"T_C": T4_K - 273.15, "h": h4_actual,
          "s": CP.PropsSI("S","P",P_low_Pa,"H",h4_actual*1000,fluid)/1000}

    # ── Recuperator ──────────────────────────────────────────────────
    # Hot side: turbine outlet (state 4) → cooler inlet (state 5)
    # Cold side: compressor outlet (state 2) → heater inlet (state 3)
    # Effectiveness ε = Q_actual / Q_max
    # Q_max limited by the stream with smaller capacity (usually cold side)

    # Max heat recoverable (cold side limited — from T2 up to T4)
    h_cold_max = CP.PropsSI("H", "T", T4_K, "P", P_high_Pa, fluid) / 1000
    Q_max = h_cold_max - s2["h"]

    # Actual heat transferred
    Q_recup = recuperator_effectiveness * Q_max

    # State 3: heater inlet (after recuperator, cold side)
    h3_actual = s2["h"] + Q_recup
    T3_K = CP.PropsSI("T", "P", P_high_Pa, "H", h3_actual*1000, fluid)
    s3 = {"T_C": T3_K - 273.15, "h": h3_actual,
          "s": CP.PropsSI("S","P",P_high_Pa,"H",h3_actual*1000,fluid)/1000}

    # State 5: cooler inlet (after recuperator, hot side)
    h5_actual = s4["h"] - Q_recup
    T5_K = CP.PropsSI("T", "P", P_low_Pa, "H", h5_actual*1000, fluid)
    s5 = {"T_C": T5_K - 273.15, "h": h5_actual,
          "s": CP.PropsSI("S","P",P_low_Pa,"H",h5_actual*1000,fluid)/1000}

    # ── Performance ──────────────────────────────────────────────────
    q_in   = s3h["h"] - s3["h"]         # heat added in primary heater
    w_net  = w_turb_actual - w_comp_actual
    bwr    = w_comp_actual / w_turb_actual
    eta_th = w_net / q_in

    return {
        "state_1":  s1,    # cooler outlet
        "state_2":  s2,    # compressor outlet
        "state_3":  s3,    # recuperator cold-side outlet / heater inlet
        "state_3h": s3h,   # turbine inlet / heater outlet
        "state_4":  s4,    # turbine outlet
        "state_5":  s5,    # recuperator hot-side outlet / cooler inlet
        "Q_recup_kJ_kg":   round(Q_recup, 2),
        "q_in_kJ_kg":      round(q_in, 2),
        "w_turbine_kJ_kg": round(w_turb_actual, 2),
        "w_comp_kJ_kg":    round(w_comp_actual, 2),
        "w_net_kJ_kg":     round(w_net, 2),
        "bwr_pct":         round(bwr * 100, 2),
        "eta_th_pct":      round(eta_th * 100, 2),
        "P_high_MPa":      round(P_high_Pa/1e6, 2),
        "P_low_MPa":       round(P_low_Pa/1e6, 2),
    }


# ── Single point test first ──────────────────────────────────────────
if __name__ == "__main__":

    r = sco2_brayton_cycle(
        T_heater_in_C          = 550,    # turbine inlet temperature
        P_high_Pa              = 20e6,   # 20 MPa high side
        P_low_Pa               = 8e6,    # 8 MPa low side (supercritical)
        eta_turbine            = 0.85,
        eta_compressor         = 0.80,
        recuperator_effectiveness = 0.85,
    )

    print("=" * 55)
    print("sCO2 BRAYTON CYCLE (Recuperated) — State Points")
    print("=" * 55)
    labels = {
        "state_1":  "Cooler outlet (near critical)",
        "state_2":  "Compressor outlet",
        "state_3":  "Heater inlet (after recuperator)",
        "state_3h": "Turbine inlet (heater outlet)",
        "state_4":  "Turbine outlet",
        "state_5":  "Cooler inlet (after recuperator)",
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
    print(f"  Recuperator heat:  {r['Q_recup_kJ_kg']} kJ/kg")
    print(f"  Heat input:        {r['q_in_kJ_kg']} kJ/kg")
    print(f"  Turbine work:      {r['w_turbine_kJ_kg']} kJ/kg")
    print(f"  Compressor work:   {r['w_comp_kJ_kg']} kJ/kg")
    print(f"  Net work:          {r['w_net_kJ_kg']} kJ/kg")
    print(f"  Back-work ratio:   {r['bwr_pct']} %")
    print(f"  Thermal efficiency:{r['eta_th_pct']} %")
# ── Add this to the bottom of sco2_model.py ──────────────────────────────

def src_efficiency_for_comparison(T_C, P_high_Pa=10e6, P_low_Pa=10000,
                                   eta_t=0.85, eta_p=0.80):
    """Inline SRC model for comparison plotting."""
    fluid = "Water"
    T_CRIT_W = CP.PropsSI("Tcrit", "Water")
    T_K = T_C + 273.15

    if T_K < T_CRIT_W:
        P_sat = CP.PropsSI("P", "T", T_K, "Q", 1, fluid)
        if P_high_Pa >= P_sat:
            P_high_Pa = 0.95 * P_sat
    else:
        P_high_Pa = min(P_high_Pa, 20e6)

    s1h = CP.PropsSI("H","P",P_low_Pa,"Q",0,fluid)
    s1s = CP.PropsSI("S","P",P_low_Pa,"Q",0,fluid)

    h2s = CP.PropsSI("H","P",P_high_Pa,"S",s1s,fluid)
    w_p = (h2s - s1h) / eta_p
    h2  = s1h + w_p

    h3 = CP.PropsSI("H","T",T_K,"P",P_high_Pa,fluid)
    s3 = CP.PropsSI("S","T",T_K,"P",P_high_Pa,fluid)

    h4s = CP.PropsSI("H","P",P_low_Pa,"S",s3,fluid)
    w_t = (h3 - h4s) * eta_t

    q_in   = (h3 - h2) / 1000
    w_net  = (w_t - w_p) / 1000
    return round(w_net / q_in * 100, 2)


# ── Parametric sweep: both cycles, 150–600°C ─────────────────────────────
temperatures = range(150, 625, 25)

sco2_effs, sco2_bwrs = [], []
src_effs = []

print(f"\n{'TIT':>6}  {'sCO2 η_th':>10}  {'sCO2 BWR':>9}  {'SRC η_th':>9}  {'Δη':>7}")
print("-" * 55)

for T in temperatures:
    # sCO2 Brayton
    try:
        r = sco2_brayton_cycle(
            T_heater_in_C             = T,
            P_high_Pa                 = 20e6,
            P_low_Pa                  = 8e6,
            eta_turbine               = 0.85,
            eta_compressor            = 0.80,
            recuperator_effectiveness = 0.85,
        )
        sco2_effs.append(r["eta_th_pct"])
        sco2_bwrs.append(r["bwr_pct"])
    except Exception as e:
        sco2_effs.append(None)
        sco2_bwrs.append(None)

    # SRC
    try:
        e = src_efficiency_for_comparison(T)
        src_effs.append(e)
    except:
        src_effs.append(None)

    # Print comparison
    s = sco2_effs[-1]
    w = src_effs[-1]
    delta = f"{s - w:+.2f}%" if s and w else "  N/A"
    print(f"{T:>5}°C  {str(s):>9}%  {str(sco2_bwrs[-1]):>8}%  "
          f"{str(w):>8}%  {delta:>7}")

temps = list(temperatures)

# ── THE KEY CHART: RQ1 Comparison ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# ── Top panel: efficiency comparison ─────────────────────────────────────
ax1.plot(temps, src_effs,   color="#51247A", linewidth=2.5,
         marker="o", markersize=4, label="Steam Rankine Cycle (SRC)")
ax1.plot(temps, sco2_effs,  color="#0064B4", linewidth=2.5,
         marker="s", markersize=4, label="sCO₂ Brayton (recuperated, ε=0.85)")

# Shade the region where sCO2 beats SRC
t_arr    = np.array(temps, dtype=float)
src_arr  = np.array([e if e else np.nan for e in src_effs])
sco2_arr = np.array([e if e else np.nan for e in sco2_effs])
ax1.fill_between(t_arr, src_arr, sco2_arr,
                 where=(sco2_arr > src_arr),
                 alpha=0.15, color="#0064B4",
                 label="sCO₂ advantage zone")

# Reference lines
ax1.axvline(300, color="gray",  linestyle=":", alpha=0.6,
            label="SRC viability limit (~300°C)")
ax1.axvline(374, color="red",   linestyle="--", alpha=0.5,
            label="H₂O critical point (374°C)")

ax1.set_ylabel("Thermal Efficiency (%)", fontsize=12)
ax1.set_title("RQ1: Thermal Efficiency Comparison — SRC vs sCO₂ Brayton\n"
              "150–600°C WHR Temperature Range  |  "
              "η_turbine=0.85, η_comp=0.80, P_low(sCO₂)=8 MPa, P_high(sCO₂)=20 MPa",
              fontsize=10)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 55)

# Annotate the crossover point
for i, (t, s, w) in enumerate(zip(temps, sco2_effs, src_effs)):
    if s and w and s > w and (i == 0 or sco2_effs[i-1] <= src_effs[i-1]):
        ax1.annotate(f"sCO₂ > SRC\nfrom ~{t}°C",
                     xy=(t, s), xytext=(t+30, s-8),
                     arrowprops=dict(arrowstyle="->", color="#0064B4"),
                     fontsize=9, color="#0064B4")

# ── Bottom panel: BWR comparison ─────────────────────────────────────────
src_bwrs = []
for T in temperatures:
    try:
        fluid = "Water"
        T_K = T + 273.15
        T_CRIT_W = CP.PropsSI("Tcrit","Water")
        P_h = 10e6
        if T_K < T_CRIT_W:
            Psat = CP.PropsSI("P","T",T_K,"Q",1,fluid)
            if P_h >= Psat: P_h = 0.95*Psat
        else:
            P_h = min(P_h, 20e6)
        s1s = CP.PropsSI("S","P",10000,"Q",0,fluid)
        h2s = CP.PropsSI("H","P",P_h,"S",s1s,fluid)
        h1  = CP.PropsSI("H","P",10000,"Q",0,fluid)
        h3  = CP.PropsSI("H","T",T_K,"P",P_h,fluid)
        s3  = CP.PropsSI("S","T",T_K,"P",P_h,fluid)
        h4s = CP.PropsSI("H","P",10000,"S",s3,fluid)
        w_p = (h2s-h1)/0.80
        w_t = (h3-h4s)*0.85
        src_bwrs.append(round(w_p/w_t*100, 2))
    except:
        src_bwrs.append(None)

ax2.plot(temps, src_bwrs,  color="#51247A", linewidth=2.5,
         marker="o", markersize=4, label="SRC BWR")
ax2.plot(temps, sco2_bwrs, color="#0064B4", linewidth=2.5,
         marker="s", markersize=4, label="sCO₂ Brayton BWR")
ax2.axhline(20, color="#0064B4", linestyle=":", alpha=0.5,
            label="20% BWR reference")
ax2.set_xlabel("Turbine Inlet / Heat Source Temperature (°C)", fontsize=12)
ax2.set_ylabel("Back-Work Ratio (%)", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 45)

plt.tight_layout()
plt.savefig("RQ1_SRC_vs_sCO2_comparison.png", dpi=150)
plt.show()
print("\nChart saved: RQ1_SRC_vs_sCO2_comparison.png")
print("This is your first RQ1 result figure.")
# ═══════════════════════════════════════════════════════════════════════════
# EXERGY ANALYSIS — sCO2 BRAYTON
# ═══════════════════════════════════════════════════════════════════════════

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

T0_K  = 298.15
P0_Pa = 101325

def specific_exergy_co2(h_kJkg, s_kJkgK):
    h0 = CP.PropsSI("H", "T", T0_K, "P", P0_Pa, "CO2") / 1000
    s0 = CP.PropsSI("S", "T", T0_K, "P", P0_Pa, "CO2") / 1000
    return (h_kJkg - h0) - T0_K * (s_kJkgK - s0)


def sco2_exergy_analysis(T_heater_C=550, P_high=20e6, P_low=8e6,
                          eta_t=0.85, eta_c=0.80, eps=0.85):
    """
    Exergy analysis for recuperated sCO2 Brayton cycle.

    Components:
      Compressor  (adiabatic): Ex_dest = T0 * (s2 - s1)
      Recuperator (adiabatic): Ex_dest = T0 * (s3 + s5 - s2 - s4)
      Heater      (heat in):   Ex_dest = Q_in*(1-T0/T_source) - (ex3h - ex3)
      Turbine     (adiabatic): Ex_dest = T0 * (s4 - s3h)
      Cooler      (to T0):     Ex_dest = ex5 - ex1
    """
    r = sco2_brayton_cycle(T_heater_C, P_high, P_low,
                           eta_t, eta_c, eps)

    s1  = r["state_1"]   # cooler out
    s2  = r["state_2"]   # compressor out
    s3  = r["state_3"]   # heater in (after recuperator)
    s3h = r["state_3h"]  # turbine in
    s4  = r["state_4"]   # turbine out
    s5  = r["state_5"]   # cooler in (after recuperator)

    ex1  = specific_exergy_co2(s1["h"],  s1["s"])
    ex2  = specific_exergy_co2(s2["h"],  s2["s"])
    ex3  = specific_exergy_co2(s3["h"],  s3["s"])
    ex3h = specific_exergy_co2(s3h["h"], s3h["s"])
    ex4  = specific_exergy_co2(s4["h"],  s4["s"])
    ex5  = specific_exergy_co2(s5["h"],  s5["s"])

    # Exergy destruction per component
    ex_dest_comp   = T0_K * (s2["s"] - s1["s"])
    ex_dest_recup  = T0_K * ((s3["s"] + s5["s"]) - (s2["s"] + s4["s"]))
    T_source_K     = T_heater_C + 273.15
    q_in           = r["q_in_kJ_kg"]
    ex_heat_in     = q_in * (1 - T0_K / T_source_K)
    ex_dest_heater = ex_heat_in - (ex3h - ex3)
    ex_dest_turb   = T0_K * (s4["s"] - s3h["s"])
    ex_dest_cooler = ex5 - ex1

    total_dest = (ex_dest_comp + ex_dest_recup + ex_dest_heater +
                  ex_dest_turb + ex_dest_cooler)
    w_net      = r["w_net_kJ_kg"]
    ex_eff     = w_net / ex_heat_in * 100

    return {
        "ex_heat_in":       round(ex_heat_in, 2),
        "ex_dest_comp":     round(ex_dest_comp, 2),
        "ex_dest_recup":    round(ex_dest_recup, 2),
        "ex_dest_heater":   round(ex_dest_heater, 2),
        "ex_dest_turb":     round(ex_dest_turb, 2),
        "ex_dest_cooler":   round(ex_dest_cooler, 2),
        "total_dest":       round(total_dest, 2),
        "w_net":            round(w_net, 2),
        "eta_th_pct":       r["eta_th_pct"],
        "ex_eff_pct":       round(ex_eff, 2),
    }


# ── Run and print ─────────────────────────────────────────────────────────
ex = sco2_exergy_analysis(T_heater_C=400)

print("\n" + "=" * 58)
print("sCO2 BRAYTON EXERGY ANALYSIS — 400°C")
print("=" * 58)
print(f"  Exergy input from heat source: {ex['ex_heat_in']:.2f} kJ/kg")
print(f"\n  Exergy destruction by component:")
print(f"    Compressor:  {ex['ex_dest_comp']:.2f} kJ/kg  "
      f"({ex['ex_dest_comp']/ex['total_dest']*100:.1f}%)")
print(f"    Recuperator: {ex['ex_dest_recup']:.2f} kJ/kg  "
      f"({ex['ex_dest_recup']/ex['total_dest']*100:.1f}%)")
print(f"    Heater:      {ex['ex_dest_heater']:.2f} kJ/kg  "
      f"({ex['ex_dest_heater']/ex['total_dest']*100:.1f}%)")
print(f"    Turbine:     {ex['ex_dest_turb']:.2f} kJ/kg  "
      f"({ex['ex_dest_turb']/ex['total_dest']*100:.1f}%)")
print(f"    Cooler:      {ex['ex_dest_cooler']:.2f} kJ/kg  "
      f"({ex['ex_dest_cooler']/ex['total_dest']*100:.1f}%)")
print(f"    ─────────────────────────────────────")
print(f"    Total:       {ex['total_dest']:.2f} kJ/kg")
print(f"\n  Net work:           {ex['w_net']:.2f} kJ/kg")
print(f"  Thermal efficiency: {ex['eta_th_pct']} %")
print(f"  Exergy efficiency:  {ex['ex_eff_pct']} %")
print(f"\n  Check: W_net + Ex_dest = "
      f"{ex['w_net'] + ex['total_dest']:.2f} kJ/kg "
      f"(should equal {ex['ex_heat_in']:.2f} kJ/kg)")


# ── THREE-CYCLE EXERGY COMPARISON CHART ──────────────────────────────────
# Hard-code SRC values from your earlier run at 400°C
src_components = {
    "Pump/\nComp":    2.35,
    "HX\n(Boiler)":  375.44,
    "Turbine":        158.54,
    "Cond/\nCooler":  126.85,
    "Recuperator":    0.0,
}
src_total = sum(src_components.values())
src_w_net = 948.49

sco2_components = {
    "Pump/\nComp":    ex["ex_dest_comp"],
    "HX\n(Boiler)":  ex["ex_dest_heater"],
    "Turbine":        ex["ex_dest_turb"],
    "Cond/\nCooler":  ex["ex_dest_cooler"],
    "Recuperator":    ex["ex_dest_recup"],
}
sco2_total = ex["total_dest"]
sco2_w_net = ex["w_net"]

# Labels for grouped bars
categories = ["Pump/\nComp", "HX\n(Boiler)", "Turbine",
              "Cond/\nCooler", "Recuperator"]

src_vals  = [src_components[c]  for c in categories]
sco2_vals = [sco2_components[c] for c in categories]

x     = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, src_vals,  width, label="SRC",
               color="#51247A", alpha=0.85)
bars2 = ax.bar(x + width/2, sco2_vals, width, label="sCO₂ Brayton",
               color="#0064B4", alpha=0.85)

# Value labels
for bar in bars1:
    h = bar.get_height()
    if h > 1:
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f"{h:.0f}", ha="center", va="bottom",
                fontsize=8.5, color="#51247A", fontweight="bold")

for bar in bars2:
    h = bar.get_height()
    if h > 1:
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f"{h:.0f}", ha="center", va="bottom",
                fontsize=8.5, color="#0064B4", fontweight="bold")

ax.set_ylabel("Exergy Destruction (kJ/kg)", fontsize=12)
ax.set_title(
    "Exergy Destruction by Component — SRC vs sCO₂ Brayton\n"
    "TIT = 400°C  |  Same turbine efficiency (η=0.85)  |  "
    "Dead state: 25°C, 101.325 kPa",
    fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add total destroyed annotation
ax.text(0.02, 0.95,
        f"SRC total destroyed:   {src_total:.0f} kJ/kg  "
        f"(η_ex = {src_w_net/(src_w_net+src_total)*100:.1f}%)\n"
        f"sCO₂ total destroyed:  {sco2_total:.0f} kJ/kg  "
        f"(η_ex = {sco2_w_net/(sco2_w_net+sco2_total)*100:.1f}%)",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow",
                  alpha=0.8, edgecolor="gray"))

plt.tight_layout()
plt.savefig("Exergy_SRC_vs_sCO2.png", dpi=150)
plt.show()
print("\nSaved: Exergy_SRC_vs_sCO2.png")
