import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

# Water critical point
T_CRIT_WATER = CP.PropsSI("Tcrit", "Water")  # 647.1 K
P_CRIT_WATER = CP.PropsSI("Pcrit", "Water")  # 22.06 MPa

def get_props(fluid, T_K=None, P_Pa=None, h_Jkg=None, s_Jkg=None, Q=None):
    if T_K is not None and P_Pa is not None:
        h = CP.PropsSI("H", "T", T_K, "P", P_Pa, fluid)
        s = CP.PropsSI("S", "T", T_K, "P", P_Pa, fluid)
    elif P_Pa is not None and Q is not None:
        h = CP.PropsSI("H", "P", P_Pa, "Q", Q, fluid)
        s = CP.PropsSI("S", "P", P_Pa, "Q", Q, fluid)
        T_K = CP.PropsSI("T", "P", P_Pa, "Q", Q, fluid)
    elif P_Pa is not None and s_Jkg is not None:
        h = CP.PropsSI("H", "P", P_Pa, "S", s_Jkg, fluid)
        s = s_Jkg
        T_K = CP.PropsSI("T", "P", P_Pa, "S", s_Jkg, fluid)
    return {"T_C": T_K - 273.15, "h": h/1000, "s": s/1000}


def steam_rankine_cycle(T_turbine_in_C, P_high_Pa, P_low_Pa,
                        eta_turbine=0.85, eta_pump=0.80):
    fluid = "Water"
    T3_K = T_turbine_in_C + 273.15

    # ── Pressure logic ───────────────────────────────────────────────
    # Below critical temp: cap P_high below saturation pressure at TIT
    # Above critical temp: fluid is supercritical — use P_high directly
    if T3_K < T_CRIT_WATER:
        P_sat_at_TIT = CP.PropsSI("P", "T", T3_K, "Q", 1, fluid)
        if P_high_Pa >= P_sat_at_TIT:
            P_high_Pa = 0.95 * P_sat_at_TIT
    else:
        # Supercritical steam — P_high is valid as long as it's below
        # the pressure CoolProp can handle (cap at 20 MPa for safety)
        P_high_Pa = min(P_high_Pa, 20e6)

    # ── State 1: Saturated liquid at condenser ───────────────────────
    s1 = get_props(fluid, P_Pa=P_low_Pa, Q=0)

    # ── State 2: Pump outlet ─────────────────────────────────────────
    h2s = CP.PropsSI("H", "P", P_high_Pa, "S", s1["s"]*1000, fluid) / 1000
    w_pump_ideal  = h2s - s1["h"]
    w_pump_actual = w_pump_ideal / eta_pump
    h2_actual = s1["h"] + w_pump_actual
    T2 = CP.PropsSI("T", "P", P_high_Pa, "H", h2_actual*1000, fluid) - 273.15
    s2 = {"T_C": T2, "h": h2_actual,
          "s": CP.PropsSI("S","P",P_high_Pa,"H",h2_actual*1000,fluid)/1000}

    # ── State 3: Turbine inlet ───────────────────────────────────────
    s3 = get_props(fluid, T_K=T3_K, P_Pa=P_high_Pa)

    # ── State 4: Turbine outlet ──────────────────────────────────────
    h4s = CP.PropsSI("H", "P", P_low_Pa, "S", s3["s"]*1000, fluid) / 1000
    w_turbine_ideal  = s3["h"] - h4s
    w_turbine_actual = w_turbine_ideal * eta_turbine
    h4_actual = s3["h"] - w_turbine_actual
    T4 = CP.PropsSI("T", "P", P_low_Pa, "H", h4_actual*1000, fluid) - 273.15
    s4 = {"T_C": T4, "h": h4_actual,
          "s": CP.PropsSI("S","P",P_low_Pa,"H",h4_actual*1000,fluid)/1000}

    # ── Performance ──────────────────────────────────────────────────
    q_in   = s3["h"] - s2["h"]
    w_net  = w_turbine_actual - w_pump_actual
    bwr    = w_pump_actual / w_turbine_actual
    eta_th = w_net / q_in
    T_cond = CP.PropsSI("T", "P", P_low_Pa, "Q", 0, fluid) - 273.15

    return {
        "P_high_MPa":      round(P_high_Pa/1e6, 3),
        "supercritical":   T3_K >= T_CRIT_WATER,
        "state_1": s1, "state_2": s2,
        "state_3": s3, "state_4": s4,
        "q_in_kJ_kg":      round(q_in, 2),
        "w_turbine_kJ_kg": round(w_turbine_actual, 2),
        "w_pump_kJ_kg":    round(w_pump_actual, 2),
        "w_net_kJ_kg":     round(w_net, 2),
        "bwr_pct":         round(bwr * 100, 2),
        "eta_th_pct":      round(eta_th * 100, 2),
        "T_condenser_C":   round(T_cond, 2),
    }


# ── Parametric sweep ─────────────────────────────────────────────────
temperatures = range(150, 625, 25)
efficiencies, pressures, bwrs, supercrit_flags = [], [], [], []

print(f"{'TIT':>6} {'P_high':>8} {'η_th':>7} {'BWR':>6}  {'Mode'}")
print("-" * 50)

for T in temperatures:
    try:
        r = steam_rankine_cycle(
            T_turbine_in_C = T,
            P_high_Pa      = 10e6,
            P_low_Pa       = 10000,
            eta_turbine    = 0.85,
            eta_pump       = 0.80,
        )
        efficiencies.append(r["eta_th_pct"])
        pressures.append(r["P_high_MPa"])
        bwrs.append(r["bwr_pct"])
        supercrit_flags.append(r["supercritical"])
        mode = "SUPERCRITICAL" if r["supercritical"] else "subcritical"
        print(f"{T:>5}°C  {r['P_high_MPa']:>6.2f} MPa"
              f"  {r['eta_th_pct']:>5.2f}%  {r['bwr_pct']:>4.2f}%  {mode}")
    except Exception as e:
        print(f"{T:>5}°C  ERROR: {e}")
        efficiencies.append(None)
        pressures.append(None)
        bwrs.append(None)
        supercrit_flags.append(None)

temps = list(temperatures)

# ── Plot ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Split into subcritical and supercritical segments for different colours
temps_sub  = [t for t, sc in zip(temps, supercrit_flags) if sc is False]
effs_sub   = [e for e, sc in zip(efficiencies, supercrit_flags) if sc is False]
temps_sup  = [t for t, sc in zip(temps, supercrit_flags) if sc is True]
effs_sup   = [e for e, sc in zip(efficiencies, supercrit_flags) if sc is True]

ax1.plot(temps_sub, effs_sub, color="#51247A", linewidth=2.5,
         marker="o", markersize=4, label="SRC — subcritical steam")
ax1.plot(temps_sup, effs_sup, color="#BB9B1C", linewidth=2.5,
         marker="s", markersize=4, label="SRC — supercritical steam")
ax1.axvline(374, color="red", linestyle="--", alpha=0.6,
            label="H₂O critical point (374°C)")
ax1.axvline(300, color="gray", linestyle=":", alpha=0.5,
            label="Practical viability threshold (~300°C)")
ax1.set_ylabel("Thermal Efficiency (%)", fontsize=11)
ax1.set_title("Steam Rankine Cycle — Full Parametric Sweep 150–600°C\n"
              "(η_turbine=0.85, η_pump=0.80, P_low=10 kPa)", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 45)

pres_sub = [p for p, sc in zip(pressures, supercrit_flags) if sc is False]
pres_sup = [p for p, sc in zip(pressures, supercrit_flags) if sc is True]
ax2.plot(temps_sub, pres_sub, color="#51247A", linewidth=2.5,
         marker="o", markersize=4)
ax2.plot(temps_sup, pres_sup, color="#BB9B1C", linewidth=2.5,
         marker="s", markersize=4)
ax2.axvline(374, color="red", linestyle="--", alpha=0.6)
ax2.axhline(22.06, color="gray", linestyle=":", alpha=0.5,
            label="H₂O critical pressure (22.06 MPa)")
ax2.set_xlabel("Turbine Inlet Temperature (°C)", fontsize=11)
ax2.set_ylabel("High-side Pressure (MPa)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("SRC_sweep_v3.png", dpi=150)
plt.show()
print("\nPlot saved: SRC_sweep_v3.png")
# ═══════════════════════════════════════════════════════════════════════════
# EXERGY ANALYSIS — CORRECTED
# ═══════════════════════════════════════════════════════════════════════════

T0_K  = 298.15    # dead state: 25°C (ASHRAE standard)
P0_Pa = 101325    # dead state: 101.325 kPa

def specific_exergy(h_kJkg, s_kJkgK, fluid="Water"):
    """
    Specific flow exergy: ex = (h - h0) - T0*(s - s0)
    h0, s0 are properties at dead state (25°C, 101.325 kPa)
    """
    h0 = CP.PropsSI("H", "T", T0_K, "P", P0_Pa, fluid) / 1000
    s0 = CP.PropsSI("S", "T", T0_K, "P", P0_Pa, fluid) / 1000
    return (h_kJkg - h0) - T0_K * (s_kJkgK - s0)


def src_exergy_analysis(T_turbine_C, P_high_Pa=10e6, P_low_Pa=10000,
                        eta_turbine=0.85, eta_pump=0.80):
    """
    Full exergy analysis of the Steam Rankine Cycle.

    Component equations:
      Pump      (adiabatic): Ex_dest = T0 * (s2 - s1)
      Boiler    (heat in):   Ex_dest = Q_in*(1 - T0/T_source) - (ex3 - ex2)
      Turbine   (adiabatic): Ex_dest = T0 * (s4 - s3)
      Condenser (to T0 env): Ex_dest = ex4 - ex1
    """
    fluid = "Water"

    # Run the cycle model to get state points
    r = steam_rankine_cycle(
        T_turbine_in_C = T_turbine_C,
        P_high_Pa      = P_high_Pa,
        P_low_Pa       = P_low_Pa,
        eta_turbine    = eta_turbine,
        eta_pump       = eta_pump,
    )

    s1 = r["state_1"]
    s2 = r["state_2"]
    s3 = r["state_3"]
    s4 = r["state_4"]

    # ── Specific exergy at each state point ───────────────────────────
    ex1 = specific_exergy(s1["h"], s1["s"], fluid)
    ex2 = specific_exergy(s2["h"], s2["s"], fluid)
    ex3 = specific_exergy(s3["h"], s3["s"], fluid)
    ex4 = specific_exergy(s4["h"], s4["s"], fluid)

    # ── Exergy destruction by component ───────────────────────────────

    # Pump: adiabatic — formula T0 * Δs is correct
    ex_dest_pump = T0_K * (s2["s"] - s1["s"])

    # Turbine: adiabatic — formula T0 * Δs is correct
    ex_dest_turbine = T0_K * (s4["s"] - s3["s"])

    # Boiler: heat comes in from hot source at T_source = TIT
    # Exergy of heat input = Q_in * (1 - T0/T_source)
    T_source_K = T_turbine_C + 273.15
    q_in       = r["q_in_kJ_kg"]
    ex_heat_in = q_in * (1 - T0_K / T_source_K)
    ex_dest_boiler = ex_heat_in - (ex3 - ex2)

    # Condenser: rejects heat to environment at T0
    # Exergy of rejected heat = 0 (environment is dead state)
    # All exergy entering condenser is destroyed
    ex_dest_condenser = ex4 - ex1

    # ── Summary ───────────────────────────────────────────────────────
    total_dest = (ex_dest_pump + ex_dest_boiler +
                  ex_dest_turbine + ex_dest_condenser)

    w_net  = r["w_net_kJ_kg"]
    ex_eff = w_net / ex_heat_in * 100   # exergy efficiency

    return {
        "ex1": ex1, "ex2": ex2, "ex3": ex3, "ex4": ex4,
        "ex_heat_in":       round(ex_heat_in, 2),
        "ex_dest_pump":     round(ex_dest_pump, 2),
        "ex_dest_boiler":   round(ex_dest_boiler, 2),
        "ex_dest_turbine":  round(ex_dest_turbine, 2),
        "ex_dest_condenser":round(ex_dest_condenser, 2),
        "total_dest":       round(total_dest, 2),
        "w_net":            round(w_net, 2),
        "eta_th_pct":       r["eta_th_pct"],
        "ex_eff_pct":       round(ex_eff, 2),
    }


# ── Run at 400°C and print results ───────────────────────────────────────
ex = src_exergy_analysis(T_turbine_C=400)

print("\n" + "=" * 58)
print("SRC EXERGY ANALYSIS — 400°C, 10 MPa, P_low=10 kPa")
print("=" * 58)

print(f"\n  Exergy at each state point:")
print(f"    State 1 (condenser out): {ex['ex1']:.2f} kJ/kg")
print(f"    State 2 (pump out):      {ex['ex2']:.2f} kJ/kg")
print(f"    State 3 (turbine in):    {ex['ex3']:.2f} kJ/kg")
print(f"    State 4 (turbine out):   {ex['ex4']:.2f} kJ/kg")

print(f"\n  Exergy input from heat source: {ex['ex_heat_in']:.2f} kJ/kg")

print(f"\n  Exergy destruction by component:")
print(f"    Pump:      {ex['ex_dest_pump']:.2f} kJ/kg  "
      f"({ex['ex_dest_pump']/ex['total_dest']*100:.1f}%)")
print(f"    Boiler:    {ex['ex_dest_boiler']:.2f} kJ/kg  "
      f"({ex['ex_dest_boiler']/ex['total_dest']*100:.1f}%)")
print(f"    Turbine:   {ex['ex_dest_turbine']:.2f} kJ/kg  "
      f"({ex['ex_dest_turbine']/ex['total_dest']*100:.1f}%)")
print(f"    Condenser: {ex['ex_dest_condenser']:.2f} kJ/kg  "
      f"({ex['ex_dest_condenser']/ex['total_dest']*100:.1f}%)")
print(f"    ──────────────────────────────────────")
print(f"    Total:     {ex['total_dest']:.2f} kJ/kg")

print(f"\n  Net work output:    {ex['w_net']:.2f} kJ/kg")
print(f"  Thermal efficiency: {ex['eta_th_pct']} %")
print(f"  Exergy efficiency:  {ex['ex_eff_pct']} %")
print(f"\n  Check: W_net + Ex_dest = {ex['w_net'] + ex['total_dest']:.2f} "
      f"kJ/kg  (should equal {ex['ex_heat_in']:.2f} kJ/kg)")

# ── Bar chart ─────────────────────────────────────────────────────────────
components = {
    "Pump":      ex["ex_dest_pump"],
    "Boiler":    ex["ex_dest_boiler"],
    "Turbine":   ex["ex_dest_turbine"],
    "Condenser": ex["ex_dest_condenser"],
    "Net Work\n(output)": ex["w_net"],
}

fig, ax = plt.subplots(figsize=(9, 5))

labels = list(components.keys())
values = list(components.values())
bar_colors = ["#BB9B1C", "#CC0000", "#51247A", "#0064B4", "#2E8B57"]

bars = ax.bar(labels, values, color=bar_colors,
              width=0.55, edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val:.1f}\n({val/ex['ex_heat_in']*100:.1f}%)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_ylabel("Exergy (kJ/kg)", fontsize=12)
ax.set_title(
    "SRC Exergy Balance by Component\n"
    "TIT=400°C, P_high=10 MPa, η_turbine=0.85  |  "
    f"Total exergy input = {ex['ex_heat_in']:.1f} kJ/kg",
    fontsize=10)
ax.set_ylim(0, max(values) * 1.35)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add a horizontal line showing total exergy input
ax.axhline(ex["ex_heat_in"], color="red", linestyle="--",
           alpha=0.5, linewidth=1.2)
ax.text(4.1, ex["ex_heat_in"] + 3,
        f"Total input\n{ex['ex_heat_in']:.1f} kJ/kg",
        fontsize=8, color="red", alpha=0.7)

plt.tight_layout()
plt.savefig("SRC_exergy_corrected.png", dpi=150)
plt.show()
print("\nSaved: SRC_exergy_corrected.png")
