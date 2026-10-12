import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

def pseudo_critical_profile(P_Pa, T_start_C, T_end_C, n_points=50):
    """
    Compute CO2 properties along the heater at fixed pressure.
    Shows how properties transition through the pseudo-critical region.
    """
    T_range = np.linspace(T_start_C + 273.15, T_end_C + 273.15, n_points)
    
    cp_vals   = []
    rho_vals  = []
    h_vals    = []
    visc_vals = []
    
    for T in T_range:
        try:
            cp   = CP.PropsSI("CPMASS", "T", T, "P", P_Pa, "CO2") / 1000
            rho  = CP.PropsSI("D",      "T", T, "P", P_Pa, "CO2")
            h    = CP.PropsSI("H",      "T", T, "P", P_Pa, "CO2") / 1000
            visc = CP.PropsSI("VISCOSITY","T",T,"P",P_Pa,"CO2") * 1e6
            cp_vals.append(cp)
            rho_vals.append(rho)
            h_vals.append(h)
            visc_vals.append(visc)
        except:
            cp_vals.append(None)
            rho_vals.append(None)
            h_vals.append(None)
            visc_vals.append(None)
    
    T_C = T_range - 273.15
    
    # Find pseudo-critical temperature (where cp peaks)
    valid = [(t, c) for t, c in zip(T_C, cp_vals) if c is not None]
    T_pc  = max(valid, key=lambda x: x[1])[0]
    
    return T_C, cp_vals, rho_vals, h_vals, T_pc


# ── Run for three pressures relevant to your cycles ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

pressures = [
    (8e6,  "sCO2 low-side (8 MPa)",  "#0064B4"),
    (12e6, "tCO2 low P_high (12 MPa)", "#BB9B1C"),
    (20e6, "sCO2/tCO2 high-side (20 MPa)", "#51247A"),
]

for ax, (P, label, color) in zip(axes, pressures):
    T_C, cp_vals, rho_vals, h_vals, T_pc = pseudo_critical_profile(
        P, T_start_C=20, T_end_C=120, n_points=80)
    
    ax.plot(T_C, cp_vals, color=color, linewidth=2.5)
    ax.axvline(T_pc, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
    ax.text(T_pc + 1, max(c for c in cp_vals if c), 
            f"$T_{{pc}}$ = {T_pc:.1f}°C", fontsize=8, color="red")
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("$c_p$ (kJ/kg·K)", fontsize=11)
    ax.set_title(label, fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle("CO₂ Specific Heat Profiles — Pseudo-Critical Transitions\n"
             "Transitional effects neglected in steady-state model",
             fontsize=11)
plt.tight_layout()
plt.savefig("pseudo_critical_profiles.png", dpi=150)
plt.show()
print("Saved: pseudo_critical_profiles.png")
