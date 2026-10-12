"""
Counter-flow heat exchanger integration (RQ1 pinch analysis, RQ2 sizing).

The waste-gas stream is treated as an ideal gas with constant cp, so its
temperature is LINEAR IN HEAT DUTY — not linear in grid index.  The working
fluid temperature is evaluated from real-fluid properties at each of n
equally spaced enthalpy nodes, which resolves the pseudo-critical cp peak.

For a given working-fluid profile the gas outlet temperature is found by
bisection so that the minimum local temperature difference equals the
specified approach (10 K).  This is the maximum heat recovery compatible
with the pinch constraint.  The conductance UA is then accumulated segment
by segment, UA = sum dQ_i / LMTD_i, which is exact for counter-flow with
piecewise-linear profiles.
"""
import numpy as np
import CoolProp.CoolProp as CP
from .sites import CP_GAS, DT_APPROACH

N_NODES = 200


def wf_profile(fluid, P, h_in_J, h_out_J, n=N_NODES):
    h = np.linspace(h_in_J, h_out_J, n)
    T = np.array([CP.PropsSI("T", "P", P, "H", hh, fluid) for hh in h]) - 273.15
    cp = np.array([_cp_safe(fluid, P, hh) for hh in h])
    return h, T, cp


def _cp_safe(fluid, P, h):
    try:
        return CP.PropsSI("CPMASS", "P", P, "H", h, fluid) / 1000.0
    except ValueError:            # inside the two-phase dome cp is undefined
        return np.nan


def _seg_lmtd(a, b):
    if a <= 0 or b <= 0:
        return np.nan
    return a if abs(a - b) < 1e-9 else (a - b) / np.log(a / b)


def gas_outlet_for_approach(T_wf, q_frac, T_src, dT_min=DT_APPROACH):
    """Bisection on the gas outlet temperature so that min(T_gas - T_wf) = dT_min."""
    # the hot terminal is fixed at T_src, so exclude it from the test (it is
    # already at the approach when TIT = T_src - dT_min)
    lo, hi = T_wf[0] + dT_min - 200.0, T_src
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        T_gas = mid + (T_src - mid) * q_frac
        if (T_gas - T_wf)[:-1].min() < dT_min - 1e-9:
            lo = mid
        else:
            hi = mid
    T_out = hi
    T_gas = T_out + (T_src - T_out) * q_frac
    T_gas[-1] = T_src
    return T_out, T_gas


def integrate_heater(fluid, P, h_in_J, h_out_J, T_src, dT_min=DT_APPROACH, n=N_NODES):
    """
    Size the primary heater of one cycle against a waste-gas stream.
    Returns per kg/s of working fluid: UA (W/K), gas outlet temperature,
    effective LMTD, the T-h profiles and the location(s) of the pinch.
    """
    h, T_wf, cp = wf_profile(fluid, P, h_in_J, h_out_J, n)
    q = h - h[0]
    Q = q[-1]
    q_frac = q / Q
    T_out, T_gas = gas_outlet_for_approach(T_wf, q_frac, T_src, dT_min)
    dT = T_gas - T_wf
    UA = 0.0
    for i in range(n - 1):
        L = _seg_lmtd(dT[i], dT[i + 1])
        UA += (h[i + 1] - h[i]) / L
    # two-point LMTD using terminal temperatures only
    L2 = _seg_lmtd(dT[0], dT[-1])
    return dict(UA=UA, UA_2pt=Q / L2, Q=Q / 1e3, T_gas_out=T_out,
                LMTD_eff=Q / UA, LMTD_2pt=L2, h_kJ=h / 1e3, q_kJ=q / 1e3,
                T_wf=T_wf, T_gas=T_gas, dT=dT, cp=cp,
                dT_min=dT.min(), dT_max=dT.max(),
                T_wf_at_dTmax=T_wf[dT.argmax()], T_wf_at_dTmin=T_wf[dT.argmin()],
                gas_per_wf=(Q / 1e3) / (CP_GAS * max(T_src - T_out, 1e-9)))   # kg gas per kg wf


def heater_for_cycle(c, T_src, dT_min=DT_APPROACH, n=N_NODES):
    """Primary heater of a cycle dict (from cycles.py) against source T_src."""
    return integrate_heater(c["fluid"], c["P_high"], c["state_hx_in"]["h"] * 1e3,
                            c["state_3"]["h"] * 1e3, T_src, dT_min, n)


def recuperator_UA(c, n=N_NODES):
    """
    Counter-flow recuperator conductance per kg/s, with real properties on
    both sides.  Hot side: turbine exhaust (P_low) cooled to the cooler
    inlet; cold side: pump/compressor exit (P_high) heated to the heater
    inlet.  Equal duty on both sides at each node.
    """
    if c["q_recup"] <= 0:
        return dict(UA=0.0, dT_min=np.nan)
    f = c["fluid"]
    Q = c["q_recup"] * 1e3
    q = np.linspace(0, Q, n)
    h_c = c["state_2"]["h"] * 1e3 + q                 # cold side, from its inlet
    h_h = c["state_cool_in"]["h"] * 1e3 + q           # hot side, from its OUTLET (same end)
    T_c = np.array([CP.PropsSI("T", "P", c["P_high"], "H", x, f) for x in h_c]) - 273.15
    T_h = np.array([CP.PropsSI("T", "P", c["P_low"], "H", x, f) for x in h_h]) - 273.15
    dT = T_h - T_c
    UA = 0.0
    for i in range(n - 1):
        L = _seg_lmtd(dT[i], dT[i + 1])
        UA += (q[i + 1] - q[i]) / L
    return dict(UA=UA, dT_min=dT.min(), dT_max=dT.max(), LMTD_eff=Q / UA,
                T_cold=T_c, T_hot=T_h, q_kJ=q / 1e3)


def cooler_UA(c, dT_mean=DT_APPROACH):
    """Heat-rejection exchanger conductance per kg/s with an assumed mean
    temperature difference to the cooling medium (W/K)."""
    return c["q_out"] * 1e3 / dT_mean


def pinch_report(res):
    """Human-readable list of the locations where the approach is reached."""
    dT, T = res["dT"], res["T_wf"]
    at = dT <= res["dT_min"] + 0.05
    locs = []
    i = 0
    while i < len(at):
        if at[i]:
            j = i
            while j + 1 < len(at) and at[j + 1]:
                j += 1
            if i == 0:
                locs.append("cold terminal")
            elif j == len(at) - 1:
                locs.append("hot terminal")
            else:
                locs.append(f"interior at T_wf = {T[i]:.0f}–{T[j]:.0f} °C")
            i = j + 1
        else:
            i += 1
    return "; ".join(locs)
