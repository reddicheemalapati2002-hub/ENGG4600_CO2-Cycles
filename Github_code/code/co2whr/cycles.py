"""
Steady-state cycle models (RQ1).

All three models share the same conventions:
  * states are dicts with T_C, P_Pa, h (kJ/kg), s (kJ/kg K)
  * isentropic efficiencies: turbine 0.85, pump/compressor 0.80
  * zero pressure drop in heat exchangers
  * CoolProp HEOS backend (accurate near the CO2 critical point)

Recuperator effectiveness is defined on the limiting stream,
    Q_recup = eps * min( h_cold(T_hot,in) - h_cold,in ,  h_hot,in - h_hot(T_cold,in) ),
so the second law is never violated inside the recuperator.
"""
import CoolProp.CoolProp as CP

T_CRIT_CO2 = CP.PropsSI("Tcrit", "CO2")      # 304.13 K
P_CRIT_CO2 = CP.PropsSI("Pcrit", "CO2")      # 7.377 MPa
T_CRIT_H2O = CP.PropsSI("Tcrit", "Water")    # 647.10 K
ETA_T, ETA_P = 0.85, 0.80
T0_K, P0_PA = 298.15, 101325.0               # dead state (ASHRAE)


def _state(fluid, P, h):
    return dict(T_C=CP.PropsSI("T", "P", P, "H", h, fluid) - 273.15, P_Pa=P,
                h=h / 1000.0, s=CP.PropsSI("S", "P", P, "H", h, fluid) / 1000.0)


def flow_exergy(fluid, h_kJ, s_kJ):
    """Specific flow exergy ex = (h-h0) - T0 (s-s0), dead state 25 °C, 1 atm."""
    h0 = CP.PropsSI("H", "T", T0_K, "P", P0_PA, fluid) / 1000.0
    s0 = CP.PropsSI("S", "T", T0_K, "P", P0_PA, fluid) / 1000.0
    return (h_kJ - h0) - T0_K * (s_kJ - s0)


# --------------------------------------------------------------------------
# Steam Rankine cycle
# --------------------------------------------------------------------------
def src_sweep_pressure(TIT_C, P_max=10e6):
    """Pressure rule used in the RQ1 parametric sweep: 95 % of saturation
    pressure at TIT (subcritical), capped at 10 MPa (also used above T_crit)."""
    T = TIT_C + 273.15
    if T < T_CRIT_H2O:
        return min(P_max, 0.95 * CP.PropsSI("P", "T", T, "Q", 1, "Water"))
    return P_max


def src_cycle(TIT_C, P_high, P_low=10e3, eta_t=ETA_T, eta_p=ETA_P):
    f = "Water"
    T3 = TIT_C + 273.15
    h1 = CP.PropsSI("H", "P", P_low, "Q", 0, f); s1 = CP.PropsSI("S", "P", P_low, "Q", 0, f)
    h2s = CP.PropsSI("H", "P", P_high, "S", s1, f)
    w_p = (h2s - h1) / eta_p
    h2 = h1 + w_p
    h3 = CP.PropsSI("H", "T", T3, "P", P_high, f); s3 = CP.PropsSI("S", "T", T3, "P", P_high, f)
    h4s = CP.PropsSI("H", "P", P_low, "S", s3, f)
    w_t = (h3 - h4s) * eta_t
    h4 = h3 - w_t
    q_in = h3 - h2
    return dict(cycle="SRC", fluid=f, TIT_C=TIT_C, P_high=P_high, P_low=P_low,
                state_1=_state(f, P_low, h1), state_2=_state(f, P_high, h2),
                state_hx_in=_state(f, P_high, h2), state_3=_state(f, P_high, h3),
                state_4=_state(f, P_low, h4), state_cool_in=_state(f, P_low, h4),
                q_in=q_in / 1e3, q_out=(h4 - h1) / 1e3, q_recup=0.0,
                w_t=w_t / 1e3, w_p=w_p / 1e3, w_net=(w_t - w_p) / 1e3,
                eta_th=(w_t - w_p) / q_in, bwr=w_p / w_t,
                T_sat_C=CP.PropsSI("T", "P", P_high, "Q", 0, f) - 273.15
                if P_high < CP.PropsSI("Pcrit", f) else None,
                T_cond_C=CP.PropsSI("T", "P", P_low, "Q", 0, f) - 273.15,
                dh_is=(h3 - h4s) / 1e3,
                v_in=1 / CP.PropsSI("D", "T", T3, "P", P_high, f),
                v_out=1 / CP.PropsSI("D", "P", P_low, "H", h4, f))


# --------------------------------------------------------------------------
# Recuperator helper (shared by sCO2 and tCO2)
# --------------------------------------------------------------------------
def _recuperate(f, P_high, P_low, h2, h4, eps):
    """Return (h3, h5, Q_recup) in J/kg for effectiveness eps."""
    if eps <= 0:
        return h2, h4, 0.0
    T4 = CP.PropsSI("T", "P", P_low, "H", h4, f)
    T2 = CP.PropsSI("T", "P", P_high, "H", h2, f)
    q_max_cold = CP.PropsSI("H", "T", T4, "P", P_high, f) - h2
    q_max_hot = h4 - CP.PropsSI("H", "T", T2, "P", P_low, f)
    Q = eps * min(q_max_cold, q_max_hot)
    return h2 + Q, h4 - Q, Q


# --------------------------------------------------------------------------
# Supercritical CO2 recuperated Brayton cycle
# --------------------------------------------------------------------------
def sco2_cycle(TIT_C, P_high=20e6, P_low=8e6, eps=0.85, T1_C=None,
               eta_t=ETA_T, eta_c=ETA_P):
    f = "CO2"
    assert P_low > P_CRIT_CO2 and P_high > P_low
    T1 = (T_CRIT_CO2 + 5.0) if T1_C is None else T1_C + 273.15
    T3 = TIT_C + 273.15
    h1 = CP.PropsSI("H", "T", T1, "P", P_low, f); s1 = CP.PropsSI("S", "T", T1, "P", P_low, f)
    w_c = (CP.PropsSI("H", "P", P_high, "S", s1, f) - h1) / eta_c
    h2 = h1 + w_c
    h3h = CP.PropsSI("H", "T", T3, "P", P_high, f); s3h = CP.PropsSI("S", "T", T3, "P", P_high, f)
    h4s = CP.PropsSI("H", "P", P_low, "S", s3h, f)
    w_t = (h3h - h4s) * eta_t
    h4 = h3h - w_t
    h3, h5, Q = _recuperate(f, P_high, P_low, h2, h4, eps)
    q_in = h3h - h3
    return dict(cycle="sCO2", fluid=f, TIT_C=TIT_C, P_high=P_high, P_low=P_low, eps=eps,
                state_1=_state(f, P_low, h1), state_2=_state(f, P_high, h2),
                state_hx_in=_state(f, P_high, h3), state_3=_state(f, P_high, h3h),
                state_4=_state(f, P_low, h4), state_cool_in=_state(f, P_low, h5),
                q_in=q_in / 1e3, q_out=(h5 - h1) / 1e3, q_recup=Q / 1e3,
                w_t=w_t / 1e3, w_p=w_c / 1e3, w_net=(w_t - w_c) / 1e3,
                eta_th=(w_t - w_c) / q_in, bwr=w_c / w_t, T_cond_C=T1 - 273.15,
                dh_is=(h3h - h4s) / 1e3,
                v_in=1 / CP.PropsSI("D", "T", T3, "P", P_high, f),
                v_out=1 / CP.PropsSI("D", "P", P_low, "H", h4, f))


# --------------------------------------------------------------------------
# Transcritical CO2 recuperated Rankine cycle
# --------------------------------------------------------------------------
def tco2_high_pressure(TIT_C):
    """High-side pressure rule (Liu et al.): rises with TIT, 12–20 MPa."""
    return min(20e6, max(12e6, 8e6 + (TIT_C - 100.0) * 48e3))


def tco2_cycle(TIT_C, P_high=None, P_low=5e6, eps=0.70, eta_t=ETA_T, eta_p=ETA_P):
    f = "CO2"
    if P_high is None:
        P_high = tco2_high_pressure(TIT_C)
    assert P_low < P_CRIT_CO2 < P_high
    T3 = TIT_C + 273.15
    h1 = CP.PropsSI("H", "P", P_low, "Q", 0, f); s1 = CP.PropsSI("S", "P", P_low, "Q", 0, f)
    w_p = (CP.PropsSI("H", "P", P_high, "S", s1, f) - h1) / eta_p
    h2 = h1 + w_p
    h3h = CP.PropsSI("H", "T", T3, "P", P_high, f); s3h = CP.PropsSI("S", "T", T3, "P", P_high, f)
    h4s = CP.PropsSI("H", "P", P_low, "S", s3h, f)
    w_t = (h3h - h4s) * eta_t
    h4 = h3h - w_t
    h3, h5, Q = _recuperate(f, P_high, P_low, h2, h4, eps)
    q_in = h3h - h3
    return dict(cycle="tCO2", fluid=f, TIT_C=TIT_C, P_high=P_high, P_low=P_low, eps=eps,
                state_1=_state(f, P_low, h1), state_2=_state(f, P_high, h2),
                state_hx_in=_state(f, P_high, h3), state_3=_state(f, P_high, h3h),
                state_4=_state(f, P_low, h4), state_cool_in=_state(f, P_low, h5),
                q_in=q_in / 1e3, q_out=(h5 - h1) / 1e3, q_recup=Q / 1e3,
                w_t=w_t / 1e3, w_p=w_p / 1e3, w_net=(w_t - w_p) / 1e3,
                eta_th=(w_t - w_p) / q_in, bwr=w_p / w_t,
                T_cond_C=CP.PropsSI("T", "P", P_low, "Q", 0, f) - 273.15,
                dh_is=(h3h - h4s) / 1e3,
                v_in=1 / CP.PropsSI("D", "T", T3, "P", P_high, f),
                v_out=1 / CP.PropsSI("D", "P", P_low, "H", h4, f))


# --------------------------------------------------------------------------
# Exergy balance (component destruction) for any of the three cycles
# --------------------------------------------------------------------------
def exergy_balance(c, T_source_C=None):
    """
    Component exergy destruction, kJ per kg of working fluid, with the heat
    source treated as a reservoir at T_source (default: turbine inlet
    temperature).  Also returns destruction per kJ of net work, which is the
    basis on which cycles with very different mass flows should be compared.
    """
    f = c["fluid"]
    T_src = (c["TIT_C"] if T_source_C is None else T_source_C) + 273.15
    s = {k: c[k] for k in ("state_1", "state_2", "state_hx_in", "state_3", "state_4", "state_cool_in")}
    ex = {k: flow_exergy(f, v["h"], v["s"]) for k, v in s.items()}
    ex_in = c["q_in"] * (1 - T0_K / T_src)
    d = dict(
        pump=T0_K * (s["state_2"]["s"] - s["state_1"]["s"]),
        recuperator=T0_K * ((s["state_hx_in"]["s"] - s["state_2"]["s"])
                            + (s["state_cool_in"]["s"] - s["state_4"]["s"])),
        heater=ex_in - (ex["state_3"] - ex["state_hx_in"]),
        turbine=T0_K * (s["state_4"]["s"] - s["state_3"]["s"]),
        cooler=ex["state_cool_in"] - ex["state_1"],
    )
    total = sum(d.values())
    return dict(destruction=d, total=total, ex_in=ex_in, w_net=c["w_net"],
                eta_ex=c["w_net"] / ex_in, closure=c["w_net"] + total - ex_in,
                per_kJ_work={k: v / c["w_net"] for k, v in d.items()},
                total_per_kJ_work=total / c["w_net"])
