"""
Site design points (RQ2/RQ3).

A site is a fixed waste-gas stream (T_src, m_gas).  Each cycle is integrated
with the stream through a counter-flow primary heater with a 10 K minimum
approach, and the turbine inlet temperature is T_src - 10 K.

For the CO2 cycles the high- and low-side pressures follow the RQ1 rules
(sCO2: 20/8 MPa; tCO2: P_high(TIT), P_low = 5 MPa).  For the steam cycle the
boiler pressure is optimised for each site: the evaporation temperature
fixes an internal pinch, so a higher pressure raises cycle efficiency but
reduces the heat that can be recovered from the gas.  The design maximises
net power per kilogram of waste gas,
      w_gas = eta_th * cp_gas * (T_src - T_gas,out).
"""
import numpy as np
import CoolProp.CoolProp as CP
from .cycles import src_cycle, sco2_cycle, tco2_cycle, tco2_high_pressure, T_CRIT_H2O
from .heatex import heater_for_cycle, recuperator_UA, cooler_UA, integrate_heater
from .sites import SITES, CP_GAS, DT_APPROACH, DT_HOT_END


def _w_gas(c, hx, T_src):
    return c["eta_th"] * CP_GAS * (T_src - hx["T_gas_out"])


def src_site_pressure(T_src, dT_min=DT_APPROACH):
    """Golden-section search for the boiler pressure that maximises net
    power per kg of waste gas (subcritical, <= 10 MPa)."""
    TIT = T_src - DT_HOT_END
    P_hi = min(10e6, 0.95 * CP.PropsSI("P", "T", min(TIT, T_CRIT_H2O - 273.15 - 1) + 273.15, "Q", 1, "Water"))
    lo, hi = 0.15e6, P_hi
    g = (np.sqrt(5) - 1) / 2

    def f(P):
        c = src_cycle(TIT, P)
        return _w_gas(c, heater_for_cycle(c, T_src, dT_min), T_src)
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(40):
        if fa < fb:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo); fb = f(b)
        else:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo); fa = f(a)
    return 0.5 * (lo + hi)


def site_cycle(cycle, T_src, P_src=None):
    TIT = T_src - DT_HOT_END
    if cycle == "SRC":
        return src_cycle(TIT, src_site_pressure(T_src) if P_src is None else P_src)
    if cycle == "sCO2":
        return sco2_cycle(TIT)
    if cycle == "tCO2":
        return tco2_cycle(TIT, P_high=tco2_high_pressure(T_src))   # pressure rule on source T
    raise ValueError(cycle)


def site_design(cycle, site_key, dT_min=DT_APPROACH):
    """Full design point of one cycle at one site, per kg/s of working fluid."""
    s = SITES[site_key]
    c = site_cycle(cycle, s["T_src"])
    hx = heater_for_cycle(c, s["T_src"], dT_min)
    rec = recuperator_UA(c)
    w_gas = _w_gas(c, hx, s["T_src"])
    W_site = s["m_gas"] * w_gas                      # kW from the whole stream
    m_wf = W_site / c["w_net"]                       # kg/s working fluid
    return dict(cycle=cycle, site=site_key, c=c, hx=hx, rec=rec,
                UA_heater=hx["UA"], UA_recup=rec["UA"], UA_cooler=cooler_UA(c),
                T_gas_out=hx["T_gas_out"], w_gas=w_gas, W_site_kW=W_site, m_wf=m_wf,
                m_gas_per_wf=hx["gas_per_wf"], Q_recovered_kW=s["m_gas"] * CP_GAS * (s["T_src"] - hx["T_gas_out"]),
                recovery_frac=(s["T_src"] - hx["T_gas_out"]) / (s["T_src"] - 25.0))


def lmtd_comparison(site_key, eps=0.0, U=379.7):
    """
    Two-point LMTD versus segmented (NTU-type) sizing of the tCO2 heater at
    recuperator effectiveness eps (eps = 0 puts the pseudo-critical cp peak
    inside the heater).  Areas in m^2 per kg/s with the PCHE U of the thesis.
    """
    s = SITES[site_key]
    T_src = s["T_src"]
    c = tco2_cycle(T_src - DT_HOT_END, P_high=tco2_high_pressure(T_src), eps=eps)
    hx = heater_for_cycle(c, T_src)
    return dict(A_seg=hx["UA"] / U, A_2pt=hx["UA_2pt"] / U, over_pct=(hx["UA_2pt"] / hx["UA"] - 1) * 100,
                LMTD_eff=hx["LMTD_eff"], LMTD_2pt=hx["LMTD_2pt"], hx=hx, c=c, U=U)
