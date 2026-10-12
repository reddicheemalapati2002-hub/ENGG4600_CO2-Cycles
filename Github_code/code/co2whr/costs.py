"""
Equipment cost correlations and total erected cost (RQ2).

Every cycle is costed with the same method:
  * primary heater, recuperator ...... Weiland, Lance & Pidaparti (NETL 2019)
                                       PCHE/compact-HX correlation, by UA
  * heat rejection ................... Weiland et al. (2019) air-cooler
                                       correlation, by UA
  * CO2 turbines (sCO2, tCO2) ........ Weiland et al. (2019) radial turbine,
                                       by shaft power
  * steam turbine .................... power law calibrated to the US DOE
                                       small steam turbine-generator
                                       benchmarks (Steam Tip Sheet 22)
  * sCO2 compressor .................. Turton et al. centrifugal compressor
  * feed / CO2 pumps ................. Turton et al. centrifugal pump
  * generator ........................ Weiland et al. (2019)
  * installation ..................... +35 % on all equipment (Turton)

All correlations are escalated to 2026 with CEPCI and converted at the RBA
spot rate of 17 April 2026.  Validity ranges are recorded so that
extrapolation can be reported explicitly.
"""
import numpy as np

CEPCI = {2001: 397.0, 2012: 584.6, 2016: 541.7, 2017: 567.5, 2026: 830.0}
USD_TO_AUD = 1.395
INSTALL_FACTOR = 1.35


def esc(year):
    return CEPCI[2026] / CEPCI[year]


# ---- Weiland et al. 2019 (2017 USD) ---------------------------------------
def f_T(T_max_C):
    return 1.0 if T_max_C <= 550 else 1.0 + 0.02141 * (T_max_C - 550)


def hx_pche_usd(UA_W_K, T_max_C):
    """Recuperator / compact HX: C = 49.45 UA^0.7544 f_T  (UA 1.6e5–2.15e8 W/K)."""
    return 49.45 * UA_W_K ** 0.7544 * f_T(T_max_C) * esc(2017)


def cooler_usd(UA_W_K):
    """Direct air cooler: C = 32.88 UA^0.75  (UA 8.6e5–7.5e7 W/K)."""
    return 32.88 * UA_W_K ** 0.75 * esc(2017)


def co2_turbine_usd(W_shaft_kW):
    """Radial sCO2 turbine: C = 406,200 W_MW^0.8  (8–35 MW; extrapolated below)."""
    return 406_200 * (W_shaft_kW / 1000.0) ** 0.8 * esc(2017)


def generator_usd(W_e_kW):
    """Generator: C = 108,900 W_MWe^0.5463  (4–750 MWe; extrapolated below)."""
    return 108_900 * (W_e_kW / 1000.0) ** 0.5463 * esc(2017)


# ---- Steam turbine: DOE Steam Tip Sheet 22 benchmarks (2012 USD) ---------
# ~USD 900/kW at 150 kW and ~USD 200/kW at 2,000 kW (equipment only)
_b_st = np.log(200 * 2000 / (900 * 150)) / np.log(2000 / 150)
_a_st = 900 * 150 / 150 ** _b_st


def steam_turbine_usd(W_shaft_kW):
    return _a_st * W_shaft_kW ** _b_st * esc(2012)


# ---- Turton et al. 2012 (2001 USD, CEPCI 397) ------------------------------
def _turton(K1, K2, K3, X):
    lx = np.log10(X)
    return 10 ** (K1 + K2 * lx + K3 * lx ** 2)


def compressor_usd(W_kW):
    """Centrifugal compressor, 450–3000 kW (extrapolated outside)."""
    return _turton(2.2897, 1.3604, -0.1027, max(W_kW, 1.0)) * esc(2001)


def pump_usd(W_kW):
    """Centrifugal pump, 1–300 kW (extrapolated outside).  No pressure
    factor is applied (conservative for the 13–20 MPa tCO2 pump)."""
    return _turton(3.3892, 0.0536, 0.1538, max(W_kW, 1.0)) * esc(2001)


# ---- Total erected cost for a design point --------------------------------
def tec_breakdown(design, W_net_kW):
    """
    Cost of one cycle sized to a net output W_net_kW, using the per-kg/s
    design point `design` (from design.site_design).  Returns kAUD.
    """
    c, cyc = design["c"], design["cycle"]
    m = W_net_kW / c["w_net"]                       # kg/s working fluid
    W_t = c["w_t"] * m                              # turbine shaft power, kW
    W_p = c["w_p"] * m                              # pump / compressor power, kW
    T_src = design["hx"]["T_gas"][-1]
    T_max_rec = c["state_4"]["T_C"]

    turb = steam_turbine_usd(W_t) if cyc == "SRC" else co2_turbine_usd(W_t)
    comp = compressor_usd(W_p) if cyc == "sCO2" else pump_usd(W_p)
    heater = hx_pche_usd(design["UA_heater"] * m, T_src)
    recup = hx_pche_usd(design["UA_recup"] * m, T_max_rec) if design["UA_recup"] > 0 else 0.0
    cooler = cooler_usd(design["UA_cooler"] * m)
    gen = generator_usd(W_net_kW)
    items = dict(turbine=turb, pump_comp=comp, heater=heater, recuperator=recup,
                 cooler=cooler, generator=gen)
    items = {k: v * USD_TO_AUD / 1e3 for k, v in items.items()}     # kAUD
    equip = sum(items.values())
    tec = equip * INSTALL_FACTOR
    return dict(items=items, equipment_kAUD=equip, TEC_kAUD=tec,
                TEC_per_kW=tec * 1e3 / W_net_kW, m_wf=m, W_turbine_kW=W_t, W_pump_kW=W_p,
                hx_share=(items["heater"] + items["recuperator"]) / equip)
