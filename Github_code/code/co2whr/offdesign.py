"""
Off-design / part-load performance (RQ4).

Turbine (and sCO2 compressor) isentropic efficiency is degraded with a
parabolic Stodola-type correction,  eta_od/eta_0 = 1 - 0.3 (1 - f)^2,
where f is the mass-flow fraction, and sliding-pressure control is applied:
  SRC   : P_high = P_design * f
  sCO2  : P_high = 20 MPa * (0.5 + 0.5 f)       (P_low fixed, 8 MPa)
  tCO2  : P_high = P_design * (0.6 + 0.4 f)     (P_low fixed, 5 MPa)
Turbine inlet temperature is held at the site design value.  The heater
integration with the source is not re-solved at part load (steady-state
cycle efficiency only).
"""
import numpy as np
from .cycles import src_cycle, sco2_cycle, tco2_cycle, ETA_T, ETA_P
from .design import site_cycle
from .sites import SITES, DT_HOT_END

LOADS = [1.0, 0.8, 0.6, 0.4, 0.2]
STARTUP = {"SRC": 0.55, "sCO2": 0.75, "tCO2": 0.80}      # qualitative, Marchionni et al.
WEIGHTS = dict(turndown=0.30, r60=0.30, r40=0.25, startup=0.15)


def stodola(f):
    return 1.0 - 0.3 * (1.0 - f) ** 2


def part_load_eta(cycle, site_key, f):
    T_src = SITES[site_key]["T_src"]
    TIT = T_src - DT_HOT_END
    d = site_cycle(cycle, T_src)
    if cycle == "SRC":
        c = src_cycle(TIT, d["P_high"] * f, eta_t=ETA_T * stodola(f))
    elif cycle == "sCO2":
        c = sco2_cycle(TIT, P_high=20e6 * (0.5 + 0.5 * f), eta_t=ETA_T * stodola(f),
                       eta_c=ETA_P * stodola(f))
    else:
        c = tco2_cycle(TIT, P_high=d["P_high"] * (0.6 + 0.4 * f), eta_t=ETA_T * stodola(f))
    return max(c["eta_th"], 0.0)


def sweep():
    out = {}
    for sk in SITES:
        out[sk] = {}
        for cyc in ("SRC", "sCO2", "tCO2"):
            etas = [part_load_eta(cyc, sk, f) * 100 for f in LOADS]
            out[sk][cyc] = dict(eta=etas, retention=[e / etas[0] for e in etas])
    return out


def marchionni(res):
    scores = {}
    for cyc in ("SRC", "sCO2", "tCO2"):
        r60 = float(np.mean([res[sk][cyc]["retention"][LOADS.index(0.6)] for sk in res]))
        r40 = float(np.mean([res[sk][cyc]["retention"][LOADS.index(0.4)] for sk in res]))
        turndown = 0.85 if r40 > 0.80 else (0.70 if r40 > 0.70 else 0.55)
        total = (WEIGHTS["turndown"] * turndown + WEIGHTS["r60"] * r60
                 + WEIGHTS["r40"] * r40 + WEIGHTS["startup"] * STARTUP[cyc])
        scores[cyc] = dict(turndown=turndown, r60=r60, r40=r40, startup=STARTUP[cyc], total=total)
    return scores
