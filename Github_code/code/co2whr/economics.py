"""
Techno-economics (RQ3): LCOE, NPV, discounted payback, IRR and Monte Carlo.

Basis: 7 % real discount rate, 20-year life, 85 % capacity factor, O&M at
2.5 % of capital per year, waste heat free of charge, no carbon price.
Payback is DISCOUNTED (first year in which cumulative discounted net cash
flow exceeds the capital), consistent with NPV and LCOE.
"""
import numpy as np

R, LIFE, CF, OM = 0.07, 20, 0.85, 0.025
HOURS = 8760.0
CRF = R * (1 + R) ** LIFE / ((1 + R) ** LIFE - 1)


def cashflow_metrics(tec_kAUD, W_kW, price, cf=CF, r=R, life=LIFE, om=OM):
    cap = tec_kAUD * 1e3
    e_gen = W_kW * HOURS * cf / 1e3                      # MWh/yr
    om_a = om * cap
    net = e_gen * price - om_a
    disc = np.array([(1 + r) ** -t for t in range(1, life + 1)])
    npv = net * disc.sum() - cap
    lcoe = (cap * CRF + om_a) / e_gen
    cum = np.cumsum(net * disc)
    if net <= 0 or cum[-1] < cap:
        pb = np.inf
    else:
        k = int(np.argmax(cum >= cap))                 # first year index with cum >= cap
        prev = cum[k - 1] if k > 0 else 0.0
        pb = k + (cap - prev) / (cum[k] - prev)         # linear interpolation within the year
    irr = _irr(cap, net, life)
    return dict(TEC_kAUD=tec_kAUD, W_kW=W_kW, E_MWh=e_gen, net_cash_kAUD=net / 1e3,
                LCOE=lcoe, NPV_kAUD=npv / 1e3, payback_disc=pb, IRR=irr,
                payback_simple=cap / net if net > 0 else np.inf)


def _irr(cap, net, life):
    if net <= 0:
        return np.nan
    lo, hi = -0.99, 10.0
    f = lambda r: net * sum((1 + r) ** -t for t in range(1, life + 1)) - cap
    if f(hi) > 0:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def monte_carlo(tec_kAUD, W_kW, price, n=10_000, seed=42,
                cap_range=(0.70, 1.30), price_range=(0.60, 1.40), cf_range=(0.60, 0.95)):
    """Uniform sampling of capital cost, electricity price and capacity factor."""
    rng = np.random.default_rng(seed)
    cap = tec_kAUD * 1e3 * rng.uniform(*cap_range, n)
    pr = price * rng.uniform(*price_range, n)
    cf = rng.uniform(*cf_range, n)
    e = W_kW * HOURS * cf / 1e3
    net = e * pr - OM * cap
    disc = sum((1 + R) ** -t for t in range(1, LIFE + 1))
    npv = net * disc - cap
    lcoe = (cap * CRF + OM * cap) / e
    return dict(p_viable=float(np.mean(npv > 0) * 100),
                npv_p10=float(np.percentile(npv, 10) / 1e3), npv_p50=float(np.percentile(npv, 50) / 1e3),
                npv_p90=float(np.percentile(npv, 90) / 1e3),
                lcoe_p10=float(np.percentile(lcoe, 10)), lcoe_p50=float(np.percentile(lcoe, 50)),
                lcoe_p90=float(np.percentile(lcoe, 90)))


def hx_cost_reduction_to_match(items_kAUD, target_tec_kAUD, install=1.35):
    """
    Fractional reduction in heat-exchanger cost (heater + recuperator) that
    would bring a cycle's TEC down to target_tec_kAUD.  Returns None if even
    free heat exchangers would not reach the target.
    """
    hx = items_kAUD["heater"] + items_kAUD["recuperator"]
    other = sum(items_kAUD.values()) - hx
    needed = target_tec_kAUD / install - other
    if needed < 0:
        return None
    return max(0.0, 1.0 - needed / hx)
