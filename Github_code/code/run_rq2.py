"""RQ2 — compactness, heat-exchanger sizing and capital cost."""
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import save, COLORS, MARKERS, LABELS, SITE_SHORT
from co2whr.design import site_design, lmtd_comparison
from co2whr.costs import tec_breakdown
from co2whr.sites import SITES, SITE_ORDER, CYCLES

SIZES = [100, 500, 1000, 5000]


def sp_vfr(designs, fig_dir):
    out = {}
    for sk in SITE_ORDER:
        out[sk] = {}
        for cyc in CYCLES:
            c = designs[sk][cyc]["c"]
            sp = c["v_out"] ** 0.5 / c["dh_is"] ** 0.25         # per kg/s
            out[sk][cyc] = dict(SP=sp, VFR=c["v_out"] / c["v_in"], dh_is=c["dh_is"], v_out=c["v_out"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(3); w = 0.26
    for i, cyc in enumerate(CYCLES):
        v = [out[sk][cyc]["SP"] for sk in SITE_ORDER]; b = a1.bar(x + (i - 1) * w, v, w, color=COLORS[cyc], label=LABELS[cyc])
        for bb, vv in zip(b, v): a1.text(bb.get_x() + w / 2, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=7.5)
        v = [out[sk][cyc]["VFR"] for sk in SITE_ORDER]; b = a2.bar(x + (i - 1) * w, v, w, color=COLORS[cyc], label=LABELS[cyc])
        for bb, vv in zip(b, v): a2.text(bb.get_x() + w / 2, vv * 1.1, f"{vv:.1f}", ha="center", fontsize=7.5)
    a1.set_xticks(x); a1.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a1.set_ylabel("Size parameter SP (m·(kJ/kg)⁻⁰·²⁵) at 1 kg/s")
    a1.set_title("Turbine size parameter (smaller = more compact)"); a1.legend()
    a2.set_yscale("log"); a2.axhline(5, color="#CC0000", ls="--", lw=1); a2.text(2.1, 5.6, "VFR = 5: single-stage limit", color="#CC0000", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a2.set_ylabel("Volumetric flow ratio v_out / v_in (log)")
    a2.set_title("Turbine volumetric flow ratio"); a2.legend()
    fig.suptitle("Turbine compactness at the site design points (TIT = T_source − 10 K, η_t = 0.85)", fontsize=10)
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ2_TURBINE_SP_VFR.png")
    return out


def hx_lmtd(fig_dir):
    out = {sk: lmtd_comparison(sk) for sk in SITE_ORDER}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(3); w = 0.35
    seg = [out[s]["A_seg"] for s in SITE_ORDER]; two = [out[s]["A_2pt"] for s in SITE_ORDER]
    b1 = a1.bar(x - w / 2, seg, w, color="#51247A", label="Segmented (200-node T–h)"); b2 = a1.bar(x + w / 2, two, w, color="#BB9B1C", label="Two-point LMTD")
    for b, v in list(zip(b1, seg)) + list(zip(b2, two)): a1.text(b.get_x() + w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a1.set_ylabel("Heater area (m² per kg/s CO₂), U = 380 W/m²K"); a1.legend()
    a1.set_title("tCO₂ heater area, ε = 0 (c_p peak inside the heater)")
    ov = [out[s]["over_pct"] for s in SITE_ORDER]
    b = a2.bar(x, ov, 0.5, color=["#BB9B1C", "#51247A", "#0064B4"])
    for bb, v in zip(b, ov): a2.text(bb.get_x() + 0.25, v + 4, f"+{v:.0f} %", ha="center", fontsize=10, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels([SITE_SHORT[s] for s in SITE_ORDER]); a2.set_ylabel("Two-point LMTD overestimate of area (%)"); a2.set_ylim(0, max(ov) * 1.25)
    a2.set_title("Both terminals sit at the 10 K approach, so the two-point\nLMTD (10 K) understates the mean driving force")
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ2_HX_AREA_NTU_vs_LMTD.png")
    return {sk: {k: v for k, v in out[sk].items() if k not in ("hx", "c")} for sk in out}


def capital(designs, fig_dir):
    out = {}
    for sk in SITE_ORDER:
        out[sk] = {}
        for cyc in CYCLES:
            out[sk][cyc] = {}
            for W in SIZES:
                b = tec_breakdown(designs[sk][cyc], W)
                out[sk][cyc][W] = dict(TEC_per_kW=b["TEC_per_kW"], TEC_kAUD=b["TEC_kAUD"], items=b["items"],
                                       hx_share=b["hx_share"], W_turbine_kW=b["W_turbine_kW"], W_pump_kW=b["W_pump_kW"], m_wf=b["m_wf"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for a, sk in zip(axes, SITE_ORDER):
        for cyc in CYCLES:
            v = [out[sk][cyc][W]["TEC_per_kW"] for W in SIZES]
            a.plot(SIZES, v, color=COLORS[cyc], marker=MARKERS[cyc], label=LABELS[cyc])
            for W, vv in zip(SIZES, v): a.annotate(f"{vv/1000:.1f}k", (W, vv), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=COLORS[cyc])
        a.set_xscale("log"); a.set_xticks(SIZES); a.set_xticklabels(["100 kW", "500 kW", "1 MW", "5 MW"]); a.set_title(SITES[sk]["name"]); a.set_xlabel("Net output"); a.legend()
    axes[0].set_ylabel("Total erected cost (AUD/kW, 2026)")
    fig.suptitle("Specific capital cost versus plant size at each site's design conditions (CEPCI 2026 ≈ 830, 1 USD = 1.395 AUD, +35 % installation)", fontsize=9.5)
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ2_CAPITAL_COST_AUD_per_kW.png")
    # breakdown at 1 MW
    comps = ["turbine", "pump_comp", "heater", "recuperator", "cooler", "generator"]
    names = ["Turbine", "Pump / compressor", "Primary heater", "Recuperator", "Condenser / cooler", "Generator"]
    cols = ["#51247A", "#8E6BB0", "#BB9B1C", "#E0CF7A", "#0064B4", "#8AB4D9"]
    fig, a = plt.subplots(figsize=(11, 5))
    xs, labels = [], []
    for i, sk in enumerate(SITE_ORDER):
        for j, cyc in enumerate(CYCLES):
            xpos = i * 4 + j; xs.append(xpos); labels.append(LABELS[cyc])
            bottom = 0
            for comp, nm, col in zip(comps, names, cols):
                v = out[sk][cyc][1000]["items"][comp] * 1.35
                a.bar(xpos, v, 0.8, bottom=bottom, color=col, label=nm if (i == 0 and j == 0) else None)
                bottom += v
            a.text(xpos, bottom + 80, f"{bottom:,.0f}", ha="center", fontsize=7.5)
        a.text(i * 4 + 1, -1500, SITES[sk]["name"], ha="center", fontsize=9, fontweight="bold")
    a.set_xticks(xs); a.set_xticklabels(labels); a.set_ylabel("Total erected cost at 1 MWe (kAUD, 2026)"); a.legend(ncol=3, fontsize=8, loc="upper right")
    a.set_title("Capital cost breakdown at 1 MWe net output (installed basis)"); a.set_ylim(0, max(out[s][c][1000]["TEC_kAUD"] for s in SITE_ORDER for c in CYCLES) * 1.15)
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ2_COST_BREAKDOWN.png")
    return out


def run(fig_dir):
    print("RQ2 ...")
    designs = {sk: {cyc: site_design(cyc, sk) for cyc in CYCLES} for sk in SITE_ORDER}
    dsum = {sk: {cyc: dict(P_high_MPa=d["c"]["P_high"] / 1e6, TIT=d["c"]["TIT_C"], eta=d["c"]["eta_th"] * 100, bwr=d["c"]["bwr"] * 100,
                           w_net=d["c"]["w_net"], q_in=d["c"]["q_in"], q_recup=d["c"]["q_recup"], hx_in=d["c"]["state_hx_in"]["T_C"],
                           T_gas_out=d["T_gas_out"], w_gas=d["w_gas"], W_site_kW=d["W_site_kW"], m_wf=d["m_wf"],
                           Q_recovered_kW=d["Q_recovered_kW"], UA_heater=d["UA_heater"], UA_recup=d["UA_recup"], UA_cooler=d["UA_cooler"],
                           LMTD_heater=d["hx"]["LMTD_eff"], rec_dTmin=float(d["rec"].get("dT_min", np.nan)), T_sat=d["c"].get("T_sat_C"),
                           T_cond=d["c"]["T_cond_C"], pinch=__import__("co2whr.heatex", fromlist=["pinch_report"]).pinch_report(d["hx"]))
                 for cyc, d in designs[sk].items()} for sk in SITE_ORDER}
    return designs, dict(design=dsum, sp_vfr=sp_vfr(designs, fig_dir), lmtd=hx_lmtd(fig_dir), capital=capital(designs, fig_dir))
