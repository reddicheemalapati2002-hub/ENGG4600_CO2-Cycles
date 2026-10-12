"""RQ1 — thermodynamic performance: parametric sweep, exergy, pinch, sensitivities."""
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import save, COLORS, MARKERS, LABELS, GREY
from co2whr.cycles import (src_cycle, src_sweep_pressure, sco2_cycle, tco2_cycle,
                           tco2_high_pressure, exergy_balance)
from co2whr.heatex import heater_for_cycle, recuperator_UA, integrate_heater, pinch_report
from co2whr.design import site_cycle
from co2whr.sites import SITES, SITE_ORDER, CP_GAS, DT_HOT_END

TEMPS = list(range(150, 625, 25))


def sweep():
    rows = []
    for T in TEMPS:
        s = src_cycle(T, src_sweep_pressure(T)); b = sco2_cycle(T); t = tco2_cycle(T)
        rows.append(dict(TIT=T, SRC=s["eta_th"] * 100, sCO2=b["eta_th"] * 100, tCO2=t["eta_th"] * 100,
                         SRC_bwr=s["bwr"] * 100, sCO2_bwr=b["bwr"] * 100, tCO2_bwr=t["bwr"] * 100,
                         SRC_P=s["P_high"] / 1e6, tCO2_P=t["P_high"] / 1e6,
                         tCO2_recup_dTmin=recuperator_UA(t)["dT_min"],
                         sCO2_recup_dTmin=recuperator_UA(b)["dT_min"]))
    return rows


def plot_sweep(rows, fig_dir):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)
    T = [r["TIT"] for r in rows]
    for cyc in ("SRC", "sCO2", "tCO2"):
        a1.plot(T, [r[cyc] for r in rows], color=COLORS[cyc], marker=MARKERS[cyc], label=LABELS[cyc])
        a2.plot(T, [r[cyc + "_bwr"] for r in rows], color=COLORS[cyc], marker=MARKERS[cyc], label=LABELS[cyc])
    for sk in SITE_ORDER:
        for a in (a1, a2):
            a.axvline(SITES[sk]["T_src"], color=GREY, ls="--", lw=1, alpha=0.6)
        a1.text(SITES[sk]["T_src"] + 3, 2, SITES[sk]["name"], rotation=90, va="bottom", fontsize=7.5, color=GREY)
    a1.set_ylabel("Thermal efficiency η_th (%)"); a1.set_ylim(0, 45); a1.legend(loc="lower right")
    a1.set_title("Thermal efficiency versus turbine inlet temperature\n"
                 "SRC: ≤10 MPa, 10 kPa condenser  |  sCO₂: 20/8 MPa, ε = 0.85, T₁ = 36 °C  |  "
                 "tCO₂: P_high(TIT), 5 MPa, ε = 0.70", fontsize=9)
    a2.set_ylabel("Back-work ratio (%)"); a2.set_xlabel("Turbine inlet temperature (°C)")
    a2.set_ylim(0, 90); a2.legend()
    a2.set_title("Back-work ratio (compression work ÷ turbine work)", fontsize=9)
    save(fig, f"{fig_dir}/RQ1_THREE_CYCLE.png")


def exergy(fig_dir):
    """Component exergy destruction at a common TIT of 400 °C, per kg of
    working fluid and per kJ of net work."""
    T = 400
    cyc = {"SRC": src_cycle(T, src_sweep_pressure(T)), "sCO2": sco2_cycle(T), "tCO2": tco2_cycle(T)}
    ex = {k: exergy_balance(c) for k, c in cyc.items()}
    comps = ["pump", "heater", "turbine", "cooler", "recuperator"]
    names = ["Pump /\ncompressor", "Primary\nheater", "Turbine", "Cond. /\ncooler", "Recup-\nerator"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(comps)); w = 0.26
    for i, k in enumerate(cyc):
        v1 = [ex[k]["destruction"][c] for c in comps]
        v2 = [ex[k]["per_kJ_work"][c] for c in comps]
        a1.bar(x + (i - 1) * w, v1, w, color=COLORS[k], label=LABELS[k])
        a2.bar(x + (i - 1) * w, v2, w, color=COLORS[k], label=LABELS[k])
    for a, yl, tt in ((a1, "Exergy destruction (kJ per kg working fluid)", "Per kilogram of working fluid"),
                      (a2, "Exergy destruction (kJ per kJ net work)", "Per unit of net work (comparable basis)")):
        a.set_xticks(x); a.set_xticklabels(names, fontsize=8.5); a.set_ylabel(yl); a.set_title(tt); a.legend()
    txt = "  ".join(f"{LABELS[k]}: η_ex = {ex[k]['eta_ex']*100:.1f} %" for k in cyc)
    fig.suptitle(f"Exergy destruction by component at TIT = 400 °C (dead state 25 °C, 1 atm)   —   {txt}", fontsize=9.5)
    save(fig, f"{fig_dir}/RQ1_EXERGY.png")
    return {k: dict(destruction=ex[k]["destruction"], per_kJ_work=ex[k]["per_kJ_work"], total=ex[k]["total"],
                    total_per_kJ_work=ex[k]["total_per_kJ_work"], ex_in=ex[k]["ex_in"], eta_ex=ex[k]["eta_ex"],
                    closure=ex[k]["closure"], w_net=ex[k]["w_net"], eta_th=cyc[k]["eta_th"],
                    heater_share=ex[k]["destruction"]["heater"] / ex[k]["total"]) for k in cyc}


def exergy_sites():
    """Exergy destruction at the three site design points (per kJ net work)."""
    out = {}
    for sk in SITE_ORDER:
        T_src = SITES[sk]["T_src"]; out[sk] = {}
        for cyc in ("SRC", "sCO2", "tCO2"):
            c = site_cycle(cyc, T_src); e = exergy_balance(c, T_source_C=T_src)
            out[sk][cyc] = dict(eta_th=c["eta_th"], eta_ex=e["eta_ex"], total_per_kJ_work=e["total_per_kJ_work"],
                                per_kJ_work=e["per_kJ_work"], destruction=e["destruction"], ex_in=e["ex_in"], w_net=e["w_net"])
    return out


def chiller_penalty():
    """Indicative parasitic load if the tCO2 condenser (14.3 °C) and the sCO2
    cooler (36 °C inlet) had to be served by mechanical chilling, as a
    fraction of net work, for an electrically driven chiller of COP 4 and 6."""
    out = {}
    for sk in SITE_ORDER:
        T_src = SITES[sk]["T_src"]; out[sk] = {}
        for cyc in ("tCO2",):
            c = site_cycle(cyc, T_src)
            out[sk][cyc] = {f"COP{cop}": c["q_out"] / cop / c["w_net"] for cop in (4, 6)}
            out[sk][cyc]["q_out"] = c["q_out"]; out[sk][cyc]["w_net"] = c["w_net"]
    return out


def pinch(fig_dir):
    """tCO2 heater T–h diagrams at eps = 0 (cp peak inside the heater)."""
    out = {}
    for sk in SITE_ORDER:
        s = SITES[sk]; T_src = s["T_src"]
        c = tco2_cycle(T_src - DT_HOT_END, P_high=tco2_high_pressure(T_src), eps=0.0)
        hx = heater_for_cycle(c, T_src)
        c7 = tco2_cycle(T_src - DT_HOT_END, P_high=tco2_high_pressure(T_src), eps=0.70)
        hx7 = heater_for_cycle(c7, T_src)
        ipc = np.nanargmax(hx["cp"])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.5, 1]))
        a1.plot(hx["q_kJ"], hx["T_wf"], color=COLORS["tCO2"], label="tCO₂, pump exit → turbine inlet (ε = 0)")
        a1.plot(hx["q_kJ"], hx["T_gas"], color="#CC0000", ls="--", label=f"Waste gas, {T_src:.0f} °C → {hx['T_gas_out']:.1f} °C (linear in duty)")
        a1.fill_between(hx["q_kJ"], hx["T_wf"], hx["T_gas"], color=COLORS["tCO2"], alpha=0.10)
        a1.plot(hx["q_kJ"][ipc], hx["T_wf"][ipc], "D", color="green", ms=7, zorder=5)
        a1.annotate(f"pseudo-critical T_pc = {hx['T_wf'][ipc]:.1f} °C", (hx["q_kJ"][ipc], hx["T_wf"][ipc]),
                    xytext=(hx["q_kJ"][ipc] + 0.05 * hx["q_kJ"][-1], hx["T_wf"][ipc] - 28), fontsize=8.5,
                    color="green", arrowprops=dict(arrowstyle="->", color="green"))
        im = hx["dT"].argmax()
        a1.annotate(f"largest gap ΔT = {hx['dT_max']:.1f} K\nat T_CO₂ = {hx['T_wf'][im]:.0f} °C",
                    (hx["q_kJ"][im], 0.5 * (hx["T_wf"][im] + hx["T_gas"][im])), xytext=(hx["q_kJ"][im] * 0.55, hx["T_gas"][im] + 25),
                    fontsize=8.5, color="#51247A", arrowprops=dict(arrowstyle="->", color="#51247A"))
        for q0, T0, lab in ((hx["q_kJ"][0], hx["T_wf"][0], "cold-end pinch 10 K"), (hx["q_kJ"][-1], hx["T_wf"][-1], "hot-end pinch 10 K")):
            a1.plot(q0, T0, "o", color="#51247A", ms=6, zorder=6); a1.text(q0, T0 + 8, lab, fontsize=7.5, color="#51247A", ha="left" if q0 == 0 else "right")
        a1.set_ylabel("Temperature (°C)"); a1.legend(loc="upper left", fontsize=8)
        a1.set_title(f"tCO₂ primary heater T–h diagram — {s['name']}  |  P_high = {c['P_high']/1e6:.1f} MPa, P_low = 5 MPa, 10 K approach\n"
                     f"With ε = 0.70 (design) the heater inlet rises to {c7['state_hx_in']['T_C']:.0f} °C and the gas can only be cooled to {hx7['T_gas_out']:.0f} °C", fontsize=9)
        a2.plot(hx["q_kJ"], hx["cp"], color=COLORS["tCO2"]); a2.fill_between(hx["q_kJ"], hx["cp"], alpha=0.15, color=COLORS["tCO2"])
        a2.axvline(hx["q_kJ"][ipc], color="green", ls=":", lw=1)
        a2.set_ylabel("c_p of CO₂ (kJ/kg·K)"); a2.set_xlabel("Cumulative heat transferred from the cold end (kJ per kg CO₂)")
        a2.set_title("Isobaric specific heat along the heater", fontsize=9)
        save(fig, f"{fig_dir}/tco2_pinch_{T_src:.0f}C.png")
        out[sk] = dict(P_high=c["P_high"] / 1e6, T_pump_exit=c["state_2"]["T_C"], T_pc=float(hx["T_wf"][ipc]),
                       cp_peak=float(np.nanmax(hx["cp"])), T_gas_out=hx["T_gas_out"], Q=hx["Q"],
                       dT_min=hx["dT_min"], dT_max=hx["dT_max"], T_at_dTmax=float(hx["T_wf"][im]),
                       pinch=pinch_report(hx), LMTD_eff=hx["LMTD_eff"],
                       eps070_hx_in=c7["state_hx_in"]["T_C"], eps070_T_gas_out=hx7["T_gas_out"], eps070_LMTD_eff=hx7["LMTD_eff"])
    return out


def recuperator_sensitivity(fig_dir):
    eps_vals = [0.0, 0.5, 0.7, 0.85]
    res = {}
    for sk in SITE_ORDER:
        s = SITES[sk]; T_src = s["T_src"]
        res[sk] = []
        for e in eps_vals:
            c = tco2_cycle(T_src - DT_HOT_END, P_high=tco2_high_pressure(T_src), eps=e)
            hx = heater_for_cycle(c, T_src); r = recuperator_UA(c)
            res[sk].append(dict(eps=e, eta=c["eta_th"] * 100, bwr=c["bwr"] * 100, w_net=c["w_net"], q_in=c["q_in"],
                                q_recup=c["q_recup"], hx_in=c["state_hx_in"]["T_C"], T_gas_out=hx["T_gas_out"],
                                w_gas=c["eta_th"] * CP_GAS * (T_src - hx["T_gas_out"]), rec_dTmin=r.get("dT_min", np.nan),
                                UA_recup=r["UA"], UA_heater=hx["UA"]))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8)); ax = ax.flatten()
    panels = [("eta", "Thermal efficiency (%)"), ("w_gas", "Net work per kg of waste gas (kJ/kg)"),
              ("q_recup", "Recuperator duty (kJ per kg CO₂)"), ("T_gas_out", "Achievable gas outlet temperature (°C)")]
    site_col = {"CS2": "#BB9B1C", "CS1": "#51247A", "CS3": "#0064B4"}
    for a, (k, yl) in zip(ax, panels):
        for sk in SITE_ORDER:
            a.plot([d["eps"] for d in res[sk]], [d[k] for d in res[sk]], marker=MARKERS["tCO2"], color=site_col[sk], label=SITES[sk]["name"])
        a.set_xlabel("Recuperator effectiveness ε"); a.set_ylabel(yl); a.set_xticks(eps_vals); a.legend(fontsize=8)
    fig.suptitle("tCO₂ Rankine — recuperator effectiveness sensitivity (P_low = 5 MPa, 10 K approach)", fontsize=10)
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ1_RECUPERATOR_SENSITIVITY.png")
    return res


def plow_sensitivity(fig_dir):
    plows = [3.5e6, 4.0e6, 5.0e6, 6.0e6, 7.0e6]
    res = {}
    for sk in SITE_ORDER:
        s = SITES[sk]; T_src = s["T_src"]; res[sk] = []
        for P in plows:
            c = tco2_cycle(T_src - DT_HOT_END, P_high=tco2_high_pressure(T_src), P_low=P, eps=0.70)
            hx = heater_for_cycle(c, T_src)
            res[sk].append(dict(P_low=P / 1e6, T_cond=c["T_cond_C"], eta=c["eta_th"] * 100, bwr=c["bwr"] * 100,
                                w_net=c["w_net"], T_gas_out=hx["T_gas_out"],
                                w_gas=c["eta_th"] * CP_GAS * (T_src - hx["T_gas_out"])))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    site_col = {"CS2": "#BB9B1C", "CS1": "#51247A", "CS3": "#0064B4"}
    for a, (k, yl) in zip(ax, [("T_cond", "Condensing temperature (°C)"), ("eta", "Thermal efficiency (%)"), ("bwr", "Back-work ratio (%)")]):
        for sk in SITE_ORDER:
            a.plot([d["P_low"] for d in res[sk]], [d[k] for d in res[sk]], marker="^", color=site_col[sk], label=SITES[sk]["name"])
        a.set_xlabel("Low-side pressure P_low (MPa)"); a.set_ylabel(yl); a.set_xticks([3.5, 4.0, 5.0, 6.0, 7.0]); a.legend(fontsize=8)
    ax[0].axhspan(20, 32, color="#CC0000", alpha=0.08); ax[0].text(3.55, 21, "cooling-tower water,\nAustralian sites: ~20–32 °C", fontsize=7.5, color="#CC0000")
    fig.suptitle("tCO₂ Rankine — low-side pressure sensitivity (ε = 0.70)", fontsize=10)
    fig.tight_layout(); save(fig, f"{fig_dir}/RQ1_PLOW_SENSITIVITY.png")
    return res


def sco2_inlet_sensitivity():
    return {T1: [round(sco2_cycle(T, T1_C=T1)["eta_th"] * 100, 2) for T in (200, 350, 500, 600)] for T1 in (32, 36, 40)}


def run(fig_dir):
    print("RQ1 ...")
    rows = sweep(); plot_sweep(rows, fig_dir)
    return dict(sweep=rows, exergy=exergy(fig_dir), pinch=pinch(fig_dir),
                recup=recuperator_sensitivity(fig_dir), plow=plow_sensitivity(fig_dir),
                sco2_inlet=sco2_inlet_sensitivity(), exergy_sites=exergy_sites(), chiller=chiller_penalty())
