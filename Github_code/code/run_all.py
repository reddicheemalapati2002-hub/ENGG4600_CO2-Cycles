"""
Run every analysis in the thesis and write results/ and figures/.

    python run_all.py            (from the code/ directory)

Outputs
  ../figures/*.png            all figures referenced by the thesis
  ../results/results.json     every number used in the thesis
  ../results/summary.txt      human-readable summary tables
"""
import json, os, sys, time
import numpy as np
import run_rq1, run_rq2, run_rq3, run_rq4
from co2whr.sites import SITES, SITE_ORDER, CYCLES, LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures"); RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)


def main():
    t0 = time.time()
    R = {}
    R["rq1"] = run_rq1.run(FIG)
    designs, R["rq2"] = run_rq2.run(FIG)
    R["rq3"] = run_rq3.run(designs, FIG)
    R["rq4"] = run_rq4.run(FIG)
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if hasattr(o, "tolist") else str(o)))
    write_summary(R)
    print(f"done in {time.time()-t0:.0f} s")


def write_summary(R):
    L = []
    P = L.append
    P("RQ1 — parametric sweep (thermal efficiency %, BWR %)")
    P(f"{'TIT':>5} {'SRC':>7} {'sCO2':>7} {'tCO2':>7} | {'SRC':>6} {'sCO2':>6} {'tCO2':>6}")
    for r in R["rq1"]["sweep"]:
        P(f"{r['TIT']:>5} {r['SRC']:7.2f} {r['sCO2']:7.2f} {r['tCO2']:7.2f} | {r['SRC_bwr']:6.2f} {r['sCO2_bwr']:6.2f} {r['tCO2_bwr']:6.2f}")
    P("\nRQ1 — exergy at TIT 400 C (kJ/kg wf | kJ per kJ net work)")
    for k, e in R["rq1"]["exergy"].items():
        d = e["destruction"]; p = e["per_kJ_work"]
        P(f"{k:5s} eta_th={e['eta_th']*100:5.2f} eta_ex={e['eta_ex']*100:5.2f} total={e['total']:7.1f} ({e['total_per_kJ_work']:.3f}/kJ)  "
          + "  ".join(f"{c}={d[c]:.1f}({p[c]:.3f})" for c in d) + f"  heater share={e['heater_share']*100:.1f}%  closure={e['closure']:.1e}")
    P("\nRQ1 — tCO2 heater pinch (eps=0)")
    for sk, p in R["rq1"]["pinch"].items():
        P(f"{sk}: P_high={p['P_high']:.1f} MPa pump exit {p['T_pump_exit']:.1f} C, T_pc={p['T_pc']:.1f} C cp_peak={p['cp_peak']:.2f}, gas out {p['T_gas_out']:.1f} C, Q={p['Q']:.1f} kJ/kg, "
          f"dT min {p['dT_min']:.1f} K [{p['pinch']}], dT max {p['dT_max']:.1f} K at {p['T_at_dTmax']:.0f} C, LMTD_eff {p['LMTD_eff']:.1f} K | eps=0.70: hx_in {p['eps070_hx_in']:.1f} C, gas out {p['eps070_T_gas_out']:.1f} C, LMTD_eff {p['eps070_LMTD_eff']:.1f} K")
    P("\nRQ1 — recuperator sensitivity (tCO2)")
    for sk, rows in R["rq1"]["recup"].items():
        for d in rows:
            P(f"{sk} eps={d['eps']:.2f} eta={d['eta']:6.2f} bwr={d['bwr']:5.2f} w_net={d['w_net']:6.2f} q_in={d['q_in']:6.1f} q_rec={d['q_recup']:6.1f} hx_in={d['hx_in']:6.1f} gas_out={d['T_gas_out']:6.1f} w_gas={d['w_gas']:6.2f} rec_dTmin={d['rec_dTmin']:.1f} UA_rec={d['UA_recup']:.0f}")
    P("\nRQ1 — P_low sensitivity (tCO2, eps=0.70)")
    for sk, rows in R["rq1"]["plow"].items():
        for d in rows:
            P(f"{sk} P_low={d['P_low']:.1f} T_cond={d['T_cond']:5.2f} eta={d['eta']:6.2f} bwr={d['bwr']:5.2f} w_net={d['w_net']:6.2f} gas_out={d['T_gas_out']:6.1f} w_gas={d['w_gas']:6.2f}")
    P("\nRQ1 — exergy at site design points (per kJ net work)")
    for sk, d in R["rq1"]["exergy_sites"].items():
        for cyc, e in d.items():
            P(f"{sk} {cyc:5s} eta_th={e['eta_th']*100:5.2f} eta_ex={e['eta_ex']*100:5.2f} total/kJ={e['total_per_kJ_work']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in e["per_kJ_work"].items()))
    P("\nRQ1 — indicative chiller parasitic for tCO2 condensing at 14.3 C (fraction of net work)")
    for sk, d in R["rq1"]["chiller"].items():
        P(f"{sk}: " + str({k: round(v, 3) for k, v in d['tCO2'].items()}))
    P("\nRQ1 — sCO2 compressor inlet sensitivity (eta % at 200/350/500/600 C): " + str(R["rq1"]["sco2_inlet"]))
    P("\nRQ2 — site design points")
    for sk in SITE_ORDER:
        for cyc in CYCLES:
            d = R["rq2"]["design"][sk][cyc]
            P(f"{sk} {cyc:5s} P_hi={d['P_high_MPa']:5.2f} TIT={d['TIT']:.0f} eta={d['eta']:5.2f} bwr={d['bwr']:5.1f} w_net={d['w_net']:6.1f} hx_in={d['hx_in']:6.1f} gas_out={d['T_gas_out']:6.1f} "
              f"w_gas={d['w_gas']:6.2f} W_site={d['W_site_kW']:7.0f} kW Q_rec={d['Q_recovered_kW']:7.0f} kW m_wf={d['m_wf']:6.2f} UA_h={d['UA_heater']:7.0f} UA_r={d['UA_recup']:6.0f} UA_c={d['UA_cooler']:7.0f} LMTD_h={d['LMTD_heater']:5.1f} rec_dTmin={d['rec_dTmin']:.1f} Tsat={d['T_sat']} Tcond={d['T_cond']:.1f} pinch={d['pinch']}")
    P("\nRQ2 — SP / VFR")
    for sk in SITE_ORDER:
        P(f"{sk}: " + "  ".join(f"{cyc} SP={R['rq2']['sp_vfr'][sk][cyc]['SP']:.4f} VFR={R['rq2']['sp_vfr'][sk][cyc]['VFR']:.2f}" for cyc in CYCLES))
    P("\nRQ2 — LMTD vs segmented (tCO2 heater, eps=0)")
    for sk in SITE_ORDER:
        l = R["rq2"]["lmtd"][sk]; P(f"{sk}: A_seg={l['A_seg']:.1f} A_2pt={l['A_2pt']:.1f} over={l['over_pct']:.1f}% LMTD_eff={l['LMTD_eff']:.1f} LMTD_2pt={l['LMTD_2pt']:.1f}")
    P("\nRQ2 — capital cost TEC AUD/kW (100 kW / 500 kW / 1 MW / 5 MW) and 1 MW breakdown kAUD (equipment, before +35%)")
    for sk in SITE_ORDER:
        for cyc in CYCLES:
            c = R["rq2"]["capital"][sk][cyc]
            P(f"{sk} {cyc:5s} " + " ".join(f"{c[str(W)]['TEC_per_kW'] if str(W) in c else c[W]['TEC_per_kW']:7.0f}" for W in (100, 500, 1000, 5000))
              + "   1MW items: " + " ".join(f"{k}={v:.0f}" for k, v in (c[1000] if 1000 in c else c['1000'])['items'].items()) + f"  HXshare={(c[1000] if 1000 in c else c['1000'])['hx_share']*100:.0f}%")
    P("\nRQ3 — fixed-source economics")
    for sk in SITE_ORDER:
        for cyc in CYCLES:
            e = R["rq3"][sk][cyc]
            red = e["hx_reduction_to_match"]
            P(f"{sk} {cyc:5s} W={e['W_kW']:7.0f} kW TEC={e['TEC_kAUD']:8.0f} kAUD ({e['TEC_per_kW']:6.0f}/kW) E={e['E_MWh']:7.0f} MWh LCOE={e['LCOE']:6.1f} NPV={e['NPV_kAUD']:8.0f} kAUD "
              f"PB_disc={e['payback_disc']:5.1f} PB_simple={e['payback_simple']:5.1f} IRR={e['IRR']*100:5.1f}% | MC P={e['mc']['p_viable']:5.1f}% NPV P10/50/90={e['mc']['npv_p10']:.0f}/{e['mc']['npv_p50']:.0f}/{e['mc']['npv_p90']:.0f} LCOE P10/50/90={e['mc']['lcoe_p10']:.0f}/{e['mc']['lcoe_p50']:.0f}/{e['mc']['lcoe_p90']:.0f} | HX cut to match best: {('n/a' if red is None else f'{red*100:.0f}%')}")
        P(f"   best LCOE: {R['rq3'][sk]['best_lcoe']}  best NPV: {R['rq3'][sk]['best_npv']}")
    P("\nRQ4 — part load (eta % at 100/80/60/40/20 % load; retention)")
    for sk in SITE_ORDER:
        for cyc in CYCLES:
            r = R["rq4"]["sweep"][sk][cyc]
            P(f"{sk} {cyc:5s} eta=" + " ".join(f"{v:6.2f}" for v in r["eta"]) + "  ret=" + " ".join(f"{v:.3f}" for v in r["retention"]))
    for cyc, s in R["rq4"]["scores"].items():
        P(f"Marchionni {cyc}: turndown={s['turndown']:.2f} r60={s['r60']:.3f} r40={s['r40']:.3f} startup={s['startup']:.2f} TOTAL={s['total']:.3f}")
    with open(os.path.join(RES, "summary.txt"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
