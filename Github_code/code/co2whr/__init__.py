"""
co2whr — thermodynamic, sizing, cost, economic and off-design models for the
ENGG4600 thesis "Performance of CO2 Cycle in the Context of Energy Transition"
(V. R. Cheemalapati, UQ, 2026).

Modules
-------
cycles     : steady-state state-point models for SRC, sCO2 Brayton, tCO2 Rankine
heatex     : counter-flow heat exchanger integration (duty-linear gas profile,
             pinch search, segmented UA), recuperator UA, LMTD comparison
costs      : equipment cost correlations, escalation, total erected cost
economics  : LCOE, NPV, discounted payback, IRR, Monte Carlo
offdesign  : part-load (Stodola) and Marchionni flexibility scoring
sites      : the three Australian case-study definitions
"""
__version__ = "2.0 (corrected, October 2026)"
