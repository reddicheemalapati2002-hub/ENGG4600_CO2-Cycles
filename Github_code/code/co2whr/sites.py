"""
Australian case-study definitions (Table 3.2 of the thesis).

Each site is a fixed waste-heat stream.  The RQ3 economics are evaluated on a
fixed-source basis: the plant output is whatever each cycle can recover from
the stream, subject to a 10 K minimum approach in the primary heater.
"""

CP_GAS = 1.10          # kJ/(kg K)  — exhaust gas, treated as an ideal gas
DT_APPROACH = 10.0     # K          — minimum approach / pinch in all heat exchangers
DT_HOT_END = 10.0      # K          — turbine inlet = source temperature − 10 K

SITES = {
    "CS2": dict(name="CS2: Gas 200°C",    place="Pilbara, WA",   T_src=200.0,
                m_gas=30.0, price=130.0, industry="LNG gas-processing intercooler"),
    "CS1": dict(name="CS1: Cement 350°C", place="Gladstone, QLD", T_src=350.0,
                m_gas=50.0, price=90.0,  industry="Cement preheater exhaust"),
    "CS3": dict(name="CS3: Steel 500°C",  place="Sydney, NSW",    T_src=500.0,
                m_gas=20.0, price=100.0, industry="Electric arc furnace off-gas"),
}
SITE_ORDER = ["CS2", "CS1", "CS3"]       # ascending source temperature
CYCLES = ["SRC", "sCO2", "tCO2"]

COLORS = {"SRC": "#51247A", "sCO2": "#0064B4", "tCO2": "#BB9B1C"}
MARKERS = {"SRC": "o", "sCO2": "s", "tCO2": "^"}
LABELS = {"SRC": "SRC", "sCO2": "sCO₂", "tCO2": "tCO₂"}
