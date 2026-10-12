These are the Interim Report model files exactly as written by the student
(October 2026), kept for comparison. They contain the errors listed in
Table 8.3 of the thesis and must NOT be used to generate thesis results.

CHECK_pinch_bug.py and CHECK_cost_model_bugs.py reproduce, with the
original code's own formulas, the two main defects:
  * pinch diagram: gas profile linear in node index -> constant 10 K gap
  * rq2_cost.py:    sCO2 heater dT cancels to -10 K; boiler dT ad hoc
