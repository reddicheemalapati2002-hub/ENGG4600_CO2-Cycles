import sys; sys.path.insert(0,'/home/claude/code_src')
import numpy as np, CoolProp.CoolProp as CP
# Reproduce the thesis pinch calculation exactly (tco2_pinch_diagram), CS2
P_hi, P_lo, Tsrc, eta_p = 12.8e6, 5e6, 200, 0.80
h1=CP.PropsSI("H","P",P_lo,"Q",0,"CO2"); s1=CP.PropsSI("S","P",P_lo,"Q",0,"CO2")
h2s=CP.PropsSI("H","P",P_hi,"S",s1,"CO2"); h2=h1+(h2s-h1)/eta_p
Tp=CP.PropsSI("T","P",P_hi,"H",h2,"CO2"); Tpc=Tp-273.15; Tsink=Tpc+10
Tco2=np.linspace(Tp, Tsrc+273.15-10, 50)-273.15
h=np.array([CP.PropsSI("H","T",T+273.15,"P",P_hi,"CO2") for T in Tco2])/1000; hn=h-h[0]
# (a) thesis method: source linear in INDEX
src_idx=np.array([Tsink+(Tsrc-Tsink)*i/49 for i in range(50)])
d_idx=src_idx-Tco2
print("THESIS METHOD: dT min %.6f max %.6f  argmin idx %d -> T_CO2 %.1f C"%(d_idx.min(),d_idx.max(),d_idx.argmin(),Tco2[d_idx.argmin()]))
# (b) correct counterflow: source linear in heat transferred
src_q=Tsink+(Tsrc-Tsink)*hn/hn[-1]
d_q=src_q-Tco2
print("CORRECT METHOD: dT min %.2f at T_CO2 %.1f C | max %.2f at T_CO2 %.1f C"%(d_q.min(),Tco2[d_q.argmin()],d_q.max(),Tco2[d_q.argmax()]))
print("ends:",round(d_q[0],2),round(d_q[-1],2))
