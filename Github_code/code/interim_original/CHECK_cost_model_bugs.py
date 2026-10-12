import sys; sys.path.insert(0,'/home/claude/code_src')
import numpy as np, CoolProp.CoolProp as CP
from tco2_model import tco2_rankine_cycle
for label,T_src,P_hi in [("CS2",200,12.8e6),("CS1",350,20e6),("CS3",500,20e6)]:
    T_K=T_src+273.15
    # SRC boiler, exactly as rq2_cost.py
    P_src_hi = min(10e6, 0.95*CP.PropsSI("P","T",T_K,"Q",1,"Water")) if T_K < CP.PropsSI("Tcrit","Water") else 20e6
    Tsat=CP.PropsSI("T","P",P_src_hi,"Q",1,"Water")-273.15
    dT1=T_src-Tsat; dT2=T_src*0.1+10
    LMTD_b=max((dT1-dT2)/np.log(max(dT1/max(dT2,1),1.01)),5)
    # sCO2 heater, exactly as rq2_cost.py
    dT1s=T_src-T_K+273.15-10; dT2s=T_src*0.12
    LMTD_s=max(abs(dT1s-dT2s)/np.log(max(abs(dT1s/max(dT2s,1)),1.01)),5)
    LMTD_s_used=max(LMTD_s,10)
    # tCO2 heater: index-linear source -> effective LMTD
    r=tco2_rankine_cycle(T_src,P_hi,5e6,0.85,0.80,0.70)
    Tin=r["state_2"]["T_C"]; Ts=Tin+10
    T=np.linspace(Tin+273.15,T_K-10,50); h=np.array([CP.PropsSI("H","T",t,"P",P_hi,"CO2") for t in T])
    Tc=T-273.15; srcp=np.array([Ts+(T_src-Ts)*i/49 for i in range(50)])
    UA=sum((h[i+1]-h[i])/ (max(srcp[i+1]-Tc[i+1],.1) if abs(max(srcp[i+1]-Tc[i+1],.1)-max(srcp[i]-Tc[i],.1))<.01 else ((max(srcp[i+1]-Tc[i+1],.1)-max(srcp[i]-Tc[i],.1))/np.log(max(srcp[i+1]-Tc[i+1],.1)/max(srcp[i]-Tc[i],.1)))) for i in range(49))
    print(f"{label}: SRC boiler dT1={dT1:6.1f} dT2={dT2:5.1f} -> LMTD used {LMTD_b:7.1f} K | "
          f"sCO2 heater dT1={dT1s:6.1f} dT2={dT2s:5.1f} -> LMTD used {LMTD_s_used:8.1f} K | "
          f"tCO2 heater effective LMTD {(h[-1]-h[0])/UA:5.2f} K")
