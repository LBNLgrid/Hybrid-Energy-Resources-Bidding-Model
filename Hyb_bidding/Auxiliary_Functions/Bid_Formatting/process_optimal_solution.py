import numpy as np
import pyomo.environ as pyo


def get_solution(bid, priceDA, rt_vs_da, PRICE_STEPS,bid_type):
    (nS, nT) = priceDA.shape
    (nSp, nTp) = PRICE_STEPS.shape
    phi = np.zeros(nS)
    soc_f = np.zeros(nS)
    ph_da_s = np.zeros((nSp, nT))
    pesr_da = np.zeros((nS, nT))
    pesr_rt = np.zeros((nS, nT))
    pesr_sc = np.zeros((nS, nT))
    ph_da = np.zeros((nS, nT))
    ph_rt = np.zeros((nS, nT))
    ph_sc = np.zeros((nS, nT))
    pg_da = np.zeros((nS, nT))
    pg_rt = np.zeros((nS, nT))
    pg_sc = np.zeros((nS, nT))
    pesm_c = np.zeros((nS, nT))
    pesm_d = np.zeros((nS, nT))
    soc = np.zeros((nS, nT))
    Pc = np.zeros((nS, nT))
    ep = np.zeros((nS, nT))
    en = np.zeros((nS, nT))

    for t in range(nT):
        for s in range(nS):
            phi[s] = pyo.value(bid.phi[s])
            soc_f[s] = pyo.value(bid.SOC_f[s])
            pesr_sc[s, t] = pyo.value(bid.Pesr_sc[s, t])
            ph_da[s, t] = pyo.value(bid.Ph_da[s, t])
            pg_sc[s, t] = pyo.value(bid.Pg_sc[s, t])
            pesm_c[s, t] = pyo.value(bid.Esm_ch[s, t])
            pesm_d[s, t] = pyo.value(bid.Esm_dis[s, t])
            soc[s, t] = pyo.value(bid.SOC[s, t])
            Pc[s, t] = pyo.value(bid.Pc[s, t])
            ep[s, t] = pyo.value(bid.e_p[s, t])
            en[s, t] = pyo.value(bid.e_n[s, t])
            if rt_vs_da == 1:
                pg_rt[s, t] = pyo.value(bid.Pg_rt[s, t])
                pesr_rt[s, t] = pyo.value(bid.Pesr_rt[s, t])
                ph_rt[s, t] = pyo.value(bid.Ph_rt[s, t])
                pesr_da[s, t] = pyo.value(bid.Pesr_sc[s, t])
                pg_da[s, t] = pyo.value(bid.Pg_da[s, t])
                pesr_da[s, t] = pyo.value(bid.Pesr_da[s, t])
                ph_sc[s, t] = pyo.value(bid.Ph_sc[s, t])
        for k in range(nSp):
            if bid_type==2 :
               ph_da_s[k, t] = pyo.value(bid.ph_da_steps[k, t])

    class Sol:
        def __init__(self, soc, pesr_da, pesr_sc, pesr_rt, Pc,pg_rt,ph_da):
            self.soc = soc
            self.pesr_da = pesr_da
            self.pesr_sc = pesr_sc
            self.pesr_rt = pesr_rt
            self.curtail = Pc
            self.pg_rt = pg_rt
            self.ph_da = ph_da

    return Sol(soc, pesr_da, pesr_sc, pesr_rt, Pc,pg_rt,ph_da)
