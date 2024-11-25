def get_bidcurve(esr,gen,POI,rt_vs_da,enforce_da_rt,eta_plus,eta_minus,dev_plus_addifzero, dev_minus_addifzero, priceDA, priceRTadd, gen_data, probS,cvar_weight,cvar_confid,gridcharge,type,PRICE_STEPS,np,pyo,Time_p,nscen_breaks) :
   if (type==0) :
      print(type)
   elif (type==1):
      print("1")
   elif (type==2):
      if len(priceDA) == 0:
         print("must supply price steps")
      elif ( rt_vs_da==0 and enforce_da_rt==1):
            from opt_prob import get_stoch_cvar_stairstep_prob_noRT
            bid, bid1 = get_stoch_cvar_stairstep_prob_noRT(esr, gen, POI, eta_plus, eta_minus, dev_plus_addifzero, dev_minus_addifzero, priceDA, gen_data, probS, cvar_weight,cvar_confid, gridcharge, PRICE_STEPS, np, pyo, Time_p,nscen_breaks)
      else :
            from opt_prob_RT import get_stoch_cvar_stairstep_prob_RT
            bid,bid1=get_stoch_cvar_stairstep_prob_RT(esr, gen, POI, 1, 1, eta_plus, eta_minus,dev_plus_addifzero,dev_minus_addifzero, priceDA, priceRTadd, gen_data, probS, cvar_weight,cvar_confid,gridcharge,PRICE_STEPS,np,pyo,Time_p,nscen_breaks)
   else :
      print("type must be 0, 1, or 2")

   class Sol:
       def __init__(self, soc, pesr_da,pesr_sc,pesr_rt,curtail ):
           self.soc = soc
           self.pesr_da = pesr_da
           self.pesr_sc = pesr_sc
           self.pesr_rt = pesr_rt
           self.curtail = curtail

   (nS1, nT1) = priceDA.shape
   #phi = np.zeros(200)
   #soc_f = np.zeros(200)
   #ph_da_s = np.zeros((6, 48))
   pesr_da = np.zeros((nS1, nT1))
   pesr_rt = np.zeros((nS1, nT1))
   pesr_sc = np.zeros((nS1, nT1))
   # ph_da = np.zeros((200, 48))
   # ph_rt = np.zeros((200, 48))
   # ph_sc = np.zeros((200, 48))
   # pg_da = np.zeros((200, 48))
   # pg_rt = np.zeros((200, 48))
   # pg_sc = np.zeros((200, 48))
   # pesm_c = np.zeros((200, 48))
   # pesm_d = np.zeros((200, 48))
   soc = np.zeros((nS1, nT1))
   Pc = np.zeros((nS1, nT1))
   # ph2 = np.zeros(200)
   for t in range(nT1):
       for s in range(nS1):
           #phi[s] = pyo.value(bid.phi[s])
           #soc_f[s] = pyo.value(bid.SOC_f[s])
           # pesr_da[sc, t] = pyo.value(bid.ph_da_steps[s, t])
           pesr_sc[s, t] = pyo.value(bid.Pesr_sc[s, t])
           pesr_rt[s, t] = pyo.value(bid.Pesr_rt[s, t])
           pesr_da[s, t] = pyo.value(bid.Pesr_da[s, t])
           # ph_sc[s, t] = pyo.value(bid.Ph_sc[s, t])
           # ph_rt[s, t] = pyo.value(bid.Ph_rt[s, t])
           # ph_da[s, t] = pyo.value(bid.Ph_da[s, t])
           # pg_sc[s, t] = pyo.value(bid.Pg_sc[s, t])
           # pg_rt[s, t] = pyo.value(bid.Pg_rt[s, t])
           # pg_da[s, t] = pyo.value(bid.Pg_da[s, t])
           # pesm_c[s, t] = pyo.value(bid.Esm_ch[s, t])
           # pesm_d[s, t] = pyo.value(bid.Esm_dis[s, t])
           soc[s, t] = pyo.value(bid.SOC[s, t])
           Pc[s, t] = pyo.value(bid.Pc[s, t])
   instance = Sol(soc,pesr_da,pesr_sc,pesr_rt,Pc)
   if(type<2):
       (nS, nT) = priceDA.shape
       pqt_format_bids = np.empty
   elif(type==2) :
       (nS, nT) = PRICE_STEPS.shape
       pqt_format_bids = np.empty([0,3])
       #pq_bids_by_hr=np.zeros((nS, nT))
       ph_da_step = np.zeros((nS, nT))
       #pr_sort=PRICE_STEPS
       pq_bids_by_hr=PRICE_STEPS.copy()
       for t in range(nT) :
           for sc in range(nS) :
               ph_da_step[sc,t]=pyo.value(bid.ph_da_steps[sc,t])
           pqt_format_bids=np.vstack((pqt_format_bids,np.column_stack((pq_bids_by_hr[:,t],ph_da_step[:,t],t*np.ones((6,1))))))
   return bid,pqt_format_bids, instance