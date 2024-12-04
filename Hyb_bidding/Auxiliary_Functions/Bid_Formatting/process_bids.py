
import numpy as np
import pyomo.environ as pyo
def get_process_sol(bid,priceDA,opt_data,PRICE_STEPS):
     (nS, nT) = priceDA.shape
     pqt_format_bids = np.empty
     Ph_da=np.zeros((nS,nT))
     for s in range (nS):
         for t in range (nT):
              Ph_da[s,t]= pyo.value(bid.Ph_da[s, t])

     if (opt_data["bid_type"] < 2):
            pqt_format_bids = np.empty([0,3])
            for t in range(nT) :
                order = np.argsort(priceDA[:,t], axis=0)
                pqt_format_bids=np.vstack((pqt_format_bids,np.column_stack((priceDA[order,t],Ph_da[order,t],t*np.ones((nS,1))))))

     elif(opt_data["bid_type"]==2) :
            (nS2, nT2) = PRICE_STEPS.shape
            pqt_format_bids = np.empty([0,3])
            #pq_bids_by_hr=np.zeros((nS, nT))
            ph_da_step = np.zeros((nS2, nT))
            pr_sort=np.sort(PRICE_STEPS, axis=0)
            pq_bids_by_hr=pr_sort.copy()
            for t in range(nT2) :
                for sc in range(nS2) :
                    ph_da_step[sc,t]=pyo.value(bid.ph_da_steps[sc,t])
                pqt_format_bids=np.vstack((pqt_format_bids,np.column_stack((pq_bids_by_hr[:,t],ph_da_step[:,t],t*np.ones((nS2,1))))))
     return pqt_format_bids
