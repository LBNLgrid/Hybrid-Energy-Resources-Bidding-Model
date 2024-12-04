import time
import pyomo.environ as pyo
import numpy as np
from Auxiliary_Functions.Bid_Computation.Opt_Prob_DA_MILP import get_stoch_cvar_stairstep_prob_noRT
from Auxiliary_Functions.Bid_Computation.Opt_Prob_RT_MILP import get_stoch_cvar_stairstep_prob_RT
from Auxiliary_Functions.Bid_Computation.Opt_Prob_DA_LP import get_stoch_cvar_stairstep_prob_noRT_LP
from Auxiliary_Functions.Bid_Computation.Opt_Prob_RT_LP  import get_stoch_cvar_stairstep_prob_RT_LP
from Auxiliary_Functions.Bid_Computation.Opt_Prob_Self_Schedule import get_stoch_cvar_selfsched_prob
from Auxiliary_Functions.Bid_Computation.Opt_Prob_Regular_Bid import get_stoch_cvar_regular_prob
from Auxiliary_Functions.Bid_Formatting.process_bids import get_process_sol
from Auxiliary_Functions.Bid_Formatting.process_optimal_solution import get_solution
def get_bidcurve(esr, gen, POI, opt_data, priceDA, priceRTadd, gen_data, probS,PRICE_STEPS,Price_Steps_RT):
   if (opt_data["bid_type"]==0) :
       bid,bid1=get_stoch_cvar_regular_prob(esr, gen, POI, opt_data, priceDA, priceRTadd, gen_data, probS)
   elif (opt_data["bid_type"]==1):
       bid,bid1=get_stoch_cvar_selfsched_prob(esr, gen, POI, opt_data, priceDA, priceRTadd, gen_data, probS)
   elif (opt_data["bid_type"]==2):
      if len(priceDA) == 0:
         raise ValueError("must supply price steps")
      else:
          if (opt_data["formulation"]=="MILP"):
              if ( opt_data["rt_vs_da"]==0 and opt_data["enforce_da_rt"]==1):
                  t_s1 = time.time()
                  bid, bid1 = get_stoch_cvar_stairstep_prob_noRT(esr, gen, POI, opt_data, priceDA, priceRTadd, gen_data, probS,PRICE_STEPS)
                  t1 = time.time() - t_s1
              else:
                  bid,bid1=get_stoch_cvar_stairstep_prob_RT(esr, gen, POI, opt_data, priceDA, priceRTadd, gen_data, probS,PRICE_STEPS)
          else:
              if (opt_data["rt_vs_da"] == 0 and opt_data["enforce_da_rt"] == 1):
                  t_s1 = time.time()
                  bid, bid1 = get_stoch_cvar_stairstep_prob_noRT_LP(esr, gen, POI, opt_data, priceDA, priceRTadd, gen_data, probS,PRICE_STEPS)
                  t1 = time.time() - t_s1
              else:
                  bid, bid1 = get_stoch_cvar_stairstep_prob_RT_LP(esr, gen, POI, opt_data, priceDA, priceRTadd, gen_data, probS,PRICE_STEPS)
   else:
      print("type must be 0, 1, or 2")
   opt_sol=get_solution(bid,priceDA,opt_data["rt_vs_da"],PRICE_STEPS,opt_data["bid_type"])
   pqt_format_bids= get_process_sol(bid,priceDA,opt_data,PRICE_STEPS)
   # For real time bids
   if Price_Steps_RT.size == 0:
       pqt_format_bids_RT=np.array([])
   else:
       pqt_format_bids_RT= get_process_sol(bid,priceRTadd,opt_data,Price_Steps_RT)
   return bid,opt_sol,pqt_format_bids,pqt_format_bids_RT
