#! pip install pyomo
import ipdb
import os
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import cplex
from docplex.mp.model import Model
#Model2 = cplex.Cplex()
#Model2.parameters.mip.tolerances.mipgap.set(float(0.1))
#df = pd.read_csv("C:/hybrid_bidding/software_module-20231023T230959Z-001/software_module/data/EPRI_final_analysis/Case 24, 28_ High VRE, High Hyb, Uncstr GC/LBNL_SCH_TMP1011.csv")
#print(df.x__Schedule)
#dates=datetime(2019,7,1,"TimeZone","America/New_York"):datetime(2019,7,31,"TimeZone","America/New_York");
## Input and parameters
cvar_weight = 1 # %0 to 1 with 0=do not consider risk and 1=fully consider risk
gridcharge = 1 #% 0=No grid charging allowed, 1=Grid charging unrestricted
ngenscen_sample=100 #number of scenarios to sample from historical distribution
nscen=200
ngenscen=20 #final number of generation scenarios to consider in bidding
fcast_band=.05 #for generation scenarios, proportion of GEN.max
err_band=.05 #for generation scenarios, proportion of GEN.max
gen_fcast_prob=0.8 #probability of given DA forecast
nprevdays=[14,21] #weekday, weekend
avg_wmae_tgt=0.05
#compute bids - may need to update once
bid_type=2 #(2=stairstep, 1=self-schedule)
nscen_breaks=6
break_mode="mid"
price_floor=-150
price_ceil=1000
soc0_ub=[0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1] #initial SOC bands
n_soc=len(soc0_ub)-1
ndates=1
#n_soc=1 ############################correct
#print(n_soc)
soc0_pos_in_range=0.5 #in (0,1] => (lower bound, upper bound]
x_step_min=0.1 #Treats two points on the bid curve as equal if x (MW) component differs by less than this threshold.
                #Points are removed from curve accordingly.
q1min=1e-10 #smallest positive quantity bid allowed by PSO
#compute bids - should not need to edit
cvar_confid = 0.95
x=np.ones(48)
Time_p=48
eta_plus = 0.5*np.ones(48) #concentrates bidding in DA (don't change)
eta_minus= 1.5*np.ones(48) #concentrates bidding in DA (don't change)
dev_plus_addifzero = -.5 #ensures costs for deviation when price scen is 0
dev_minus_addifzero = .5 #ensures costs for deviation when price scen is 0
p_thresh_heur=[0.9632,29.4278]
p_thresh_heur_pcentile=[0.2,0.8] #price percentiles (of annual prices) used to set low/high price heuristic
pHoriz_scale=2
price_ceil=1000
#for high/low price heuristic
#Process input data
#[POI,GEN,ESR] = process_PSOdata_config(inj_path,esr_path,hyb_name,gen_name,esr_name);
#[gen_data,gen_col] = process_PSOfile_gen(gendata_path,sched_inj_path,gen_name);
#[price_data,price_col] = process_PSOfile_2Rprices(pricedata_path,int_time_path,hyb_name, hyb_name);%esr_name,gen_name);
class ESR:
    pass
esr=ESR()
esr.init=6
esr.max=240
esr.min=0
esr.effCrg=0.922
esr.effDis=0.922
esr.opCost=0
esr.power=60
#print (esr.max)
gen=ESR()
gen.max =125
gen.opCost=0
POI=185
from opt_prob import get_stoch_cvar_stairstep_prob_noRT
#from opt_prob_RT import get_stoch_cvar_stairstep_prob_RT
from bid_curve import get_bidcurve
from high_low_price_heuristc import get_high_low_price_heuristic
obj_result=np.zeros(n_soc)
da=np.empty([288, 1])
priceRTadd=0.01*np.ones((200,48))
## include for more scneario
class pqt_bids_base:
    def __init__(self, bid):
        self.attribute1 = bid
bid_instances_1 =[]
object_instances=[]
for ii in range(ndates):
    ii=1
    print(ii)
    PRICE_STEP = pd.read_excel(r"C:\hybrid_bidding\software_module-20231023T230959Z-001\software_module\LBL_1R_Bidding\PRICE_STEPS2.xlsx",sheet_name=ii, engine='openpyxl')
    PRICE_STEPS = PRICE_STEP.to_numpy()
    pDA = pd.read_excel(r"C:\hybrid_bidding\software_module-20231023T230959Z-001\software_module\LBL_1R_Bidding\pricedata2.xlsx",sheet_name=ii,engine='openpyxl')
    genDA = pd.read_excel(r"C:\hybrid_bidding\software_module-20231023T230959Z-001\software_module\LBL_1R_Bidding\gendata2.xlsx",sheet_name=ii,engine='openpyxl')
    probSc = pd.read_excel(r"C:\hybrid_bidding\software_module-20231023T230959Z-001\software_module\LBL_1R_Bidding\probS2.xlsx",sheet_name=ii,engine='openpyxl')
    priceDA = pDA.to_numpy()
    gen_data= genDA.to_numpy()
    probS=probSc.to_numpy()
    (nS, nT) = PRICE_STEPS.shape
    bid_instances=[]
    for jj in range(1):
        esr.init = (soc0_ub[jj] + soc0_pos_in_range * (soc0_ub[jj+1]- soc0_ub[jj])) * esr.max
        bid,pqt_format_bids,instance=get_bidcurve(esr,gen,POI,1,1,eta_plus,eta_minus,dev_plus_addifzero, dev_minus_addifzero, priceDA, priceRTadd, gen_data, probS,cvar_weight,cvar_confid,gridcharge,bid_type,PRICE_STEPS,np,pyo,Time_p,nscen_breaks)
        object_instances.append(instance)
        bid_instances.append(pqt_format_bids)
    bid_instances_1.append(bid_instances)
    if (bid_type==2) :
       pqt_bids_full=get_high_low_price_heuristic(bid_instances_1[ii], object_instances, np.tile(priceDA,(n_soc, 1)), np.tile(gen_data, (n_soc, 1)), np.tile(probS, (n_soc, 1)), np.arange(0, 24), p_thresh_heur[0], p_thresh_heur[1], pHoriz_scale, price_ceil, esr, POI, gridcharge,n_soc)
    # for kk in bid_instances_1:
    #     print(kk.attribute1)



    df=pd.DataFrame(bid_instances_1[0][0])
    df.to_excel("C:\hybrid_bidding\software_module-20231023T230959Z-001\software_module\LBL_1R_Bidding\PRdf.xlsx")

                #da2_temp[t, s] = pyo.value(bid3.ph_da_steps[s, t])
    #             plt.xlabel("SOC_level")
    #             plt.ylabel("Objective Value")
    # for s in range(200):
    #        ph2[s] = pyo.value(bid.phi[s])
    #                 pesr_sc[s] = pyo.value(bid.td[s])
    #     da[jj:6*(jj+1)] = da_temp[:,1][0]  #np.concatenate((da,da_temp[:,1]), axis=0)
    plt.plot(plt.plot(bid_instances_1[0][0]))
    plt.show()
    #plt.plot(obj_result)

