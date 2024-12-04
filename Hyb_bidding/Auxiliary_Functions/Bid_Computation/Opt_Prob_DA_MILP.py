#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
def get_stoch_cvar_stairstep_prob_noRT(esr, gen, poi, opt_data, priceDA, priceRTadd, gens, probS,PRICE_STEPS):
    """
        Optimizes bidding strategies for energy storage and renewable generation in electricity markets.
        It calculates expected revenue from day-ahead and real-time market prices while minimizing costs and managing risk using Conditional Value at Risk (CVaR).
        The optimization model is MILP
        Parameters:
        - esr (dict): Energy storage resource parameters including:
          - 'max': Maximum energy capacity (MWh)
          - 'min': Minimum energy level (MWh)
          - 'init': Initial energy level (MWh)
          - 'power': Maximum rate of charge/discharge (MW)
          - 'effCrg': Charging efficiency (<=1, where 1 is 100% efficient)
          - 'effDis': Discharging efficiency (<=1)
          - 'opCost': Operating cost per MWh (e.g., for degradation)
        - gen (dict): Details of the generator including:
          - 'max': Maximum power capacity (MW)
          - 'opCost': Operating cost per MWh
        - poi (float): Power injection limit or threshold (MW).
        - opt_data (dict): Optimization data containing:
          - 'eta_plus': Price responsiveness factors for price increases
          - 'eta_minus': Price responsiveness factors for price decreases
          - 'dev_plus_addifzero': Additional costs for zero deviations (price increases)
          - 'dev_minus_addifzero': Additional costs for zero deviations (price decreases)
          - 'gridcharge': Constraints on grid charging
          - 'nscen_breaks': Number of scenario breakpoints
          - 'cvar_weight': Weight for CVaR calculation
          - 'cvar_confid': Confidence level for CVaR
        - priceDA (numpy.ndarray): Day-ahead market prices (per scenario and time slot).
        - priceRTadd (numpy.ndarray): Additional real-time price data (not directly used).
        - gens (numpy.ndarray): Generation capacity scenarios (per scenario).
        - probS (numpy.ndarray): Probability of each scenario occurring.
        - PRICE_STEPS (numpy.ndarray): Price steps or thresholds used in the bidding strategy.

        Operations:
        1. Initializes matrices for managing costs and sets up constraints and variables within a Pyomo model.
        2. Configures the optimization problem dynamically, considering energy storage dynamics, generation limits, and varying market prices across scenarios and time steps.
        3. Calculates deviation costs, configures price steps, and incorporates scenario probabilities to form the risk-adjusted objective function.
        4. Solves the model using the CPLEX solver, aiming to optimize profit while managing risk through CVaR.

        Returns:
        - m (pyomo.environ.ConcreteModel): The configured Pyomo model.
        - results (SolverResults): The results from solving the model, which include optimal operational strategies.
         """



    (nS,nT)=priceDA.shape
    #print(nS)
    #print(nT)
    esrEnergy_max = esr["max"]
    esrEnergy_min = esr["min"]
    esrEnergy_init = esr["init"]
    esrPwr  = esr["power"] #Maximum rate of charge/discharge (MW)
    effCrg = esr["effCrg"]#If purchase X MWh then X*effCrg are stored. <=1.
    effDis = esr["effDis"] #0.95; %To sell X MWh, must discharge X/effDis. <=1.
    esrOpCost = esr["opCost"] #$/MWh cost of operating storage, such as degradation cost. Applies', 'to MWh charged and discharged')
    genPower_max = gen["max"]
    genOpCost = gen["opCost"] #$/MWh cost of operating renewable generator (can be 0)
    #'Calcuate', 'imbalance (deviation) costs'
    #step_match=np.zeros((nS,nT))
    dev_cost_plus=np.zeros((nS,nT))
    dev_cost_minus=np.zeros((nS,nT))
    temp_pos=np.where(priceDA >= 0)
    temp_neg=np.where(priceDA < 0)
    eta_plus=opt_data["eta_plus"]
    eta_minus = opt_data["eta_minus"]
    dev_cost_plus[temp_pos]=eta_plus[temp_pos[1:2]]*priceDA[temp_pos]
    dev_cost_plus[temp_neg]=eta_minus[temp_neg[1:2]]*priceDA[temp_neg]
    dev_cost_minus[temp_pos]=eta_minus[temp_pos[1:2]]*priceDA[temp_pos]
    dev_cost_minus[temp_neg]=eta_plus[temp_neg[1:2]]*priceDA[temp_neg]
    pp= np.where(dev_cost_plus == 0)
    nps=np.where(dev_cost_minus==0)
    dev_cost_plus[pp]=opt_data["dev_plus_addifzero"]
    dev_cost_minus[nps]=opt_data["dev_minus_addifzero"]
    priceSteps=np.sort(PRICE_STEPS,axis=0)
    (nstep,nc)=priceSteps.shape #(5,48)
    #print(priceSteps)
    #print(nc)
    # create optimization model
    m =pyo.ConcreteModel("Compute Bid")
    #sets
    TPi = np.linspace(0, nT-1, num=nT, dtype=int)
    m.TP=pyo.Set(initialize=TPi)
    #m.TP.pprint()
    TSi=np.linspace(0, nS-1, num=nS, dtype=int)
    m.TS = pyo.Set(initialize=TSi)
    #m.TS.pprint()
    TBi=np.linspace(0, opt_data["nscen_breaks"]-1, num=opt_data["nscen_breaks"], dtype=int)
    m.Bp = pyo.Set(initialize=TBi)
    Tsp=np.linspace(0, nstep-1, num=nstep, dtype=int)
    m.Tstep=pyo.Set(initialize=Tsp)
    #m.Bp.pprint()
    """
    TPi = np.linspace(0, 5, num=5, dtype=int)
    m.TP=pyo.Set(initialize=TPi)
    m.TP.pprint()
    TSi=np.linspace(0, nS-1, num=nS, dtype=int)
    m.TS = pyo.Set(initialize=TSi)
    m.TS.pprint()
    TBi=np.linspace(1, 5, num=5, dtype=int)
    m.Bp = pyo.Set(initialize=TBi)
    m.Bp.pprint()
     """
    # variables
    m.theta=pyo.Var(domain=pyo.Reals)
    m.phi=pyo.Var(m.TS,domain=pyo.NonNegativeReals)
    #m.x=pyo.Var(m.TP,m.Bp,domain=pyo.Reals)
    #m.Ph_sc=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-esrPwr*gridcharge, poi))
    m.Ph_da=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-esrPwr*opt_data["gridcharge"], poi))
    #m.Ph_in=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    #m.Pg_da=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.Pg_sc=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(0, genPower_max))
    #m.Pg_in=pyo.Varm.TS,m.TP,domain=pyo.Reals)
    #m.Pesr_da=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.Pesr_sc=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-esrPwr, esrPwr))
    #m.Pesr_in=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.mode_ch=pyo.Var(m.TS,m.TP,domain=pyo.Binary) ## Binary
    m.mode_dis=pyo.Var(m.TS,m.TP,domain=pyo.Binary) ## Binary
    m.Esm_ch=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.Esm_dis=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.SOC=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(esrEnergy_min, esrEnergy_max))
    m.e_p=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.e_n=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.td=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.Pc=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.SOC_f=pyo.Var(m.TS,domain=pyo.Reals,bounds=(esrEnergy_min, esrEnergy_max))
    m.ph_da_steps=pyo.Var(m.Tstep,m.TP,domain=pyo.Reals)
    # Objective
    #revenue= sum(probS[s]*sum(dev_cost_plus[s,t]*m.e_p[s,t]+ priceDA[s,t]*m.Ph_da[s,t] for t in m.TP)for s in m.TS)
    revenue = sum(probS[s] * sum(dev_cost_plus[s, t] * m.e_p[s, t] + priceDA[s, t] * m.Ph_da[s, t] for t in m.TP) for s in m.TS)
    #revenue= sum(sum((dev_cost_plus[t,s]*m.e_p[t,s]+ priceDA[t,s]*m.Ph_da[t,s]+(priceDA[t,s]+1)*m.Ph_in[t,s])for t in m.TP)for s in m.TS)
    #ipdb.set_trace()
    cost=sum(probS[s]*sum(dev_cost_minus[s,t]*m.e_n[s,t]for t in m.TP)for s in m.TS)
    # cost = sum(probS[s] * sum(
    #     dev_cost_minus[s, t] * m.e_n[s, t] + genOpCost * m.Pg_sc[s, t] + esrOpCost * (m.Esm_ch[s, t] + m.Esm_dis[s, t])
    #     for t in m.TP) for s in m.TS)
    cvar= opt_data["cvar_weight"]*(m.theta-sum(probS[s]*m.phi[s] for s in m.TS)/(1-opt_data["cvar_confid"]))
    #a=sum(sum(dev_cost_plus[s,t]*m.e_p[t,s] for t in m.TP)for s in m.TS)
    #m.profit = pyo.Objective(a, sense=pyo.maximize)
    m.profit = pyo.Objective(expr = revenue - cost+cvar, sense=pyo.maximize)
    ## constraints
    m.cons = pyo.ConstraintList()
    for s in m.TS:
        m.cons.add(m.theta - m.phi[s] - probS[s] * sum(
            dev_cost_plus[s, t] * m.e_p[s, t] + priceDA[s, t] * m.Ph_da[s, t] for t in m.TP) + probS[s] * sum(
            dev_cost_minus[s, t] * m.e_n[s, t] + genOpCost * m.Pg_sc[s, t] + esrOpCost * (
                        m.Esm_ch[s, t] + m.Esm_dis[s, t]) for t in m.TP) <= 0)
        m.cons.add(m.SOC[s,0]==esrEnergy_init)
        m.cons.add(m.SOC_f[s] == m.SOC[s, nT - 1] + effCrg * m.Esm_ch[s, nT - 1] - (1/effDis)*m.Esm_dis[s, nT - 1])
        for t in m.TP:
            ### ESR
            m.cons.add(m.Pesr_sc[s,t]==-m.Esm_ch[s,t]+m.Esm_dis[s,t])
            m.cons.add(m.Esm_ch[s,t]<=m.mode_ch[s,t]*esrPwr)
            m.cons.add(m.Esm_dis[s,t]<=m.mode_dis[s,t]*esrPwr)
            # m.cons.add(m.Esm_ch[s, t] <=  esrPwr)
            # m.cons.add(m.Esm_dis[s, t] <=  esrPwr)
            m.cons.add(m.mode_ch[s,t]+m.mode_dis[s,t]<=1)

            m.cons.add(m.Ph_da[s,t]==m.Pg_sc[s,t]+m.Pesr_sc[s,t])
            m.cons.add(m.td[s,t]==gens[s,t]-m.Pc[s,t]-m.Pg_sc[s,t])
            m.cons.add(m.td[s,t]==m.e_p[s,t]-m.e_n[s,t])
            m.cons.add(m.Pc[s,t]<=gens[s,t])
            #SOC
            if t >0:
                m.cons.add(m.SOC[s, t] == m.SOC[s, t - 1] + effCrg * m.Esm_ch[s, t - 1] -(1/effDis)* m.Esm_dis[s, t - 1])
            for b in m.Tstep:
                if b < nstep-1:
                    m.cons.add(m.ph_da_steps[b, t] <= m.ph_da_steps[b + 1,t])
            temp2 = np.array(priceSteps[:, t]) * np.ones((1, nstep))
            steprep = temp2.T * np.ones((1, nS))
            idxmat2 = np.cumsum(steprep >= priceDA[:, t].T, axis=0)
            idxmat = np.where(idxmat2 == 1,idxmat2,0)
            step_match = np.nonzero(idxmat.T)[1]
            for a in range(step_match.size):
                m.cons.add(m.Ph_da[a, t] == m.ph_da_steps[step_match[a],t])
    solver = SolverFactory('cplex', solver_io='nl')
    results = solver.solve(m, tee=True)
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
       pass #print("Solution is optimal.")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("Model is infeasible. Solve the LP model")
    else:
        pass

    return m,results

    
