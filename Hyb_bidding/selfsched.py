#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def get_stoch_cvar_selfsched_prob(esr_prop,gen_prop,poi,rt_vs_da,enforce_da_rt,eta_plus,eta_minus,plus_adder,minus_adder,priceDA,priceRTadd,gen,probS,cvar_weight,confid_lev,gridcharge,prSteps,np,pyo,Time_p,nscen_breaks):
    Lambda = rt_vs_da
    (nS,nT)=priceDA.shape
    esrEnergy_max = esr_prop.max
    esrEnergy_min = esr_prop.min
    esrEnergy_init = esr_prop.init
    esrPwr  = esr_prop.power #Maximum rate of charge/discharge (MW)
    effCrg = esr_prop.effCrg#If purchase X MWh then X*effCrg are stored. <=1.
    effDis = esr_prop.effDis #0.95; %To sell X MWh, must discharge X/effDis. <=1.
    esrOpCost = esr_prop.opCost #$/MWh cost of operating storage, such as degradation cost. Applies', 'to MWh charged and discharged')
    genPower_max = gen_prop.max
    genOpCost = gen_prop.opCost #$/MWh cost of operating renewable generator (can be 0)
    dev_cost_plus=np.zeros((nS,nT))
    dev_cost_minus=np.zeros((nS,nT))
    temp_pos_p=np.where(np.minimum(priceDA,priceDA+priceRTadd) >= 0)
    temp_pos_n = np.where(np.minimum(priceDA, priceDA + priceRTadd) <0)
    temp_neg_p = np.where(np.maximum(priceDA, priceDA + priceRTadd) <= 0)
    temp_neg_n = np.where(np.maximum(priceDA, priceDA + priceRTadd) > 0)
    dev_cost_plus[temp_pos_p]=eta_plus[temp_pos_p[1:2]]*np.minimum(priceDA[temp_pos_p],priceDA[temp_pos_p]+priceRTadd[temp_pos_p])
    dev_cost_plus[temp_pos_n]=eta_minus[temp_pos_n[1:2]]*np.minimum(priceDA[temp_pos_n],priceDA[temp_pos_n]+priceRTadd[temp_pos_n])
    dev_cost_minus[temp_neg_p]=eta_plus[temp_neg_p[1:2]]*np.maximum(priceDA[temp_neg_p],priceDA[temp_neg_p]+priceRTadd[temp_neg_p])
    dev_cost_minus[temp_neg_n]=eta_minus[temp_neg_n[1:2]]*np.maximum(priceDA[temp_neg_n],priceDA[temp_neg_n]+priceRTadd[temp_neg_n])
    pp= np.where(dev_cost_plus == 0)[0]
    nps=np.where(dev_cost_minus==0)[0]
    dev_cost_plus[pp]=plus_adder
    dev_cost_minus[nps]=minus_adder
    priceSteps=prSteps#np.sort(prSteps)
    (nstep,nc)=priceSteps.shape
    m = pyo.ConcreteModel("Compute Bid")
    # sets
    TPi = np.linspace(0, nT - 1, num=nT, dtype=int)
    m.TP = pyo.Set(initialize=TPi)
    # m.TP.pprint()
    TSi = np.linspace(0, nS - 1, num=nS, dtype=int)
    m.TS = pyo.Set(initialize=TSi)
    Tsp = np.linspace(0, nstep - 1, num=nstep, dtype=int)
    m.Tsp = pyo.Set(initialize=Tsp)
    # m.Bp.pprint()
    # variables
    m.theta=pyo.Var(domain=pyo.Reals)
    m.phi=pyo.Var(m.TS,domain=pyo.NonNegativeReals)
    m.Ph_sc=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-esrPwr*gridcharge, poi))
    m.Ph_da=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-esrPwr*gridcharge, poi))
    m.Ph_rt=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.ph_da_steps = pyo.Var( m.Tsp, m.TP, domain=pyo.Reals)
    m.Pg_da=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.Pg_sc=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(0, genPower_max))
    m.Pg_rt=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.Pesr_da=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.Pesr_sc=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-esrPwr, esrPwr))
    m.Pesr_rt=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-Lambda*esrPwr, Lambda*esrPwr))
    m.Esm_ch = pyo.Var(m.TS,m.TP, domain=pyo.NonNegativeReals)
    m.Esm_dis = pyo.Var(m.TS,m.TP, domain=pyo.NonNegativeReals)
    m.mode_ch=pyo.Var(m.TS,m.TP,domain=pyo.Binary) ## Binary
    m.mode_dis=pyo.Var(m.TS,m.TP,domain=pyo.Binary) ## Binary
    m.SOC=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(esrEnergy_min, esrEnergy_max))
    m.e_p=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.e_n=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.td=pyo.Var(m.TS,m.TP,domain=pyo.Reals)
    m.Pc=pyo.Var(m.TS,m.TP,domain=pyo.NonNegativeReals)
    m.SOC_f=pyo.Var(m.TS,domain=pyo.Reals,bounds=(esrEnergy_min, esrEnergy_max))
    # Objective
    revenue = sum(probS[s] * sum(dev_cost_plus[s, t] * m.e_p[s, t] + priceDA[s, t] * m.Ph_da[s, t] + (priceDA[s, t] + priceRTadd[s, t])*m.Ph_rt[s,t]  for t in m.TP) for s in m.TS)
    # ipdb.set_trace()
    cost = sum(probS[s] * sum( dev_cost_minus[s, t] * m.e_n[s, t] + genOpCost * m.Pg_sc[s, t] + esrOpCost * (m.Esm_ch[s, t] + m.Esm_dis[s, t])for t in m.TP) for s in m.TS)
    cvar = cvar_weight * (m.theta - sum(probS[s] * m.phi[s] for s in m.TS) / (1 - confid_lev))
    m.profit = pyo.Objective(expr= revenue - cost + cvar, sense=pyo.maximize)
    ## constraints
    m.cons = pyo.ConstraintList()
    #m.cons.add(np.ones((nS,nT))*m.Pg_sc== np.ones((nS,nT))*m.Pg_da + np.ones((nS,nT))*m.Pg_rt)
    for s in m.TS:
        m.cons.add(m.theta - m.phi[s] - probS[s] * sum(dev_cost_plus[s, t] * m.e_p[s, t] + priceDA[s, t] * m.Ph_da[s, t] + (priceDA[s, t] + priceRTadd[s, t])* m.Ph_rt[s, t] for t in m.TP) + probS[s] * sum(dev_cost_minus[s, t] * m.e_n[s, t] + genOpCost * m.Pg_sc[s, t] + esrOpCost * (m.Esm_ch[s, t] + m.Esm_dis[s, t]) for t in m.TP) <= 0)
        m.cons.add(m.SOC[s, 0] == esrEnergy_init)
        m.cons.add(m.SOC_f[s] == m.SOC[s, nT-1] + effCrg * m.Esm_ch[s, nT-1] - m.Esm_dis[s, nT-1] / effDis)
        step_match=np.zeros(nS)
        for t in m.TP:
            m.cons.add(m.Pesr_sc[s, t] == m.Pesr_da[s, t] + m.Pesr_rt[s, t])
            m.cons.add(m.Pesr_sc[s, t] == -m.Esm_ch[s,t]+ m.Esm_dis[s,t])
            m.cons.add(m.Esm_ch[s, t] <= m.mode_ch[s,t]*esrPwr)
            m.cons.add(m.Esm_dis[s, t] <= m.mode_dis[s,t]*esrPwr)
            m.cons.add(m.mode_ch[s, t]+m.mode_dis[s,t] <= 1)
            m.cons.add(m.Pg_sc[s, t] == m.Pg_da[s, t] + m.Pg_rt[s, t])
            m.cons.add(m.Ph_sc[s, t] == m.Ph_da[s, t] + m.Ph_rt[s, t])
            m.cons.add(m.Ph_sc[s, t] == m.Pg_sc[s, t] + m.Pesr_sc[s, t])
            m.cons.add(m.Ph_da[s, t] == m.Pg_da[s, t] + m.Pesr_da[s, t])
            m.cons.add(m.Ph_rt[s, t] == m.Pg_rt[s, t] + m.Pesr_rt[s, t])
            m.cons.add(m.td[s, t] == gen[s, t] - m.Pc[s, t] - m.Pg_sc[s, t])
            m.cons.add(m.td[s, t] == m.e_p[s, t] - m.e_n[s, t])
            m.cons.add(m.Pc[s, t] <= gen[s, t])
            ## SOC
            if t >0:
                m.cons.add(m.SOC[s, t] == m.SOC[s, t - 1] + effCrg * m.Esm_ch[s, t - 1] - m.Esm_dis[s, t - 1]/effDis )
            for b in m.Tsp:
                if b < nstep-1:
                    m.cons.add(m.ph_da_steps[b, t] <= m.ph_da_steps[b + 1,t])
            temp2 = np.array(priceSteps[:, t]) * np.ones((1, nstep))
            steprep = temp2.T * np.ones((1, nS))
            idxmat2 = np.cumsum(steprep >= priceDA[:, t].T, axis=0)
            idxmat = np.where(idxmat2 == 1,idxmat2,0)
            step_match = np.nonzero(idxmat.T)[1]
            m.cons.add(-Lambda * m.Pg_da[s, t] <= m.Pg_rt[s, t])
            m.cons.add(m.Pg_rt[s, t] <= Lambda * m.Pg_da[s, t])
            # print(priceSteps)
            # print(steprep)
            # print(steprep>=priceDA[:,t])
            # print(idxmat)
            # step_match,=np.where(idxmat>0,idxmat)[0]
            #print(step_match)
            for a in m.TS:
                m.cons.add(m.Ph_da[a, t] == m.ph_da_steps[step_match[a],t])
    solver = pyo.SolverFactory('cplex', solver_io='nl')
    # solver.options['solutionpool.aggregate'] = 'yes'
    # solver.options['solutionpool.capacity'] = 5
    results = solver.solve(m, tee=True).write()
    #print(pyo.value(m.theta))
    return m, results

    # print('Number of solutions found: ' + str(nSolutions))


