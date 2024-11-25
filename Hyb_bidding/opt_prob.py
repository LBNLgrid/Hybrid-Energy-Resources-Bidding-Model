#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def get_stoch_cvar_stairstep_prob_noRT(esr_prop,gen_prop,poi,eta_plus,eta_minus,plus_adder,minus_adder,priceDA,gen,probS,cvar_weight,confid_lev,gridcharge,prSteps,np,pyo,Time_p,nscen_breaks):
    (nS,nT)=priceDA.shape
    #print(nS)
    #print(nT)
    esrEnergy_max = esr_prop.max
    esrEnergy_min = esr_prop.min
    esrEnergy_init = esr_prop.init
    esrPwr  = esr_prop.power #Maximum rate of charge/discharge (MW)
    effCrg = esr_prop.effCrg#If purchase X MWh then X*effCrg are stored. <=1.
    effDis = esr_prop.effDis #0.95; %To sell X MWh, must discharge X/effDis. <=1.
    esrOpCost = esr_prop.opCost #$/MWh cost of operating storage, such as degradation cost. Applies', 'to MWh charged and discharged')
    genPower_max = gen_prop.max
    genOpCost = gen_prop.opCost #$/MWh cost of operating renewable generator (can be 0)
    #'Calcuate', 'imbalance (deviation) costs'
    #step_match=np.zeros((nS,nT))
    dev_cost_plus=np.zeros((nS,nT))
    dev_cost_minus=np.zeros((nS,nT))
    temp_pos=np.where(priceDA >= 0)
    temp_neg=np.where(priceDA < 0)
    dev_cost_plus[temp_pos]=eta_plus[temp_pos[1:2]]*priceDA[temp_pos]
    dev_cost_plus[temp_neg]=eta_minus[temp_neg[1:2]]*priceDA[temp_neg]
    dev_cost_minus[temp_pos]=eta_minus[temp_pos[1:2]]*priceDA[temp_pos]
    dev_cost_minus[temp_neg]=eta_plus[temp_neg[1:2]]*priceDA[temp_neg]
    pp= np.where(dev_cost_plus == 0)
    nps=np.where(dev_cost_minus==0)
    dev_cost_plus[pp]=plus_adder
    dev_cost_minus[nps]=minus_adder
    priceSteps=prSteps#np.sort(prSteps)
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
    TBi=np.linspace(0, nscen_breaks-1, num=nscen_breaks, dtype=int)
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
    m.Ph_da=pyo.Var(m.TS,m.TP,domain=pyo.Reals,bounds=(-esrPwr*gridcharge, poi))
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
    m.ph_da_steps=pyo.Var(m.TP,m.Tstep,domain=pyo.Reals)
    # Objective
    revenue= sum(probS[s]*sum(dev_cost_plus[s,t]*m.e_p[s,t]+ priceDA[s,t]*m.Ph_da[s,t]for t in m.TP)for s in m.TS)
    #revenue= sum(sum((dev_cost_plus[t,s]*m.e_p[t,s]+ priceDA[t,s]*m.Ph_da[t,s]+(priceDA[t,s]+1)*m.Ph_in[t,s])for t in m.TP)for s in m.TS)
    #ipdb.set_trace()
    cost=sum(probS[s]*sum(dev_cost_minus[s,t]*m.e_n[s,t]+ genOpCost*m.Pg_sc[s,t]+esrOpCost*(m.Esm_ch[s,t]+m.Esm_dis[s,t])for t in m.TP)for s in m.TS)
    cvar= cvar_weight*(m.theta-sum(probS[s]*m.phi[s] for s in m.TS)/(1-confid_lev))
    #a=sum(sum(dev_cost_plus[s,t]*m.e_p[t,s] for t in m.TP)for s in m.TS)
    #m.profit = pyo.Objective(a, sense=pyo.maximize)
    m.profit = pyo.Objective(expr = revenue - cost+cvar, sense=pyo.maximize)
    ## constraints
    m.cons = pyo.ConstraintList()
    for s in m.TS:
        m.cons.add(m.theta-m.phi[s]-probS[s]*sum(dev_cost_plus[s,t]*m.e_p[s,t]+ priceDA[s,t]*m.Ph_da[s,t]for t in m.TP)+probS[s]*sum(dev_cost_minus[s,t]*m.e_n[s,t]+ genOpCost*m.Pg_sc[s,t]+esrOpCost*(m.Esm_ch[s,t]+m.Esm_dis[s,t])for t in m.TP)<=0)
        m.cons.add(m.SOC[s,0]==esrEnergy_init)
        for t in m.TP:
            ### ESR
            m.cons.add(m.Pesr_sc[s,t]==-m.Esm_ch[s,t]+m.Esm_dis[s,t])
            m.cons.add(m.Esm_ch[s,t]<=m.mode_ch[s,t]*esrPwr)
            m.cons.add(m.Esm_dis[s,t]<=m.mode_dis[s,t]*esrPwr)
            m.cons.add(m.mode_ch[s,t]+m.mode_dis[s,t]<=1)
            ## SOC
            if t >=1 :
                m.cons.add(m.SOC[s,t]==m.SOC[s,t-1]+effCrg*m.Esm_ch[s,t-1]-m.Esm_dis[s,t-1]/effDis)  
            m.cons.add(m.SOC_f[s]==m.SOC[s,nT-1]+effCrg*m.Esm_ch[s,nT-1]-m.Esm_dis[s,nT-1]/effDis)
            m.cons.add(m.Ph_da[s,t]==m.Pg_sc[s,t]+m.Pesr_sc[s,t])
            m.cons.add(m.td[s,t]==gen[s,t]-m.Pc[s,t]-m.Pg_sc[s,t])
            m.cons.add(m.td[s,t]==m.e_p[s,t]-m.e_n[s,t])
            m.cons.add(m.Pc[s,t]<=gen[s,t])
            for b in m.Tstep :
                if b < nstep-1 :
                    m.cons.add(m.ph_da_steps[t,b]<=m.ph_da_steps[t,b+1])
            temp=priceSteps[:,t]
            temp2=np.array(temp)
            temp3=temp2*np.ones((1,nstep))
            steprep=temp3.T*np.ones((1,nS))
            #print(priceSteps)
            #print(steprep)
            #print(steprep>=priceDA[:,t])
            idxmat2=np.cumsum(steprep>=priceDA[:,t].T,axis=0)
            idxmat=np.where(idxmat2==1)[0]
            #print(idxmat)
            step_match=idxmat[idxmat>0][0]
            #step_match,=np.where(idxmat>0,idxmat)[0]
            #print(step_match)
            #
            m.cons.add(m.Ph_da[s,t]==m.ph_da_steps[t,step_match])
            """
           
           
            
            
            m.cons.add(m.td[t,s]==m.e_p[t,s]+m.e_n[t,s])
            
            #m.cons.add(-0.9*m.Pg_da[t,s]<=m.Pg_da[t,s])
            #m.cons.add(m.Pg_da[t,s]<=0.9*m.Pg_da[t,s])
            #m.cons.add(-0.9*m.Pg_da[t,s]<=m.Pg_da[t,s])
            #m.cons.add(m.Pg_da[t,s]<=0.9*m.Pg_da[t,s])
            
            """
    #opt = pyo.SolverFactory('cplex')
    #m.vol = pyo.Constraint(expr = m.Ph_sc[t,s]==m.Ph_da[t,s]+m.Ph_in[t,s] for t in m.TP for s in m.TS)
    #params.set_attribute(‘RelativeOptimalityTolerance’, 0.05)
    #opt.options['mipgap'] = 1e-9
    solver = pyo.SolverFactory('cplex', solver_io='nl')
    #solver.options['solutionpool.aggregate'] = 'yes'
    #solver.options['solutionpool.capacity'] = 5
    results=solver.solve(m, tee= True).write()
    
    print(pyo.value(m.theta))
    #print(pyo.value(m.phi))
    #bid.write()
    #instance.solutions.load_from(bid)
    #solution.get_values()
    #print(10*m.profit())
    #nSolutions = opt.get_model_attr("SolCount")
    #for v in m.component_objects(Var, active=True):
        #print("Variable", v)
        #for index in v:
        #    print(" ", index, model.x[index].value)
#     if results.solver.termination_condition == TerminationCondition.optimal:
#         print("Optimal solution found!")
#         print("Objective value:", value(m.obj))
#         solution_pool = results.solver.solution.pool
#         print("Number of solutions in the pool:", len(solution_pool))
#         for i, solution in enumerate(solution_pool):
#             print(f"Solution {i + 1}:")
#             # for var in model.component_data_objects(Var, active=True):
#             print(f"  {var.name}: {value(solution.variable[var])}")
#         else:
#              print("Solver did not find an optimal solution.")
    return m,results

    #print('Number of solutions found: ' + str(nSolutions))
  
"""
    # Print objective values of solutions
    for e in range(nSolutions):
        opt.set_gurobi_param("SolutionNumber", e)
        print('%g ' % opt.get_model_attr("PoolObjVal"), end='')
        print('')

  """
    