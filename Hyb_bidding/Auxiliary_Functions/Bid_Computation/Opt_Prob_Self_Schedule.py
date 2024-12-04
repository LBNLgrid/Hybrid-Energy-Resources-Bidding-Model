
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
def get_stoch_cvar_selfsched_prob(esr, gen, poi, opt_data, priceDA, priceRTadd, gens, probS):
        """Optimizes bidding strategies for energy storage and renewable generation in electricity markets.
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
        - The bid curves are independent of the prices
        - m (pyomo.environ.ConcreteModel): The configured Pyomo model.
        - results (SolverResults): The results from solving the model, which include optimal operational strategies.
        """
        Lambda = opt_data["rt_vs_da"]
        (nS, nT) = priceDA.shape
        esrEnergy_max = esr["max"]
        esrEnergy_min = esr["min"]
        esrEnergy_init = esr["init"]
        esrPwr = esr["power"]  # Maximum rate of charge/discharge (MW)
        effCrg = esr["effCrg"]  # If purchase X MWh then X*effCrg are stored. <=1.
        effDis = esr["effDis"]  # 0.95; %To sell X MWh, must discharge X/effDis. <=1.
        esrOpCost = esr["opCost"]  # $/MWh cost of operating storage, such as degradation cost. Applies', 'to MWh charged and discharged')
        genPower_max = gen["max"]
        genOpCost = gen["opCost"]  # $/MWh cost of operating renewable generator (can be 0)
        dev_cost_plus = np.zeros((nS, nT))
        dev_cost_minus = np.zeros((nS, nT))
        temp_pos_p = np.where(np.minimum(priceDA, priceDA + priceRTadd) >= 0)
        temp_pos_n = np.where(np.minimum(priceDA, priceDA + priceRTadd) < 0)
        temp_neg_p = np.where(np.maximum(priceDA, priceDA + priceRTadd) <= 0)
        temp_neg_n = np.where(np.maximum(priceDA, priceDA + priceRTadd) > 0)
        eta_plus = opt_data["eta_plus"]
        eta_minus = opt_data["eta_minus"]
        dev_cost_plus[temp_pos_p] = eta_plus[temp_pos_p[1:2]] * np.minimum(priceDA[temp_pos_p],
                                                                           priceDA[temp_pos_p] + priceRTadd[temp_pos_p])
        dev_cost_plus[temp_pos_n] = eta_minus[temp_pos_n[1:2]] * np.minimum(priceDA[temp_pos_n],
                                                                            priceDA[temp_pos_n] + priceRTadd[
                                                                                temp_pos_n])
        dev_cost_minus[temp_neg_p] = eta_plus[temp_neg_p[1:2]] * np.maximum(priceDA[temp_neg_p],
                                                                            priceDA[temp_neg_p] + priceRTadd[
                                                                                temp_neg_p])
        dev_cost_minus[temp_neg_n] = eta_minus[temp_neg_n[1:2]] * np.maximum(priceDA[temp_neg_n],
                                                                             priceDA[temp_neg_n] + priceRTadd[
                                                                                 temp_neg_n])
        pp = np.where(dev_cost_plus == 0)
        nps = np.where(dev_cost_minus == 0)
        dev_cost_plus[pp] = opt_data["dev_plus_addifzero"]
        dev_cost_minus[nps] = opt_data["dev_minus_addifzero"]

        #### Variable Definition

        m = pyo.ConcreteModel("Compute Bid")
        # sets
        TPi = np.linspace(0, nT - 1, num=nT, dtype=int)
        m.TP = pyo.Set(initialize=TPi)
        # m.TP.pprint()
        TSi = np.linspace(0, nS - 1, num=nS, dtype=int)
        m.TS = pyo.Set(initialize=TSi)
        # m.Bp.pprint()

        # variables
        m.theta = pyo.Var(domain=pyo.Reals)
        m.phi = pyo.Var(m.TS, domain=pyo.NonNegativeReals)
        m.Ph_sc = pyo.Var(m.TS, m.TP, domain=pyo.Reals, bounds=(-esrPwr * opt_data["gridcharge"], poi))
        m.Ph_da = pyo.Var(m.TS, m.TP, domain=pyo.Reals, bounds=(-esrPwr * opt_data["gridcharge"], poi))
        m.Ph_rt = pyo.Var(m.TS, m.TP, domain=pyo.Reals)
        m.ph_da_s = pyo.Var(m.TP, domain=pyo.Reals)
        m.Pg_da = pyo.Var(m.TS, m.TP, domain=pyo.Reals)
        m.Pg_sc = pyo.Var(m.TS, m.TP, domain=pyo.Reals, bounds=(0, genPower_max))
        m.Pg_rt = pyo.Var(m.TS, m.TP, domain=pyo.Reals)
        m.Pesr_da = pyo.Var(m.TS, m.TP, domain=pyo.Reals)
        m.Pesr_sc = pyo.Var(m.TS, m.TP, domain=pyo.Reals, bounds=(-esrPwr, esrPwr))
        m.Pesr_rt = pyo.Var(m.TS, m.TP, domain=pyo.Reals)
        m.Esm_ch = pyo.Var(m.TS, m.TP, domain=pyo.NonNegativeReals)
        m.Esm_dis = pyo.Var(m.TS, m.TP, domain=pyo.NonNegativeReals)
        m.mode_ch = pyo.Var(m.TS, m.TP, domain=pyo.Binary)  ## Binary
        m.mode_dis = pyo.Var(m.TS, m.TP, domain=pyo.Binary)  ## Binary
        m.SOC = pyo.Var(m.TS, m.TP, domain=pyo.Reals, bounds=(esrEnergy_min, esrEnergy_max))
        m.e_p = pyo.Var(m.TS, m.TP, domain=pyo.NonNegativeReals)
        m.e_n = pyo.Var(m.TS, m.TP, domain=pyo.NonNegativeReals)
        m.td = pyo.Var(m.TS, m.TP, domain=pyo.Reals)
        m.Pc = pyo.Var(m.TS, m.TP, domain=pyo.NonNegativeReals)
        m.SOC_f = pyo.Var(m.TS, domain=pyo.Reals, bounds=(esrEnergy_min, esrEnergy_max))
        # Objective
        revenue = sum(probS[s] * sum(
            dev_cost_plus[s, t] * m.e_p[s, t] + priceDA[s, t] * m.Ph_da[s, t] + (priceDA[s, t] + priceRTadd[s, t]) *
            m.Ph_rt[s, t] for t in m.TP) for s in m.TS)
        # ipdb.set_trace()
        cost = sum(probS[s] * sum(dev_cost_minus[s, t] * m.e_n[s, t] + genOpCost * m.Pg_sc[s, t] + esrOpCost * (
                    m.Esm_ch[s, t] + m.Esm_dis[s, t]) for t in m.TP) for s in m.TS)
        cvar = opt_data["cvar_weight"] * (
                    m.theta - sum(probS[s] * m.phi[s] for s in m.TS) / (1 - opt_data["cvar_confid"]))
        m.profit = pyo.Objective(expr=revenue - cost + cvar, sense=pyo.maximize)
        ## constraints
        m.cons = pyo.ConstraintList()
        # m.cons.add(np.ones((nS,nT))*m.Pg_sc== np.ones((nS,nT))*m.Pg_da + np.ones((nS,nT))*m.Pg_rt)
        for s in m.TS:
            m.cons.add(m.theta - m.phi[s] - probS[s] * sum(
                dev_cost_plus[s, t] * m.e_p[s, t] + priceDA[s, t] * m.Ph_da[s, t] + (priceDA[s, t] + priceRTadd[s, t]) *
                m.Ph_rt[s, t] for t in m.TP) + probS[s] * sum(
                dev_cost_minus[s, t] * m.e_n[s, t] + genOpCost * m.Pg_sc[s, t] + esrOpCost * (
                            m.Esm_ch[s, t] + m.Esm_dis[s, t]) for t in m.TP) <= 0)
            m.cons.add(m.SOC[s, 0] == esrEnergy_init)
            m.cons.add(m.SOC_f[s] == m.SOC[s, nT - 1] + effCrg * m.Esm_ch[s, nT - 1] - m.Esm_dis[s, nT - 1] / effDis)
            step_match = np.zeros(nS)
            for t in m.TP:
                m.cons.add(m.Pesr_sc[s, t] == m.Pesr_da[s, t] + m.Pesr_rt[s, t])
                m.cons.add(m.Pesr_sc[s, t] == -m.Esm_ch[s, t] + m.Esm_dis[s, t])
                m.cons.add(m.Esm_ch[s, t] <= m.mode_ch[s, t] * esrPwr)
                m.cons.add(m.Esm_dis[s, t] <= m.mode_dis[s, t] * esrPwr)
                m.cons.add(m.mode_ch[s, t] + m.mode_dis[s, t] <= 1)
                m.cons.add(m.Pg_sc[s, t] == m.Pg_da[s, t] + m.Pg_rt[s, t])
                m.cons.add(m.Ph_sc[s, t] == m.Ph_da[s, t] + m.Ph_rt[s, t])
                m.cons.add(m.Ph_sc[s, t] == m.Pg_sc[s, t] + m.Pesr_sc[s, t])
                m.cons.add(m.Ph_da[s, t] == m.Pg_da[s, t] + m.Pesr_da[s, t])
                m.cons.add(m.Ph_rt[s, t] == m.Pg_rt[s, t] + m.Pesr_rt[s, t])
                m.cons.add(m.td[s, t] == gens[s, t] - m.Pc[s, t] - m.Pg_sc[s, t])
                m.cons.add(m.td[s, t] == m.e_p[s, t] - m.e_n[s, t])
                m.cons.add(m.Pc[s, t] <= gens[s, t])
                ## SOC
                if t > 0:
                    m.cons.add(
                        m.SOC[s, t] == m.SOC[s, t - 1] + effCrg * m.Esm_ch[s, t - 1] - m.Esm_dis[s, t - 1] / effDis)
                m.cons.add(m.Ph_da[s, t] == m.ph_da_s[t])
                if (opt_data['enforce_da_rt'] ==1):
                    m.cons.add(-Lambda * m.Pg_da[s, t] <= m.Pg_rt[s, t])
                    m.cons.add(m.Pg_rt[s, t] <= Lambda * m.Pg_da[s, t])
                    m.cons.add(m.Pesr_rt[s, t] <= Lambda * esrPwr)
                    m.cons.add(m.Pesr_rt[s, t] >= -Lambda * esrPwr)

        filename = 'model_MaxStorCost.lp'
        m.write(filename, io_options={'symbolic_solver_labels': True})

        solver = pyo.SolverFactory('cplex', solver_io='nl')

        results = solver.solve(m, tee=True).write()

        return m, results



