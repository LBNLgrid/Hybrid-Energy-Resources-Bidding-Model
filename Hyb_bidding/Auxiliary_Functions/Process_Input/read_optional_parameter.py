import numpy as np
import yaml
def process_optional_parameters(datapath):
    hh = datapath + "opt_param.yml"
    with open(hh) as stream:
        opt_data = yaml.safe_load(stream)
        if 'cvar_weight' not in opt_data:
            opt_data['cvar_weight'] = 1

        if 'gridcharge' not in opt_data:
            opt_data['gridcharge'] = 0

        if 'bid_type' not in opt_data:
            opt_data['bid_type'] = 2

        if 'nscen_breaks' not in opt_data:
            opt_data['nscen_breaks'] = 6

        if 'break_mode' not in opt_data:
            opt_data['break_mode'] = "mid"

        if 'price_floor' not in opt_data:
            opt_data['price_floor'] = -150

        if 'price_ceil' not in opt_data:
            opt_data['price_ceil'] = 1000

        if 'x_step_min' not in opt_data:
            opt_data['x_step_min'] = 0.1

        if 'q1min' not in opt_data:
            opt_data['q1min'] = 1e-10

        if 'cvar_confid' not in opt_data:
            opt_data['cvar_confid'] = 0.95

        if 'eta_plus' not in opt_data:
            opt_data['eta_plus'] = 0.5 * np.ones(48)

        if 'eta_minus' not in opt_data:
            opt_data['eta_minus'] = 1.5 * np.ones(48)

        if 'dev_plus_addifzero' not in opt_data:
            opt_data['dev_plus_addifzero'] = -0.5

        if 'dev_minus_addifzero' not in opt_data:
            opt_data['dev_minus_addifzero'] = 0.5

        if 'p_thresh_heur_pcentile' not in opt_data:
            opt_data['p_thresh_heur_pcentile'] = [0.2, 0.8]

        if 'pHoriz_scale' not in opt_data:
            opt_data['pHoriz_scale'] = 2

        if 'price_heuristic' not in opt_data:
            opt_data['price_heuristic'] = "yes"

        if 'formulation' not in opt_data:
            opt_data['formulation'] = "LP"  # (yes or no)

        if 'rt_vs_da' not in opt_data:
            opt_data['rt_vs_da'] = 0  # 0 day ahead 1 real time

        if 'enforce_da_rt' not in opt_data:
            opt_data['enforce_da_rt'] = 1

        return opt_data

    return opt_data
