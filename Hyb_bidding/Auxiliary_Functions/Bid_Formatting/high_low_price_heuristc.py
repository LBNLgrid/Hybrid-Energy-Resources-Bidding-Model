
import numpy as np
def get_high_low_price_heuristic(pqt_bids_base,opt_sol,pDA,gDA,scen_prob,hrs_to_mod,opt_data,esr,poi,p_thresh_heur):
    # Objective is to modify/extend bid curves to account for prices outside of
    # the price scenarios used in the bidding optimization. The heuristic is to
    # bid the maximum realistic amount if the price is exceptionally high or the
    # minimum amount if the price is exceptionally low.
    # INPUTS:
    #   pqt_bids_base: List of DataFrames (ndates x 1). Bid curves based on scen and scen_prob.
    #   opt_sols: List of DataFrames (ndates x 1). Results of bidding optimization. Used for soc and curtail variables
    #   scen, scen_prob: Both Lists of DataFrames (ndates x 1). Inputs to bidding optimization.
    #   hrs_to_mod: Array listing the hours to apply these heuristics to. Typically 0,1,2,...,23
    #   pLow(High)_thresh: Reference point for the maximum (minimum) price that
    #                      counts as a low (high) price across all hours
    #   pHoriz_scale: Value used to adjust the min/max price found in scenarios in
    #       each day's time horizon, as follows:
    #           pHigh = max([pHigh_thresh, pHoriz_scale * pHigh_horizon, pHigh_horizon / pHoriz_scale]);
    #           pLow = min([pLow_thresh, pHoriz_scale * pLow_horizon, pLow_horizon / pHoriz_scale]);
    #   price_ceil: price ceiling for the market
    #   gridcharging: 0 = No grid charging allowed, 1 = Grid charging unrestricted
    # OUTPUTS:
    #   pqt_format_bids: Each (a,b,c) row contains one point (price, quantity)=(a,b) on the bid curve for hour c.
    pLow_thresh=p_thresh_heur[0]
    pHigh_thresh=p_thresh_heur[1]
    pHoriz_scale=opt_data["pHoriz_scale"]
    price_ceil=opt_data["price_ceil"]
    gridcharging=opt_data["gridcharge"]
    class CustomError(Exception):
        pass
    #def thresh_fun(v1, v2):
    if pLow_thresh > pHigh_thresh:
            raise CustomError("pLow_thresh must be <= pHigh_thresh")

    pqt = pqt_bids_base
    pqt_new = pqt.copy()
    if gridcharging == 0 and np.min(pqt[:, 1] < 0):
        raise ValueError("Original bids contain grid charging and should not")
    #### low price heuristic
    # Assuming pqt, esr, scen[i], opt_sol[i], and scen_prob[i] are defined appropriately
    # pqt_new should be initialized before the loop if not already defined
    for hr in hrs_to_mod:
        q0 = np.min(pqt[pqt[:, 2] == hr, 1])
        p0 = np.min(pqt[pqt[:, 2] == hr, 0])

        if q0 > -esr['power']:
            pmin = np.min(pDA[:, hr + 1])
            pLow_horizon = np.min(pDA)
            pLow = min(pLow_thresh, pHoriz_scale * pLow_horizon, pLow_horizon / pHoriz_scale)

            if q0 >= 0 and pmin >= 0:
                if pLow >= 0:
                    soc = opt_sol.soc
                    curtail_avg = np.dot(scen_prob.T, np.maximum(0, gDA[:, hr] - np.minimum(esr['power'],esr['max'] - soc[ :, hr])))

                    if curtail_avg > 0 and curtail_avg < q0:
                        pqt_new = np.vstack([pqt_new, np.asarray([pLow, curtail_avg[0], hr],dtype=float)])
                        pqt_new = np.vstack([pqt_new, [0, -esr['power'] * gridcharging, hr]])
                    elif curtail_avg > 0 and curtail_avg >= q0:
                        pqt_new = np.vstack([pqt_new, [0, -esr['power'] * gridcharging, hr]])
                    else:
                        q_from_grid = np.dot(scen_prob.T, np.minimum(0, gDA[:, hr + 1] - esr['power']))
                        pqt_new = np.vstack((pqt_new, [pLow, q_from_grid[0] * gridcharging, hr]))
                        pqt_new = np.vstack((pqt_new, [0, -esr['power'] * gridcharging, hr]))
                else:
                    idx2 = (pDA[:, hr] <= p0)
                    qEsr_avg = np.dot(scen_prob[idx2], opt_sol.Pesr_da[idx2, hr, 0])
                    pqt_new = np.vstack((pqt_new, [0, max(0, min(q0, qEsr_avg[0])), hr]))
                    pqt_new = np.vstack((pqt_new, [pLow, -esr['power'] * gridcharging, hr]))
            else:
                pqt_new = np.vstack([pqt_new, [0, -esr['power'] * gridcharging, hr]])

    # pqt_new will contain the updated list of tuples or arrays to process further or store

    #     High  price heuristic
    pqt_bids_full = []
    # for i in range(len(pqt_bids_base)):
    #     pqt = pqt_bids_base[i]
    #     pqt_new = pqt.copy()
    # Assuming pqt, esr, pDA, gDA, poi, pHigh_thresh, pHoriz_scale, price_ceil, opt_sol, scen_prob are defined

    for hr in hrs_to_mod:
        qtop = np.max(pqt[pqt[:, 2] == hr, 1])
        qmax_byscen = np.minimum(poi, esr["power"] + gDA[:, hr])
        qmax_val = np.mean(qmax_byscen)

        if qtop < qmax_val:
            ptop = np.max(pqt[pqt[:, 2] == hr, 0])
            pmax = np.max(pDA[:, hr])
            pHigh_horizon = np.max(pDA[:, hr:])
            pHigh = max(pHigh_thresh, pHoriz_scale * pHigh_horizon, pHigh_horizon / pHoriz_scale)

            idx = np.where((pqt_new[:, 0] == ptop) & (pqt_new[:, 1] == qtop) & (pqt_new[:, 2] == hr))[0]

            if pmax >= 0:
                pqt_new[idx, 0] = pHigh
            else:
                if pHigh > 0:
                    curtail_data = opt_sol.curtail

                    if len(np.unique(pqt[pqt[:, 2] == hr, 0])) > 1:
                        p_2most = np.max(np.setdiff1d(pqt[pqt[:, 2] == hr, 0], ptop))
                        idx2 = (pDA[:, hr] > p_2most)
                    else:
                        idx2 = np.full(len(pDA[:, hr]), True)

                    curtail_avg = np.dot(scen_prob[idx2], curtail_data[idx2, hr])

                    if curtail_avg > 0:
                        pqt_new[idx, 0] = 0
                        pqt_new = np.vstack([pqt_new, [pHigh, min(curtail_avg[0] + max(0, qtop), poi), hr]])
                    else:
                        pqt_new[idx, 0] = pHigh
                else:
                    pqt_new[idx, 0] = pHigh

            pqt_new = np.vstack([pqt_new, [price_ceil, qmax_val, hr]])

    # pqt_new now contains all the updated entries based on the conditions

    if gridcharging == 0 and np.min(pqt_new[:, 1]) < 0:
        raise ValueError("Heuristic added grid charging and should not")

        #pqt_bids_full.append(np.sort(pqt_new, axis=[2, 0], kind='mergesort'))


    return pqt_new
