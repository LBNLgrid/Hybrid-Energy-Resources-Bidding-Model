def get_high_low_price_heuristic(pqt_bids_base,opt_sol,pDA,gDA,scen_prob,hrs_to_mod,pLow_thresh,pHigh_thresh,pHoriz_scale,price_ceil,esr,poi,gridcharging,n_soc) :
    import numpy as np
    class CustomError(Exception):
        pass
    #def thresh_fun(v1, v2):
    if pLow_thresh > pHigh_thresh:
            raise CustomError("pLow_thresh must be <= pHigh_thresh")

    ns=200
    for i in range (n_soc):
        pqt = pqt_bids_base[i]
        pqt_new = pqt.copy()
        if gridcharging == 0 and np.min(pqt[:,1] < 0) :
           raise ValueError("Original bids contain grid charging and should not")
       #### low price heuristic
        for hr in hrs_to_mod:
            q0 = np.min(pqt[pqt[:, 2] == hr, 1])
            p0 = np.min(pqt[pqt[:, 2] == hr, 0])
            if q0 > -esr.power:
               pmin = np.min(pDA[i*ns:(i+1)*ns, hr])
               pLow_horizon = np.min(pDA[i*ns:(i+1)*ns,])
               pLow = min([pLow_thresh, pHoriz_scale * pLow_horizon, pLow_horizon / pHoriz_scale])
               if q0 >= 0 and pmin >= 0:
                  if pLow >= 0:
                     soc = opt_sol[i].soc
                     curtail_avg = np.dot(scen_prob[i*ns:(i+1)*ns,].T, np.maximum(0, gDA[i*ns:(i+1)*ns, hr] - np.minimum(esr.power,esr.max - soc[:,hr])))
                     if 0 < curtail_avg < q0:
                        pqt_new = np.vstack([pqt_new, [pLow, curtail_avg[0], hr], [0, -esr.power * gridcharging, hr]])
                     elif curtail_avg >= q0:
                          pqt_new = np.vstack(([pqt_new, [0, -esr.power * gridcharging, hr]]))
                     else:
                         q_from_grid = np.dot(scen_prob[i*ns:(i+1)*ns,].T, np.minimum(0, gDA[i*ns:(i+1)*ns, hr] - esr.power))
                         pqt_new = np.vstack(([pqt_new, [pLow, q_from_grid[0] * gridcharging, hr], [0, -esr.power * gridcharging, hr]]))
                  else:  # pLow < 0
                      idx2 = (pDA[i*ns:(i+1)*ns, hr] <= p0)
                      qEsr_avg = np.dot(scen_prob[i*ns:(i+1)*ns,][idx2], opt_sol[i].Pesr[idx2, hr, 0])
                      pqt_new = np.vstack(([pqt_new, [0, np.maximum(0, np.minimum(q0, qEsr_avg[0])), hr],[pLow, -esr.power * gridcharging, hr]]))
               else:
                   pqt_new = np.vstack([pqt_new, [pLow, -esr.power * gridcharging, hr]])

        #     High  price heuristic
        pqt_bids_full = []
        # for i in range(len(pqt_bids_base)):
        #     pqt = pqt_bids_base[i]
        #     pqt_new = pqt.copy()
        for hr in hrs_to_mod:
            qtop = np.max(pqt[pqt[:, 2] == hr, 1])
            qmax_byscen = np.minimum(poi, esr.power + gDA[i * ns:(i + 1) * ns, hr + 1])
            qmax_val = np.mean(qmax_byscen)

            if qtop < qmax_val:
                ptop = np.max(pqt[pqt[:, 2] == hr, 0])
                pmax = np.max(pDA[i * ns:(i + 1) * ns, hr])
                pHigh_horizon = np.max(pDA[i * ns:(i + 1) * ns, ])
                pHigh = max([pHigh_thresh, pHoriz_scale * pHigh_horizon, pHigh_horizon / pHoriz_scale])

                idx = np.where((pqt_new[:, 0] == ptop) & (pqt_new[:, 1] == qtop) & (pqt_new[:, 2] == hr))[0]

                if pmax >= 0:
                    pqt_new[idx, 0] = pHigh
                else:
                    if pHigh > 0:
                        curtail_data = opt_sol[i].curtail

                        if len(np.unique(pqt[pqt[:, 2] == hr, 0])) > 1:
                            p_2most = np.max(np.setdiff1d(pqt[pqt[:, 2] == hr, 0], ptop))
                            idx2 = (pDA[i * ns:(i + 1) * ns, hr] > p_2most)
                        else:
                            idx2 = np.full(len(pDA[i * ns:(i + 1) * ns, hr]), True)

                        curtail_avg = np.dot(scen_prob[i * ns:(i + 1) * ns][idx2], curtail_data[idx2, hr])

                        if curtail_avg > 0:
                            pqt_new[idx, 0] = 0
                            pqt_new = np.vstack(([pqt_new, [pHigh, min(curtail_avg + max(0, qtop), poi), hr]]))
                        else:
                            pqt_new[idx, 0] = pHigh
                    else:
                        pqt_new[idx, 0] = pHigh

                pqt_new = np.vstack([pqt_new, [price_ceil, qmax_val, hr]])

        if gridcharging == 0 and np.min(pqt_new[:, 1]) < 0:
            raise ValueError("Heuristic added grid charging and should not")

        #pqt_bids_full.append(np.sort(pqt_new, axis=[2, 0], kind='mergesort'))

#
    return pqt_new
