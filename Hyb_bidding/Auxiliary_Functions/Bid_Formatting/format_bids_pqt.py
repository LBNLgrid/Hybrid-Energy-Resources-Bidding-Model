import pandas as pd
import numpy as np
def format_bids_pqt_to_PSO(pqt_bids, ndate , maxSOC_MWh,hyb_name,  nT, opt_data):
    if pqt_bids.size==0:
        columns = ["Price_USD_per_MWh", "Quantity_MW", "Hour", "Day", "Point", "maxSOC_MWh", "Hybrid_Name"]

        # Create an empty DataFrame with these columns
        bidTable = pd.DataFrame(columns=columns)
    else:
        # Initialize an empty DataFrame
        bidTable = pd.DataFrame()
        price_floor = opt_data["price_floor"]
        price_ceil = opt_data["price_ceil"]
        x_step_min = opt_data["x_step_min"]
        q1min = 1e-10  # opt_data["q1min"]
        soc0_ub = 1
        for i in range(0,ndate):
            pqt = pqt_bids
            ind = np.lexsort((pqt[:, 0].astype(np.float64), pqt[:, 2].astype(np.float64).astype(np.integer)))
            pqt = pqt[ind, :4]
            temp = np.zeros((pqt.shape[0], 2))
            pqt = np.column_stack((pqt, temp))
            for hr in range(nT):
                num_pts = sum((pqt[:, 2] == hr))
                startidx = np.where(pqt[:, 2] == hr)[0][0]
                # Adjust price point to match PSO interpretation (column 3)
                # Identify if duplicate or 0 Q-coordinate: 1=Keep, 0=Duplicate (column 4)
                for k in range(num_pts):
                    if pqt[startidx + k][1] > 0:
                        if k == 0:
                            pqt[startidx + k][3] = price_floor
                            pqt[startidx + k][4] = 1
                        elif pqt[startidx + k][1] >= (pqt[startidx + k - 1][1] + x_step_min):
                            pqt[startidx + k][3] = pqt[startidx + k - 1][0]
                            pqt[startidx + k][4] = 1
                        else:
                            pqt[startidx + k][3] = pqt[startidx + k - 1][0]
                            pqt[startidx + k][4] = 0
                    elif pqt[startidx + k][1] == 0:
                        pqt[startidx + k][3] = pqt[startidx + k][0]
                        pqt[startidx + k][4] = 0
                    else:
                        if pqt[startidx + k][1] <= (pqt[startidx + k + 1][1] - x_step_min):
                            pqt[startidx + k][3] = pqt[startidx + k][0]
                            pqt[startidx + k][4] = 1
                        else:
                            pqt[startidx + k][3] = pqt[startidx + k][0]
                            pqt[startidx + k][4] = 0

            # Rearrange to drop duplicates/0s and return to pqt ordering
            temp1 = np.where(pqt[:, 4] == 1)
            pqtn = pqt[temp1]
            pqtn = pqtn[:, 1:4]
            pqtn = pqtn[:, [2, 0, 1]]
            pqtn[:, 2] = pqtn[:, 2].astype(np.float64).astype(np.integer)
            # Add positive point to any bid curve with only negative points
            hrgrps, hrIds = pd.factorize(pqtn[:, 2])
            hrlyMax = np.zeros(len(hrgrps))
            # Corrected Loop
            for ti in range(0, len(hrgrps)):
                # Accessing the second element of each group indexed by hrgrps[ti]
                if len(pqtn[hrgrps[ti]]) > 1:  # Check if there are enough sub-elements
                    hrlyMax[ti] = np.max(pqtn[hrgrps[ti]][1])  # Assumes the second element is the target
                else:
                    print(f"No second element for group index {hrgrps[ti]}")
            negative_indices = np.where(hrlyMax < 0)[0]
            negHrs = hrIds[negative_indices]
            if not len(negHrs) == 0:
                pqtn = np.vstack([pqtn, np.column_stack(
                    (np.repeat(price_ceil, len(negHrs)).T, np.repeat(q1min, len(negHrs)).T, negHrs))])

            # Add effective 0 bid for any hours that are absent after dropping 0s
            # x = (pqtn[:, 2].astype(np.float64))
            missingHrs = np.setdiff1d(np.arange(nT).astype('U32'), pqtn[:, 2].astype(np.float64).astype(np.integer))
            if not len(missingHrs) == 0:
                pqtn = np.vstack([pqtn, np.column_stack((np.repeat(price_ceil - x_step_min, len(missingHrs)).T,
                                                         np.repeat(q1min, len(missingHrs)).T, missingHrs))])

            # Add second point to any bid curve with only one point
            hrlists, hrCount = np.unique(pqtn[:, 2].astype(np.float64).astype(np.integer), return_counts=True)
            ssHrs = hrlists[hrCount == 1]
            if not len(ssHrs) == 0:
                temp_pqt_ss = np.isin(pqtn[:, 2].astype(np.float64).astype(np.integer), ssHrs)
                pqt_ss = pqtn[temp_pqt_ss]
                pqtn = np.vstack([pqtn, np.column_stack(
                    (np.repeat(price_ceil, len(ssHrs)).T, pqt_ss[:, 1].astype(np.float64) + x_step_min, pqt_ss[:, 2]))])
            ind = np.lexsort((pqtn[:, 0].astype(np.float64), pqtn[:, 2].astype(np.float64).astype(np.integer)))
            pqtn = pqtn[ind, :4]
            # add day column
            zeros_column = np.full((pqtn.shape[0], 1), i)
            pqtn = np.hstack((pqtn, zeros_column))
            # Add "point" column
            zeros_column = np.full((pqtn.shape[0], 1), -1)
            pqtn = np.hstack((pqtn, zeros_column))
            for hr in range(0, nT):
                # print(hr)
                num_pts = sum((pqtn[:, 2].astype(np.float64).astype(np.integer) == hr))
                num_neg_pts = sum(np.multiply((pqtn[:, 2].astype(np.float64).astype(np.integer) == hr),
                                              (pqtn[:, 1].astype(np.float64) < 0)))
                startidx = np.where(pqtn[:, 2].astype(np.float64).astype(np.integer) == hr)[0][0]
                # Create the sequences
                neg_sequence = np.arange(-1 * num_neg_pts, 0, 1)  # Generates sequence from -num_neg_pts to -1
                pos_sequence = np.arange(1, num_pts - num_neg_pts + 1,
                                         1)  # Generates sequence from 1 to (num_pts-num_neg_pts)

                # Concatenate sequences
                full_sequence = np.concatenate((neg_sequence, pos_sequence))
                pqtn[startidx:startidx + num_pts, 4] = full_sequence
            tempx = np.where(pqtn[:, 4] == 1)
            pqtn[tempx, 1] = np.maximum(pqtn[tempx, 1].astype(np.float64), q1min)
            # Assuming 'pqt' is a numpy array or a similar iterable that can be turned into a DataFrame
            temptable = pd.DataFrame(pqtn, columns=["Price_USD_per_MWh", "Quantity_MW", "Hour", "Day", "Point"])
            temptable = temptable[temptable['Hour'].astype(np.float64).astype(np.int64) < 24]

            # Adding new columns for datetime manipulation
            # temptable['StartDate'] = [
            #     datetime(year(dates[i].year), month(dates[i].month), day(dates[i].day), hr, 0, 0).strftime(
            #         '%Y-%m-%d %H:%M') for hr in temptable['hr']]

            # temptable['Point_SOC'] = np.ones(len(temptable))
            temptable['maxSOC_MWh'] = maxSOC_MWh * np.ones(len(temptable))

            # String manipulation for CurveSet and CostCurve columns
            # temptable['CurveSet'] = hyb_name + "-SelfSOCM-CS-" + str(i) + "-" + temptable['Point_SOC'].astype(str)
            temptable['Hybrid_Name'] = hyb_name
            # Concatenating this temporary table to a master bidTable
            bidTable = pd.concat([bidTable, temptable], ignore_index=True)
            ### Bid table for Real time

    return bidTable
