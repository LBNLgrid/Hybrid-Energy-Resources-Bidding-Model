def format_bids_pqt_to_PSO(pqt_bids, dates, nT, maxSOC_MWh,soc0_upper_bounds, hyb_name, price_floor,price_ceil, x_step_min, q1min):
    import pandas as pd
    import numpy as np
    # Initialize an empty DataFrame
    bidTable = pd.DataFrame()
    soc0_ub = soc0_upper_bounds
    for i in range(len(dates)):
        for j in range(len(soc0_ub) - 1):
            pqt = pqt_bids[i][j]
            pqt = pqt.sort_values(by=[3, 1], ascending=[True, True])

            for hr in range(nT):
                num_pts = sum((pqt[3] == hr))
                startidx = pqt[3] == hr
                for k in range(num_pts):
                    if pqt.iloc[startidx + k, 1] > 0:
                        if k == 0:
                            pqt.iloc[startidx + k, 3:5] = [price_floor, 1]
                        elif pqt.iloc[startidx + k, 1] >= (pqt.iloc[startidx + k - 1, 1] + x_step_min):
                            pqt.iloc[startidx + k, 3:5] = [pqt.iloc[startidx + k - 1, 1], 1]
                        else:
                            pqt.iloc[startidx + k, 3:5] = [pqt.iloc[startidx + k - 1, 1], 0]
                    elif pqt.iloc[startidx + k, 1] == 0:
                        pqt.iloc[startidx + k, 3:5] = [pqt.iloc[startidx + k, 1], 0]
                    else:
                        if pqt.iloc[startidx + k, 1] <= (pqt.iloc[startidx + k + 1, 1] - x_step_min):
                            pqt.iloc[startidx + k, 3:5] = [pqt.iloc[startidx + k, 1], 1]
                        else:
                            pqt.iloc[startidx + k, 3:5] = [pqt.iloc[startidx + k, 1], 0]

                # Rearrange to drop duplicates/0s and return to pqt ordering
                pqt = pqt[pqt[4] == 1][[3, 1, 2]]
                pqt = pqt.sort_values(by=[3, 1], ascending=[True, True])

                # Add positive point to any bid curve with only negative points
                hrlyMax = pqt.groupby(3).max().iloc[:, 1]
                negHrs = hrlyMax[hrlyMax < 0].index
                if not negHrs.empty:
                    pqt = pd.concat([pqt, pd.DataFrame(data={'1': np.repeat(price_ceil, len(negHrs)),'2': np.repeat(q1min, len(negHrs)),'3': negHrs})], ignore_index=True)

                # Add effective 0 bid for any hours that are absent after dropping 0s
                missingHrs = np.setdiff1d(np.arange(nT), pqt[3].unique())
                if not np.isnan(missingHrs).all():
                    pqt = pd.concat([pqt, pd.DataFrame(data={'1': np.repeat(price_ceil - x_step_min, len(missingHrs)),'2': np.repeat(q1min, len(missingHrs)),'3': missingHrs})], ignore_index=True)

                # Add second point to any bid curve with only one point
                hrCount = pqt.groupby(3).size().reset_index(name='count')
                ssHrs = hrCount[hrCount['count'] == 1]['3']
                if not ssHrs.empty:
                    pqt_ss = pqt[pqt[3].isin(ssHrs)]
                    pqt_append = pd.DataFrame(data={'1': np.repeat(price_ceil, len(ssHrs)),'2': pqt_ss.iloc[:, 1] + x_step_min, '3': ssHrs})
                    pqt = pd.concat([pqt, pqt_append], ignore_index=True)

                pqt = pqt.sort_values(by=[3, 1], ascending=[True, True])

                # Add "point" column
                pqt['4'] = 0
                for hr in range(nT):
                    num_pts = sum((pqt[3] == hr))
                    num_neg_pts = sum((pqt[3] == hr) & (pqt[2] < 0))
                    startidx = pqt[3] == hr
                    pqt.iloc[startidx, 4] = np.concatenate([-1 * np.arange(1, num_neg_pts + 1), np.arange(1, num_pts - num_neg_pts + 1)])

                # Ensure point 1 is sufficiently positive
                pqt.loc[pqt[4] == 1, 2] = np.maximum(pqt.loc[pqt[4] == 1, 2], q1min)

                temptable = pqt.rename(columns={"1": "P", "2": "Q", "3": "hr", "4": "Point"})
                temptable = tempt
