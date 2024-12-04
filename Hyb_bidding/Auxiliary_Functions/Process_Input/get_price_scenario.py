import numpy as np
from Auxiliary_Functions.Process_Input.Jenks_Breaks_Algorithm.get_jenks import jenks
def stair_breaks_from_scen(scenarios,opt_data,scen_weights):
    # Uses Jenks Natural Breaks algorithm to divide price scenarios in each hour
    # into classes. The only adjustment made to the classes Jenks defines is to
    # ensure no class has positive and negative prices.
    # Then breakpoints for a stairstep bid curve are selected in
    # between these classes according to the "mode". If the breakpoint is being set
    # between a positive and negative class, the break is set at 0, overruling "mode".
    # Inputs:
    #   Scenarios: (# scenarios) x (# intervals in time horizon)
    #   npts: # of break points in bid curve 1 x (# intervals in time horizon) or single integer
    #   mode: Determines how the breakpoints are set between the classes
    #         "mid": Midpoint between classes
    #         "prop": Proportionally between classes so that classes with more points are given a wider price interval
    #   scen_weight (optional): (# scenarios) x (# intervals in time horizon)
    #         Relative likelihood of each scenario. If in "prop" mode and this is not provided, assumes all scenarios have equal weight
    npts = opt_data["nscen_breaks"]
    mode = opt_data["break_mode"]
    nhrs = scenarios.shape[1]
    max_npts = np.max(npts)
    brk_pts = np.zeros((max_npts, nhrs))
    classes = [[[] for _ in range(nhrs)] for _ in range(max_npts)]

    if isinstance(npts, int):
        npts = np.ones((1,nhrs)) * npts
    elif len(npts) != nhrs:
        raise ValueError("npts must have the same time horizon as scenarios, or be a single integer")

    if scen_weights is None:
        scen_weights = np.ones(scenarios.shape)
    elif (scen_weights.shape) != scenarios.shape:
        raise ValueError("scen_weights must have same dimensions as scenarios")
    npts = np.array(npts, dtype=int)
    for h in range(nhrs):
        data = scenarios[:, h]
        unique_data = np.unique(data)
        npts[0][h] = min(npts[0][h], len(unique_data))
        class_counts = np.zeros((npts[0][h], 1))
        class_weights = np.zeros((npts[0][h], 1))
        kclass = jenks(data, npts[0][h])
        for i in range(0,npts[0][h]):
            if i == 0:
                class_data = data[data <= kclass[i + 1]]
            else:
                class_data = data[(data <= kclass[i + 1]) & (data > kclass[i])]

            if min(class_data) * max(class_data) < 0:
                if i == 0:
                    neg_data = data[data < 0]
                    pos_data = data[data >= 0]
                    if np.sum(scen_weights[neg_data, h]) <= np.sum(scen_weights[pos_data, h]):
                        kclass[i + 1] = np.max(neg_data)
                    else:
                        kclass[i + 1] = 0
                class_data = data[(data <= kclass[i + 1]) & (data > kclass[i])]

            classes[i][h] = np.sort(class_data)

        # Adjust break points based on mode
        for i in range(npts[0][h]-1):
            if mode == 'mid':
                brk_pts[i, h] = (np.max(classes[i][h]) + np.min(classes[i + 1][h])) / 2
            elif mode == 'prop':
                weight_sum = np.sum(scen_weights[(data <= kclass[i + 1]) & (data > kclass[i]), h])
                brk_pts[i, h] = (np.max(classes[i][h]) * weight_sum + np.min(
                    classes[i + 1][h]) * weight_sum) / weight_sum
            else:
                raise ValueError("Mode must be 'mid' or 'prop'")

        # Fill remaining break points
        brk_pts[npts[0][h]-1, h] = np.max(data)

    return brk_pts
