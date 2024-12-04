import numpy as np
def get_jenks_breaks(data, lower_class_limits, n_classes):
    data = np.sort(data)
    k = len(data) - 1
    kclass = np.zeros(n_classes + 1)

    # Set the highest and lowest breaks to the max and min of the data
    kclass[-1] = data[-1]  # Last element in data
    kclass[0] = data[0]  # First element in data

    countNum = n_classes

    while countNum > 1:
        # Calculate the index of the current break point
        elt = int(lower_class_limits[k + 1, countNum] - 2)
        kclass[countNum - 1] = data[elt + 1]

        # Move to the previous break point
        k = int(lower_class_limits[k + 1, countNum] - 1)
        countNum -= 1

    return kclass
