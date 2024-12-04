import numpy as np


def jenks_matrices_init(data, n_classes):
    # Initialize the matrices with zeros, with dimensions (length of data + 1) by (n_classes + 1)
    lower_class_limits = np.zeros((len(data) + 1, n_classes + 1))
    variance_combinations = np.zeros((len(data) + 1, n_classes + 1))

    # Set default values
    for i in range(1, n_classes + 1):
        lower_class_limits[1, i] = 1
        variance_combinations[1, i] = 0

        for j in range(2, len(data) + 1):
            variance_combinations[j, i] = np.inf

    return lower_class_limits, variance_combinations
