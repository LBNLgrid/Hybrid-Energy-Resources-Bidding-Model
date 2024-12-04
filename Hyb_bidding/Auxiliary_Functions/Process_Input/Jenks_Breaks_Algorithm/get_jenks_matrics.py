
import numpy as np
from Auxiliary_Functions.Process_Input.Jenks_Breaks_Algorithm.get_jenks_matrices_init import jenks_matrices_init
def jenks_matrices(data, n_classes):
    data = np.sort(data)
    lower_class_limits, variance_combinations = jenks_matrices_init(data, n_classes)
    variance = 0

    for lim in range(1, len(data)):
        sum_ = 0
        sum_squares = 0
        w = 0

        for m in range(0, lim):
            lower_class_limit = lim - m
            val = data[lower_class_limit]

            w = w+1
            sum_ = sum_+val
            sum_squares += val * val

            variance = sum_squares - (sum_ * sum_) / w
            i4=lower_class_limit-1
            if i4 != -1:
                for j in range(1, n_classes):
                    if variance_combinations[lim+1, j+1] >= (
                            variance + variance_combinations[i4+1, j]):
                        lower_class_limits[
                            lim+1, j+1] = lower_class_limit  # +1 to adjust for Python's 0-indexing
                        variance_combinations[lim+1, j+1] = variance + variance_combinations[i4+1, j]

        lower_class_limits[lim+1, 1] = 1
        variance_combinations[lim+1, 1] = variance

    return lower_class_limits, variance_combinations
