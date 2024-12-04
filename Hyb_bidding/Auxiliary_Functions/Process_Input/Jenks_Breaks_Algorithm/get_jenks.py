import numpy as np
from Auxiliary_Functions.Process_Input.Jenks_Breaks_Algorithm.get_jenks_matrics import jenks_matrices
from Auxiliary_Functions.Process_Input.Jenks_Breaks_Algorithm.get_get_jenks_breaks import get_jenks_breaks

def jenks(data, n_classes):
    data = np.array(data)
    unique_data = np.unique(data)
    if n_classes > len(data):
        raise ValueError("n_classes greater than data size")
    elif n_classes > len(unique_data):
        raise ValueError("n_classes greater than number of unique data points")

    if n_classes == len(unique_data):
        kclass1 = np.sort(unique_data,axis=0)
        kclass = np.reshape(kclass1, ( n_classes))
        kclass=np.hstack((kclass[1],kclass))
    else:
        data = np.sort(data)
        lower_class_limits,temp = jenks_matrices(data, n_classes)
        kclass = get_jenks_breaks(data, lower_class_limits, n_classes)

    return kclass
