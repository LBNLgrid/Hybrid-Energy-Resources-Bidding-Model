import numpy as np

a = np.array([[1,2,2],[2,1,0],[1,0,1]])

indices = np.where(a == 1)
first_occurrence = list(zip(indices[0], indices[1]))
print(indices[1:2])