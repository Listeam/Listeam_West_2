import numpy as np
data_matrix = np.arange(100).reshape(10,10)
# print(data_matrix)

sub_matrix = data_matrix[3:7,3:7]
# print(sub_matrix)

bigger_loc = np.where(data_matrix>75)
# print(bigger_loc)

data_matrix[data_matrix>75] = 0
# print(data_matrix)

mini_matrix = 0.8*data_matrix 
# print(mini_matrix)

maxe = np.max(mini_matrix)
max_loc = np.where(mini_matrix==maxe)
# print(maxe)
# print(max_loc)