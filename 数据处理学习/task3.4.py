import numpy as np
points_A = np.random.randint(0,100,(5,2))
# print(points_A)
points_B = np.random.randint(0,100,(8,2))
# print(points_B)
distance_matrix = np.sqrt(np.sum((points_A[:, np.newaxis, :]-points_B[np.newaxis, :, :])**2, axis=-1)).round(2)
#np的newaxis，创建新维度，默认为1，把二维数组转为三维数组，确保最小维度是2个元素，即点对，只需一个数组加二维度，一个加一维度即可
#把三维数组看作一本书，(5,1,2)即五页书，每一页都有一个元素，一个元素就是一个点对
#(1,8,2)即一本书只有一页，这一页有八个元素，每个元素是一个点对，广播机制会把(5,1,2)的每一页复制八份变成(5,8,2)，(1,8,2)的这一页复制五份变成(5,8,2)
#然后就相当于把第一个数组第一页的那个点对出现八次，分别和第二个数组的第一页所有点对相减，最终才能得到5*8=40个差值点对，矩阵形状为(5,8,2)，再把每个元素平方一下
#最后用sum对最后一个维度求和，最后一个维度就两个元素，每两个平方和相加就是一个距离的平方，开根完就是距离。得到(5,8)形状的矩阵

# print(distance_matrix)

min_distance_pointA = np.min(distance_matrix,axis=1)  #第二个维度求最小值，去掉第二个维度，即最小维度里各个元素比较，(5,8)变为(5,)
# print(min_distance_pointA)

small_20_point_loc = np.where(np.any(distance_matrix < 20,axis=0))  #在距离矩阵的每一个第一维度中判断是否至少有一个距离小于20，这里第一维度的原因是找B中的点的索引
print(small_20_point_loc)