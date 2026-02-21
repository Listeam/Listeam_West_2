from builtins import range
from builtins import object
import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))

        return dists

    def compute_distances_one_loop(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum((self.X_train - X[i])**2,axis=1))

        return dists

    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        test_squared = np.sum(X**2, axis=1,keepdims=True)
    
        train_squared = np.sum(self.X_train**2, axis=1,keepdims=True).T
    
        test_train_dot = X.dot(self.X_train.T)
    
        dists = np.sqrt(test_squared + train_squared - 2 * test_train_dot)  #完全向量化虽然简洁但内存太大了。。，，，

        # dists = np.zeros((num_test, num_train))
        # dists = np.sqrt(np.sum((X[:,np.newaxis,:] - self.X_train[np.newaxis, :, :])**2),axis=-1)

        return dists

    def predict_labels(self, dists, k=1):
    
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):

            closest_y = np.zeros(k)

            sorted_dists = np.argsort(dists[i])  #argsort返回从小到大排列后各元素的原索引值，跟y_train配合，可以直接定位
            
            closest_y = self.y_train[sorted_dists[:k]]  #取出前k个最近邻的标签，用列表做索引
            
            counts = np.bincount(closest_y)  #bincount统计数组中每个数字出现次数，此处即表示某标签出现次数，且统计顺序按照索引值大小排列，比如返回的数组里第一个元素为3，代表0这个标签出现3次
            
            y_pred[i] = np.argmax(counts) #argmax不同于max，它返回的是从小到大求索数组中最大值的索引，刚好与bincount配合，得到出现次数最多的标签，且如果平票则返回较小的标签

        return y_pred
