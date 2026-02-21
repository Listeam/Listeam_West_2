from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg): #交叉熵损失函数

    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    pred_y = np.zeros((num_train,num_classes))


    for i in range(num_train):  #loss公式拆解为三步，1求每个类别预测出来的对应分数并指数化，归一化，得到初步预测概率，2取对数得到每个类别的对数概率，再以样本正确标签作为索引，找到该模型对这个正确标签的预测概率，,3对每个样本相同操作得到各个loss求平均
        scores = X[i].dot(W)  #1个样本的D个特征根据对应C个类别的加权，求和得到C个分数，(1,D+1)*(D+1,C)
        scores -= np.max(scores) 

        ey_i = np.exp(scores) #y=wx,ey=e^(wx)
        ey_i = ey_i / np.sum(ey_i)  #都化为指数形式保证正值，每个类别的分数都除以总和，保证得到小于1的正数

        pred_y[i] = ey_i

        log_ey_i = np.log(ey_i)  #对数概率

        loss -= log_ey_i[y[i]]   #损失实质就是对应类别的对数概率的负值，为正数

    loss = loss / num_train + reg * np.sum(W**2)  #reg为正则化强度，控制惩罚力度，W平方即L2正则化又称权重衰减，当模型太过自信，某个类别的预测概率极大，就会使损失加大，即惩罚，从而使最终得到的概率分布更均匀不极端

    error = pred_y.copy()
    error[np.arange(num_train),y] -= 1   #二维数组的数组索引法，两个数组元素分别对应每个样本的正确标签位置
                                         #本来的梯度下降应是预测概率减去真实概率，但是在softmax中真实概率是独热编码，所以直接在预测概率矩阵中对应正确标签的位置减1即可得到误差矩阵
    dW = (1/num_train)*X.T.dot(error) + 2*reg*W  #后项正则化其实就是loss里面的正则化求导的结果
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    

    scores = X.dot(W)
    scores -= np.max(scores,axis=1,keepdims=True)  #keepdims比reshape方便点，防止维度不匹配

    e_possibility = np.exp(scores)
    e_possibility /= np.sum(e_possibility,axis=1,keepdims=True)

    loss_total = np.sum(np.log(e_possibility[np.arange(num_train),y]))
    loss = -loss_total/num_train + reg * np.sum(W**2)

    error = e_possibility.copy()
    error[np.arange(num_train),y] -= 1   
    dW = (1/num_train)*X.T.dot(error) + 2*reg*W  
    
    return loss, dW
