from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim

        self.params["W1"] = np.random.normal(0.0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params["b1"] = np.zeros(num_filters)

        pool_height = 2
        pool_width = 2
        stride = 2
        pool_H_out = 1 + (H - pool_height) // stride
        pool_W_out = 1 + (W - pool_width) // stride
        unravel_input_dim = pool_H_out * pool_W_out * num_filters  
        # 和affine层还隔了个池化层，需要进一步计算用于全连接层W的第一维度(到全连接层开始多维数组需要展开
        self.params["W2"] = np.random.normal(0.0, weight_scale, (unravel_input_dim, hidden_dim))
        self.params["b2"] = np.zeros(hidden_dim)

        self.params["W3"] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params["b3"] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_out, W2, b2)
        affine_out, affine_cache = affine_forward(affine_relu_out, W3, b3)

        scores = affine_out

        if y is None:
            return scores

        loss, grads = 0, {}
        loss_last, dout_last = softmax_loss(scores, y)
        dout3, dw3, db3 = affine_backward(dout_last, affine_cache)
        dout2, dw2, db2 = affine_relu_backward(dout3, affine_relu_cache)
        dout1, dw1, db1 = conv_relu_pool_backward(dout2, conv_cache)

        reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))  
        loss = loss_last + reg_loss

        grads['W1'] = dw1 + self.reg * W1  
        grads['b1'] = db1
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3

        return loss, grads
