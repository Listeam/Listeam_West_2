import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):  #带动量随机梯度下降
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None     #最原始应该是v = beta * v + 1-beta * dw，然后更新变为w = w - lr * v，两种方法其实是等价的，只是lr取得不同
    v = config['momentum'] * v - config['learning_rate'] * dw  #理解为改由速度决定w的更新，而速度要根据梯度每轮更新，做改变
    next_w = w + v
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):    #Root Mean Square propagation即RMS，均方根传播
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)   #记忆保持率，1-decay_rate就是
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    # cache是对先前所有梯度平方的指数加权平均，相当于保存了所有梯度的记忆，但这个保存会有衰减，每次会记住百分之99的历史，吸收新的百分之1
    # 看似记忆cache占比更大，但其实cache包括之前所有，其实实际的上一次记忆占比肯定比当前梯度记忆的百分之一少
    cache = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw ** 2  #RMSprop就是保留之前的记忆，并根据当前梯度自适应调整学习率大小  
    next_w = w - config['learning_rate'] * dw / (np.sqrt(cache) + config['epsilon'])
    config['cache'] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    t  = config['t']
    t += 1
    m = config['m']
    v = config['v']
    lr = config['learning_rate']
    eps = config['epsilon']
    beta1 = config['beta1']
    beta2 = config['beta2']

    m = beta1 * m + (1 - beta1) * dw  #一阶矩，梯度均值，把梯度改为梯度均值，使得每次下降会受先前动量抵消影响，使得波动变小
    v = beta2 * v + (1 - beta2) * dw ** 2  #二阶矩，梯度方差，根据波动程度调整学习率，防止在局部最小值反复横跳

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)  #偏差修正，是为了刚开始时当前梯度均值占比太小，会使得下降的步伐极为小心，所以除以梯度前的系数，返回它原有的值，经推理得出系数每轮刚好是(1-beta)**t
    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_w, config
