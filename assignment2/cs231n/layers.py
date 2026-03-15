from builtins import range
import numpy as np


def affine_forward(x, w, b):

    out = None
    num_train = x.shape[0]
    x_vector = x.reshape((num_train,-1))
    out = x_vector.dot(w) + b
    
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    num_train = x.shape[0]
    x_vector = x.reshape((num_train,-1))
    #dx作为中间值，只是用于计算前一层的梯度，运用公式dL/dx = dL/dout * dout/dx = sofatmax误差矩阵 * w.T
    dx = dout.dot(w.T).reshape(x.shape)  #变回多维形状
    dw = x_vector.T.dot(dout) 
    db = np.sum(dout,axis=0) 

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs)/矫正线性单元.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0,x)  
    
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout.copy()
    dx[x<=0] = 0

    return dx


def softmax_loss(x, y):
    
    loss, dout = None, None
    num_train = x.shape[0]
    scores = x - np.max(x,axis=1,keepdims=True)
    e_possibility = np.exp(scores)
    e_possibility /= np.sum(e_possibility,axis=1,keepdims=True)
    loss = 1/num_train * -np.sum(np.log(e_possibility[np.arange(num_train),y]))
    error = e_possibility.copy()
    error[np.arange(num_train),y] -= 1
    dout = 1/num_train * error #这里就除掉，后面dwdb就不用除

    return loss, dout

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == "train":
        sample_mean = np.mean(x,axis=0)  #以特征为标准分批，一列一批
        sample_var = np.var(x,axis=0)  #variance为方差

        std = np.sqrt(sample_var+eps) #standard deviation标准差
        x_minus_mean = x - sample_mean

        x_hat = x_minus_mean / std
        out = gamma * x_hat + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var 

        cache = (x, x_hat, gamma, std, x_minus_mean)

    elif mode == "test":
        
        x_hat = (x-running_mean) / np.sqrt(running_var+eps)  #训练完会得出迭代完毕的running统计量，用于测试，因为测试时就不用反向传播了
        out = gamma * x_hat + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, x_hat, gamma, std, x_minus_mean = cache
    N = x.shape[0]
    
    dbeta = np.sum(dout,axis=0)
    dgamma = np.sum(dout*x_hat,axis=0)  #分批归一层求梯度都用逐元素相乘求和，区别全连接层的点乘
    inv_N = 1 / N
    inv_std = 1 / std
    # x_hat = x-mean/std或x-mean/(根号下var-eps)
    dx_hat = dout * gamma  # dL/dx = dL/dx_hat * dx_hat/dx,该求法类比先前的全连接层。把点乘改为乘即可
    dx_hat_dx = inv_std    # dL/dx_hat = dL/dout * dout/dx_hat,out就是上面得出的out=gamma*x_hat + beta

    dvar = np.sum(-0.5 * dx_hat * inv_std ** 3 * x_minus_mean, axis=0)   # dL/dx = dL/dvar * dvar/dx
    dvar_dx = 2 * inv_N * x_minus_mean    # dL/dvar = dL/dx_hat * dx_hat/dvar ，配凑就看哪些变量表达式里包括了该变量

    dmean_from_var = np.sum(-2 * dvar * inv_N * x_minus_mean, axis=0) # c
    dmean_from_xhat = np.sum(dx_hat * (-inv_std), axis=0)  # dL/dmean = dL/dx_hat * dx_hat/dmean + dL/dvar * dvar/dmean
    dmean_dx = inv_N

    dx = dx_hat * dx_hat_dx + dvar * dvar_dx + (dmean_from_var + dmean_from_xhat) * dmean_dx  #虽然不懂但是记住法则:对x求偏导的项不用求和，对统计量或参数求偏导的项都要求和
    
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
   
    x, x_hat, gamma, std, x_minus_mean = cache
    N = x.shape[0]
    
    dbeta = np.sum(dout,axis=0)
    dgamma = np.sum(dout * x_hat,axis=0) 
    inv_N = 1 / N
    inv_std = 1 / std

    dx_hat = dout * gamma  
    part1 = inv_std * dx_hat

    dvar_simp = -np.sum(dx_hat, axis=0, keepdims=True) 
    part2 = inv_std * inv_N * dvar_simp

    dmean_simp = -x_hat * np.sum(dx_hat * x_hat, axis=0, keepdims=True)
    part3 = inv_N * inv_std * dmean_simp

    dx = part1 + part2 + part3


    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
  
    sample_mean = np.mean(x,axis=1,keepdims=True) 
    sample_var = np.var(x,axis=1,keepdims=True) 

    std = np.sqrt(sample_var+eps) 
    x_minus_mean = x - sample_mean

    x_hat = x_minus_mean / std
    out = gamma * x_hat + beta

    cache = (x, x_hat, gamma, std, x_minus_mean)   #层归一化不用分模式，因为之前批量归一化是样本组敏感的，换成测试组就不一样了，所以需要保存历史，层归一化是对单样本而言的，对测试组也只是一样的步骤
    
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    x, x_hat, gamma, std, x_minus_mean = cache
    D = x.shape[1]
    
    dbeta = np.sum(dout,axis=0)
    dgamma = np.sum(dout * x_hat,axis=0)   #层归一化只是把归一方式改变了，gamma系数还是对于特征维度的
    inv_D = 1 / D
    inv_std = 1 / std

    dx_hat = dout * gamma  
    part1 = inv_std * dx_hat

    dvar_simp = -np.sum(dx_hat, axis=1, keepdims=True) 
    part2 = inv_std * inv_D * dvar_simp

    dmean_simp = -x_hat * np.sum(dx_hat * x_hat, axis=1, keepdims=True)
    part3 = inv_D * inv_std * dmean_simp

    dx = part1 + part2 + part3

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        matrix = np.random.rand(*x.shape)  #生成0~1随机矩阵，当样本数足够庞大，小于p的个数就是样本数乘上百分p
        mask = (matrix < p).astype(x.dtype)  #x.shape要用*解包，rand要求参数是整数而不是元组;mask必须非1即0，不能用where返回原来的随机值
        out = x * mask

    elif mode == "test":
        out = x
        
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout 
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride = conv_param.get('stride', 1)  # 步长决定卷积核移动幅度，越小计算量越大，特征图尺寸越大，保留了越多细节
    pad = conv_param.get('pad', 0)   # pad代表填充格数，为了保持输出图尺寸和输入相同的参数，和步长共同作为训练前设定的参数，不参与训练
        
    N, C, H, W = x.shape   # C是样本图像的颜色通道数，HW代表形状，这样的顺序排布保证最后得到每个样本的多个特征图
    F, _, HH, WW = w.shape   # w在卷积神经网络里面是滤波器，F是滤波器个数即卷积核个数，代表最后获得特征图个数，HHWW是卷积核的'窗口'尺寸
        
    H_out = 1 + (H + 2 * pad - HH) // stride  # 输出尺寸的公式
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    if pad > 0:
        x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')  #对输入图像做填充0处理
    else:
        x_pad = x
        
    out = np.zeros((N, F, H_out, W_out))   #初始化输出图像集的形状，一般Hout会等于H
        
    for h_out in range(H_out):  #Height,一行一行滑动
        for w_out in range(W_out):  #Width
            h_start = h_out * stride   #计算，定位每一次卷积操作，滑到窗口内的所有元素
            h_end = h_start + HH 
            w_start = w_out * stride
            w_end = w_start + WW
            
            x_windows = x_pad[:, :, h_start:h_end, w_start:w_end]  # 将x形状成功转化为和W类似的(N, C, HH, WW)，作为卷积计算的单位x
                
            for f in range(F):  #对每次的单位x都要进行线性变换，W有几个滤波器就有几次线性变换(逐元素相乘求和)
                filter_w = w[f][np.newaxis, :, :, :]  # W形状取完是(C, HH, WW)，需要广播成(1, C, HH, WW)
                conv_sum = np.sum(x_windows * filter_w, axis=(1, 2, 3))  #只保留第一个维度，即样本个数，意味着一个样本对应一个求和值，要理解卷积计算的本质就是对每个窗口进行线性变换求和作为新矩阵的元素
                out[:, f, h_out, w_out] = conv_sum + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant') #重复填充0操作

    dx_pad = np.zeros_like(x_pad)
    
    db = np.sum(dout, axis=(0, 2, 3)) # 保留第二个维度即滤波器个数
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    x_window = x_pad[n, :, h_start:h_end, w_start:w_end]
                    dw[f] += dout[n, f, i, j] * x_window   # dw = dout * x,公式不变，只不过此时的x是卷积后的单位x，并且需要手动求和，因此对应的dout也要对应窗口的位置,保证dw形状(F,C,H,W)
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j] # dx = dout * w公式也同样不变

    if pad > 0:
        dx = dx_pad[:, :, pad:-pad, pad:-pad]  #去掉填充处
    else:
        dx = dx_pad
    
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out)) 
    for h_out in range(H_out): 
        for w_out in range(W_out): 
            h_start = h_out * stride 
            h_end = h_start + pool_height
            w_start = w_out * stride
            w_end = w_start + pool_width

            x_window = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, h_out, w_out] = np.max(x_window, axis=(2,3))  #max池化层本质也是一个滑动的窗口，对卷积层返回的多个特征图各做池化处理，提取出最显著的特征

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    dx = np.zeros_like(x) #初始化为0，在最大值处填充梯度就行，因为不是最大值，梯度就是0，dx = dout * dmax_fun

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    x_window = x[n, c, h_start:h_end, w_start:w_end]
                    max_index_window = np.unravel_index(np.argmax(x_window), x_window.shape)
                    max_index_global_h = h_start + max_index_window[0]
                    max_index_global_w = w_start + max_index_window[1]

                    dx[n, c, max_index_global_h, max_index_global_w] += dout[n, c, i, j]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, C, H, W = x.shape             # SBN属于N依赖型,面对所有样本，求的是每个通道的各个统计量，以此进行归一化
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)  #用之前的bn模式,大大减少步骤数，只需要调整输入数组的形状即可

    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W  = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward_alt(dout_reshaped, cache)
    dx = dx_reshaped.reshape(N, H, W, C).transpose((0, 3, 1, 2))

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)        # SGN 区别于SBN，它不依赖N总样本数，而是将视角投于单个样本，对单个样本的各个通道进行分组，求出各组的像素统计量，求的是每个样本的各通道组的统计量
    N, C, H, W = x.shape
    x_reshaped = x.reshape(N, G, C//G, H, W)

    mean = np.mean(x_reshaped, axis=(2,3,4), keepdims=True)
    var = np.var(x_reshaped, axis=(2,3,4), keepdims=True)
    std = np.sqrt(var + eps)
    std_reshaped = std.reshape(N*G, -1)   # 对于SGN，有着类似层归一化的逻辑，但是非常易错的是各参数形状的调整，特别是std和x，因为有反复多次的reshape，比较容易乱

    x_normalized_reshaped = (x_reshaped - mean) / std
    x_normalized = x_normalized_reshaped.reshape(N, C, H, W)

    out = gamma * x_normalized + beta
    cache = (x, x_normalized, x_normalized_reshaped, gamma, std_reshaped, G)

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None
    x, x_normalized, x_normalized_reshaped, gamma, std_reshaped, G = cache
    N, C, H, W = dout.shape
    dout_reshaped = dout.reshape(N*G, -1)
    x_normalized_reshaped_2 = x_normalized_reshaped.reshape(*dout_reshaped.shape)

    dbeta = np.sum(dout,axis=(0,2,3),keepdims=True)
    dgamma = np.sum(dout * x_normalized,axis=(0,2,3),keepdims=True) 

    D = dout_reshaped.shape[1]

    inv_D = 1 / D
    inv_std = 1 / std_reshaped

    dx_hat = (dout * gamma).reshape(N*G, -1)
    part1 = inv_std * dx_hat
    
    dvar_simp = -np.sum(dx_hat, axis=1, keepdims=True) 
    part2 = inv_std * inv_D * dvar_simp
    
    dmean_simp = -x_normalized_reshaped_2 * np.sum(dx_hat * x_normalized_reshaped_2, axis=1, keepdims=True)
    part3 = inv_D * inv_std * dmean_simp

    dx_reshaped = part1 + part2 + part3

    dx = dx_reshaped.reshape(N, C, H, W)

    return dx, dgamma, dbeta
