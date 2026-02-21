from .layers import *


def affine_relu_forward(x, w, b):

    
    a, af_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)  
    cache = (af_cache, relu_cache)  
    return out, cache


def affine_relu_backward(dout, cache):
    
    af_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)  
    dx, dw, db = affine_backward(da, af_cache) 
    return dx, dw, db

