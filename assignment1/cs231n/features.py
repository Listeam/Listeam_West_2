from __future__ import print_function
from builtins import zip
from builtins import range
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("Done extracting features for %d / %d images" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])  #...代表前面所有维度，即取rgb图像的三个通道分别乘黑白系数


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size(原图像规模)
    orientations = 9  # number of gradient bins 
    cx, cy = (8, 8)  # pixels per cell(每个单元格规模)

    gx = np.diff(image, n=1, axis=1,append=0)  # compute gradient on x-direction(该点x方向陡度,或称梯度强度，大小)
    gy = np.diff(image, n=1, axis=0,append=0)  # compute gradient on y-direction(该点方向陡度)

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude(陡度，其实就是长度)
    grad_orient = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # 用反正切求(x,y)与x轴的夹角角度

    n_cell_sx = int(np.floor(sx / cx))  # number of cells in x(x方向上最多能有几个8*8单元格)
    n_cell_sy = int(np.floor(sy / cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cell_sx, n_cell_sy, orientations))#每个单元格都有九个方向，每个方向有20度的范围

    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        i_orient = np.where(grad_orient < 180/orientations * (i + 1), grad_orient, 0)  #np.where函数，如果跟着原数组就不返回索引而是经筛选后的原数组，不满足条件的都改为0
        i_orient = np.where(grad_orient >= 180/orientations * i, i_orient, 0)
        # select magnitudes for those orientations
        cond2 = i_orient > 0 
        i_mag = np.where(cond2, grad_mag, 0) # 通过判断方向来提取幅值
        
        orientation_histogram[:, :, i] = uniform_filter(i_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T  
        # 滤波器完得到图像形状的该方向的梯度幅值矩阵。
        # cx/2::cx，意思就是取每个单元格的中心.比如8x8,第一个单元格的中心就在4,4(cx/2即4，cy/2即4)，而cx即8就是每隔8行每隔8列再取一次中心，
        # 取中心目的是因为用滤波器计算后的每个中心点值，就是每个单元格的在该方向梯度幅值平均值

    return orientation_histogram.ravel() #得到每个单元格的9个方向的梯度幅值均值


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1) #分界区间，10个区间需要十一个点
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized) #返回直方图y坐标和x坐标，即在该bin区间有多少个和所有分界值的列表，如果要归一化就变成占比即可
    imhist = imhist * np.diff(bin_edges)  #概率等于概率密度乘区间宽度

    return imhist  #得到每个像素在某个色调区间的概率

