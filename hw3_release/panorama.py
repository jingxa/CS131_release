"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))
    #第一步： 偏导数
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    # 第二步： 偏导数乘积
    dxx = dx * dx
    dyy = dy * dy
    dxy = dx * dy

    # 第三步： 形成矩阵
    mxx = convolve(dxx, window)
    mxy = convolve(dxy, window)
    myy = convolve(dyy, window)      #加权计算

    # 第四步： 计算response
    for i in range(H):
        for j in range(W):
            M = np.array([[mxx[i, j], mxy[i, j]], [mxy[i, j], myy[i, j]]])
            response[i, j] = np.linalg.det(M) - k * np.trace(M) ** 2

    return response


def simple_descriptor(patch):
    feature = []

    patch = patch.reshape(-1)
    mean = np.mean(patch)       # 均值
    delta = np.std(patch)       # 标准差
    if delta > 0.0:
        patch = (patch - mean) / delta
    else:
        patch = patch - mean

    feature = list(patch)
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)     # 每个向量的欧式距离 N * N

    idx = np.argsort(dists, axis=1)     # 从小到大对dist排序，返回序号， N * N

    for i in range(N):
        closed_dist = dists[i, idx[i, 0]]
        second_dist = dists[i, idx[i, 1]]
        if(closed_dist < threshold * second_dist):      # 比较
            matches.append([i, idx[i, 0]])

    matches = np.array(matches)
    return matches


def fit_affine_matrix(p1, p2):

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)        # 齐次矩阵

    H = np.linalg.lstsq(p2, p1)[0]
    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):

    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    print(N)
    n_samples = int(N * 0.2)                            # 随机取样

    matched1 = pad(keypoints1[matches[:, 0]])            #第一列的序号 齐次矩阵
    matched2 = pad(keypoints2[matches[:, 1]])            # 第二列的序号

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    for i in range(n_iters):

        temp_max = np.zeros(N, dtype=np.int32)      # 临时变量
        temp_n = 0

        idx = np.random.choice(N, n_samples, replace=False)     # 随机抽取 n_samples
        p1 = matched1[idx, :]
        p2 = matched2[idx, :]
        H = np.linalg.lstsq(p2, p1)[0]              # 临时变换 H
        H[:, 2] = np.array([0, 0, 1])

        temp_max = np.linalg.norm(matched2.dot(H) - matched1, axis=1) ** 2 < threshold      # 计算当前对应点的数量
        temp_n = np.sum(temp_max)

        if temp_n > n_inliers:          # 保存最大数量
            max_inliers = temp_max.copy()
            n_inliers = temp_n

    H = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers])[0]
    H[:, 2] = np.array([0, 0, 1])
    return H, matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):

    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins             # 8 个方向，每个方向 20 度

    # sobel求解梯度
    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # 梯度值和方向
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)            # 划分为block
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    for i in range(rows):
        for j in range(cols):
            for m in range(pixels_per_cell[0]):
                for n in range(pixels_per_cell[1]):
                    idx = int(theta_cells[i, j, m, n] // degrees_per_bin)       # 计算当前像素点 位于n_bins 直方图的那个区间
                    if idx == 9:    # 180 度
                        idx = 8
                    cells[i, j, idx] += G_cells[i, j, m, n]             # 统计

    cells = (cells - np.mean(cells)) / np.std(cells)        # 归一化
    block = cells.reshape(-1)

    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0) # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    # np.fliplr 左右翻转； np.argmax:最大值的下标
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0] # 最大值列

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    left_matrix = np.array(img1_mask,  dtype=np.float64)        # 非常重要。转换为浮点类型
    right_matrix = np.array(img2_mask, dtype=np.float64)

    # 渐进变换区域
    left_matrix[:, left_margin: right_margin] = np.tile(np.linspace(1, 0, right_margin - left_margin), (out_H, 1))
    right_matrix[:, left_margin: right_margin] = np.tile(np.linspace(0, 1, right_margin - left_margin), (out_H, 1))

    img1 = left_matrix * img1_warped
    img2 = right_matrix * img2_warped

    merged = img1 + img2

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    H = []
    robust_matches = []
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)
        # 计算仿射矩阵
        h, robust_m = ransac(keypoints[i], keypoints[i+1], matches[i])
        H.append(h)
        robust_matches.append(robust_m)

    # 变换到一张图片上,使用第二章图片为参考图片
    output_shape, offset = get_output_space(imgs[1], [imgs[0], imgs[2], imgs[3]],
                                            [np.linalg.inv(H[0]), H[1], np.dot(H[1],H[2])])

    img1_warped = warp_image(imgs[0], np.linalg.inv(H[0]), output_shape, offset)
    img1_mask = (img1_warped != -1)  # Mask == 1 inside the image
    img1_warped[~img1_mask] = 0  # Return background values to 0

    img2_warped = warp_image(img[1], np.eye(3), output_shape, offset)
    img2_mask = (img2_warped != -1)  # Mask == 1 inside the image
    img2_warped[~img2_mask] = 0  # Return background values to 0

    img3_warped = warp_image(img[2], H[1], output_shape, offset)
    img3_mask = (img3_warped != -1)  # Mask == 1 inside the image
    img3_warped[~img3_mask] = 0  # Return background values to 0

    img4_warped = warp_image(imgs[3], np.dot(H[1], H[2]), output_shape, offset)
    img4_mask = (img4_warped != -1)  # Mask == 1 inside the image
    img4_warped[~img4_mask] = 0  # Return background values to 0

    merged = linear_blend(img1_warped, img2_warped)
    merged = linear_blend(merged, img3_warped)
    merged = linear_blend(merged, img4_warped)

    return merged
