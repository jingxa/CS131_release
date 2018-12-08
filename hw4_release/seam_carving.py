"""
CS131 - Computer Vision: Foundations and Applications
Assignment 4
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 10/19/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import color


def energy_function(image):
    H, W, _ = image.shape
    out = np.zeros((H, W))
    gray_image = color.rgb2gray(image)

    ### YOUR CODE HERE
    dx, dy = np.gradient(gray_image)
    out = np.abs(dx) + np.abs(dy)
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for i in range(1, H):
        M1 = np.r_[[1e10], cost[i - 1, 0:W - 1]]  # 左边添加一个无穷大
        M2 = cost[i - 1, :]
        M3 = np.r_[cost[i - 1, 1:], [1e10]]  # 右边添加一个无穷大
        M = np.r_[M1, M2, M3].reshape(3, -1)
        cost[i] = energy[i] + np.min(M, axis=0)  # cost
        paths[i] = np.argmin(M, axis=0) - 1  # 上一层的最小值为左上角，正上方，右上角

    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    H, W = paths.shape
    # initialize with -1 to make sure that everything gets modified
    seam = - np.ones(H, dtype=np.int)

    # Initialization
    seam[H-1] = end     # 最底层的像素点位置(H-1,end)

    ### YOUR CODE HERE
    for i in range(H-2,-1,-1):
        seam[i] = seam[i+1]+paths[i+1, seam[i+1]]       # 上层点的width坐标
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)   # 增加一个维度

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE
    out = np.zeros((H, W - 1, C), dtype=image.dtype)        # 返回值，每一行删除一个像素
    for i in range(H):
        out[i, :seam[i], :]=image[i, :seam[i], :]
        out[i, seam[i]:, :]=image[i, seam[i]+1:, :]

    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    # Make sure that `out` has same type as `image`
    assert out.dtype == image.dtype, \
       "Type changed between image (%s) and out (%s) in remove_seam" % (image.dtype, out.dtype)

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    while out.shape[1] > size:
        energy = efunc(out)         # 首先重新计算energy
        cost, paths = cfunc(out, energy)        # 第二步计算cost map
        seam = backtrack_seam(paths, np.argmin(cost[-1]))       #计算optimal seam
        out = remove_seam(out, seam)                            # 移除
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    for i in range(H):
        out[i, :seam[i]+1, :] = image[i, :seam[i]+1, :]
        out[i, seam[i]+1, :] = image[i, seam[i], :]       # 复制seam
        out[i, seam[i]+2:, :] = image[i, seam[i]+1:, :]
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):


    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    while out.shape[1] < size:
        energy = efunc(out)         # 首先重新计算energy
        cost, paths = cfunc(out, energy)        # 第二步计算cost map
        seam = backtrack_seam(paths, np.argmin(cost[-1]))       #计算optimal seam
        out = duplicate_seam(out, seam)                            # 复制
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE
    seams = find_seams(out, size - W)       # 寻找size - W 条 seam
    seams = np.expand_dims(seams, axis=2)   # (H,W)
    for i in range(size - W):
        out = duplicate_seam(out, np.where(seams == i+1)[1])        # 复制当前 seam
        seams = duplicate_seam(seams, np.where(seams == i+1)[1])    # seam也复制
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for i in range(1, H):
        m1 = np.insert(image[i, 0:W-1], 0, 0, axis=0)
        m2 = np.insert(image[i, 1:W], W-1, 0, axis=0)
        m3 = image[i-1]
        c_v = abs(m1 - m2)
        c_v[0] = 0
        c_v[-1] = 0
        c_l = c_v + abs(m3 - m1)
        c_r = c_v + abs(m3 - m2)
        c_l[0] = 0
        c_r[-1] = 0
        i1 = np.insert(cost[i-1, 0:W-1], 0, 1e10, axis=0)
        i2 = cost[i-1]
        i3 = np.insert(cost[i-1, 1:W], W-1, 1e10, axis=0)
        C = np.r_[i1 + c_l, i2 + c_v, i3 + c_r].reshape(3, -1)
        cost[i] = energy[i] + np.min(C, axis=0)
        paths[i] = np.argmin(C, axis=0) - 1
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    # Delete that line, just here for the autograder to pass setup checks
    # other people code
    energy = efunc(out)     # 计算energy
    while out.shape[1] > size:
        cost, paths = cfunc(out, energy)        # 计算cost map
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)       # lowest energy seam

        # Get the seam area
        i = np.min(seam)
        j = np.max(seam)
        out = remove_seam(out, seam)
        if i <= 3:
            energy = np.c_[efunc(out[:, 0: j+2])[:, : -1], energy[:, j+2: ]]
        elif j >= out.shape[1]-3:
            energy = np.c_[energy[:, 0: i-1], efunc(out[:, i-3: ])[:, 2: ]]
        else:
            energy = np.c_[energy[:, 0: i-1], efunc(out[:, i-3: j+2])[:, 2: -1], energy[:, j+2:]]

    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    assert image.shape[:2] == mask.shape

    H, W, _ = image.shape
    out = np.copy(image)

    ### YOUR CODE HERE
    # other people code
    k = 0
    for i in range(W):
        if mask[:, i].any():
            k += 1

    for i in range(k):
        energy = energy_function(out)
        energy[mask] *= -1000
        vcost, vpath = compute_cost(out, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpath, end)
        out = remove_seam(out, seam)
        mask = remove_seam(mask, seam)
    out = enlarge(out, W)
    ### END YOUR CODE

    assert out.shape == image.shape

    return out
