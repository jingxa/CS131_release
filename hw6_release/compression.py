import numpy as np


def compress_image(image, num_values):
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    H, W = image.shape
    # Steps:
    #     1. Get SVD of the image
    u, s, vt = np.linalg.svd(image)
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    sigular = np.diag(s[:num_values])       # 格式为对角阵
    compressed_image = u[:, :num_values].dot(sigular).dot(vt[:num_values, :])    # (H,n) * (n, n) * ( n, W)
    #     3. Compute the compressed size, 保留的数据为： (H,n), (sigular), (n, w)
    compressed_size = H * num_values + num_values + num_values * W
    pass
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
