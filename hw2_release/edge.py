import numpy as np

def conv(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    # 卷积过程
    kernel = np.flip(np.flip(kernel, 0), 1)  # 上下翻转，在左右翻转
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded[i:(i + Hk), j:(j + Wk)] * kernel)


    return out



def gaussian_kernel(size, sigma):
    
    kernel = np.zeros((size, size))
    # size = 2k+1
    k = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-(((i-k)**2 + (j-k)**2))/2*sigma**2) / (2 * np.pi * (sigma**2))

    return kernel


def partial_x(img):
    out = None
    # Hi, Wi = img.shape
    # padd = np.zeros((Hi, Wi+2))
    # padd[:, 1:Wi+1] = img
    #
    # out = np.zeros((Hi, Wi))
    # for i in range(Wi):
    #     out[:, i] = (padd[:, i+2] - padd[:, i]) / 2

    # 使用卷积
    kernel = np.array([[1, 0, -1]]) / 2
    out = conv(img, kernel)
    return out


def partial_y(img):
    out = None
    # Hi, Wi = img.shape
    # padd = np.zeros((Hi + 2, Wi))
    # padd[1:Hi+1, :] = img
    #
    # out = np.zeros((Hi, Wi))
    # for i in range(Hi):
    #     out[i, :] = (padd[i+2, :] - padd[i, :]) / 2


    kernel = np.array([[1, 0, -1]]).T / 2
    out = conv(img, kernel)
    return out


def gradient(img):

    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    gx = partial_x(img)
    gy = partial_y(img)

    G = np.sqrt(gx**2 + gy**2)

    theta = np.arctan2(gy, gx)  # (-pi/2, pi/2)

    theta = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 360


    return G, theta


def non_maximum_suppression(G, theta):

    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45  # 方向定位
    # 添加一层padding
    padd = np.zeros((H+2,W +2))
    padd[1:H+1, 1:W+1] = G
    for m in range(1, H+1):
        for n in range(1, W+1):
            # 题目定义为顺时针方向，和逆时针相反,y方向相反
            rad = np.deg2rad(theta[m-1, n-1])
            i =int(np.around(np.sin(rad)))   # 行
            j =int(np.around(np.cos(rad)))   # 列
            p1 = padd[m+i, n+j]
            p2 = padd[m-i, n-j]
            if(padd[m, n] > p1 and padd[m, n] > p2): # 一个方向上
                out[m-1, n-1] = padd[m, n]
            else:
                out[m-1, n-1] = 0

    return out

def double_thresholding(img, high, low):

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    strong_edges = np.where(img>=high,1,0)
    weak_edges = np.where(img>=low,1,0)
    weak_edges = weak_edges-strong_edges

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    # 返回(y,x)周围等于(y,x)的邻居的坐标列表[(i,j),(i2,j2)..]
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    # 寻找强边像素相连的弱边像素
    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    edges = np.copy(strong_edges)
    for i in range(1, H-1):
        for j in range(1, W-1):
            neighbors = get_neighbors(j, i, H, W)
            if weak_edges[i, j] and np.any(edges[x, y] for x, y in neighbors):
                edges[i, j] = True
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):

    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)  # 1. smoothing
    smoothed = conv(img,kernel)
    G, theta = gradient(smoothed)                 # 2. 梯度计算
    nms = non_maximum_suppression(G,theta)        # 3. non-maximum_suppression
    strong_edges, weak_edges = double_thresholding(nms, low,high) #double thresholding
    edge = link_edges(strong_edges,weak_edges)    # link
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))  # 对角线长度，np.ceil 向大取整
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1) # 2倍，等差数列
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i, j in zip(ys, xs):
        for idx in range(thetas.shape[0]):
            r = j * cos_t[idx] + i * sin_t[idx]
            accumulator[int(r + diag_len), idx] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas
