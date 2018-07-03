import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    padded = zero_pad(image,Hk//2,Wk//2)
    kernel = np.flip(np.flip(kernel, 0), 1)  # 上下翻转，在左右翻转
    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                        out[m,n]+=kernel[i,j]*padded[m+i,n+j]

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros(shape=(H+pad_height*2,W+pad_width*2),dtype=np.float32)
    out[pad_height:pad_height+H,pad_width:pad_width+W]=image[:,:]
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    padd_H = Hk // 2
    padd_W = Wk // 2
    img_padd = zero_pad(image, padd_H, padd_W)
    kernel = np.flip(np.flip(kernel, 0), 1)  # 上下翻转，在左右翻转
    # 卷积过程
    for i in range(Hi):
        for j in range(Wi):

            out[i, j] = np.sum(img_padd[i:(i+Hk), j:(j+Wk)] * kernel)

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    padd_H = Hk // 2
    padd_W = Wk // 2
    img_padd = zero_pad(f, padd_H, padd_W)
    # 卷积过程
    for i in range(Hi):
        for j in range(Wi):

            out[i, j] = np.sum(img_padd[i:(i+Hk), j:(j+Wk)] * g)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    mean = np.mean(g)
    g = np.subtract(g, mean)
    out = cross_correlation(f,g)
    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    # out = None
    Hk, Wk = f.shape
    Hg, Wg = g.shape
    paddedF = zero_pad(f, Hg // 2, Wg // 2)
    out = np.zeros_like(f)
    # g = np.flip(np.flip(g, 0), 1)
    g_mean = np.mean(g)
    g_delta = np.sqrt(np.var(g))
    g_t = (g - g_mean) / g_delta

    for m in range(Hk):
        for n in range(Wk):
            conv = paddedF[m:(m+Hg), n:(n+Wg)]
            f_mean = np.mean(conv)
            f_delta = np.sqrt(np.var(conv))
            f_t = (paddedF[m:(m+Hg), n:(n+Wg)] - f_mean)/f_delta
            out[m, n] = np.sum(f_t * g_t)

    print('end')

    return out
