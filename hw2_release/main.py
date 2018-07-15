
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io


plt.rcParams['figure.figsize'] = (15.0, 12.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


from edge import *




if __name__ == "__main__":

    # Part 1
    ### 1.1 smoothing
    # Define 3x3 Gaussian kernel with std = 1
    kernel = gaussian_kernel(3, 1)
    kernel_test = np.array(
        [[0.05854983, 0.09653235, 0.05854983],
         [0.09653235, 0.15915494, 0.09653235],
         [0.05854983, 0.09653235, 0.05854983]]
    )

    # Test Gaussian kernel
    if not np.allclose(kernel, kernel_test):  #判断两个arrays的元素是否相等
        print('Incorrect values! Please check your implementation.')
    # Test with different kernel_size and sigma
    kernel_size = 5
    sigma = 1.4

    # # Load image
    img = io.imread('iguana.png', as_grey=True)

    # Define 5x5 Gaussian kernel with std = sigma
    kernel = gaussian_kernel(kernel_size, sigma)

    # Convolve image with kernel to achieve smoothed effect
    smoothed = conv(img, kernel)
    #
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.title('Original image')
    # plt.axis('off')
    #
    # plt.subplot(1,2,2)
    # plt.imshow(smoothed)
    # plt.title('Smoothed image')
    # plt.axis('off')
    #
    # plt.show()

    # ----------------------------------------------

    # 1.2 find gradients
    # Test input
    # I = np.array(
    #     [[0, 0, 0],
    #      [0, 1, 0],
    #      [0, 0, 0]]
    # )
    #
    # # Expected outputs
    # I_x_test = np.array(
    #     [[0, 0, 0],
    #      [0.5, 0, -0.5],
    #      [0, 0, 0]]
    # )
    #
    # I_y_test = np.array(
    #     [[0, 0.5, 0],
    #      [0, 0, 0],
    #      [0, -0.5, 0]]
    # )
    #
    # # Compute partial derivatives
    # I_x = partial_x(I)
    # I_y = partial_y(I)
    #
    #
    # # Test correctness of partial_x and partial_y
    # if not np.all(I_x == I_x_test):
    #     print('partial_x incorrect')
    #
    # if not np.all(I_y == I_y_test):
    #     print('partial_y incorrect')
    #
    # # Compute partial derivatives of smoothed image
    # Gx = partial_x(smoothed)
    # Gy = partial_y(smoothed)
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(Gx)
    # plt.title('Derivative in x direction')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(Gy)
    # plt.title('Derivative in y direction')
    # plt.axis('off')
    #
    # plt.show()

    G, theta = gradient(smoothed)

    if not np.all(G >= 0):
        print('Magnitude of gradients should be non-negative.')

    if not np.all((theta >= 0) * (theta < 360)):
        print('Direction of gradients should be in range 0 <= theta < 360')
    #
    # plt.imshow(G)
    # plt.title('Gradient magnitude')
    # plt.axis('off')
    # plt.show()

    # -------------------------------------------------------------------
    ### 1.3 Non - maximum suppression
    # Test input
    g = np.array(
        [[0.4, 0.5, 0.6],
         [0.3, 0.5, 0.7],
         [0.4, 0.5, 0.6]]
    )

    # Print out non-maximum suppressed output
    # varying theta
    # for angle in range(0, 180, 45):
    #     print('Thetas:', angle)
    #     t = np.ones((3, 3)) * angle  # Initialize theta
    #     print(non_maximum_suppression(g, t))
    #
    nms = non_maximum_suppression(G, theta)
    # plt.imshow(nms)
    # plt.title('Non-maximum suppressed')
    # plt.axis('off')
    # plt.show()


    # ---------------------------------------------------------
    # 1.4 Double Thresholding
    low_threshold = 0.02
    high_threshold = 0.03

    strong_edges, weak_edges = double_thresholding(nms, high_threshold, low_threshold)
    assert (np.sum(strong_edges & weak_edges) == 0)

    edges = strong_edges * 1.0 + weak_edges * 0.5

    # plt.subplot(1, 2, 1)
    # plt.imshow(strong_edges)
    # plt.title('Strong Edges')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(edges)
    # plt.title('Strong+Weak Edges')
    # plt.axis('off')
    #
    # plt.show()


    # ---------------------------------------------
    # 1.5 Edge tracking
    test_strong = np.array(
        [[1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1]]
    )

    test_weak = np.array(
        [[0, 0, 0, 1],
         [0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 1, 0]]
    )

    test_linked = link_edges(test_strong, test_weak)

    # plt.subplot(1, 3, 1)
    # plt.imshow(test_strong)
    # plt.title('Strong edges')
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(test_weak)
    # plt.title('Weak edges')
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(test_linked)
    # plt.title('Linked edges')
    # plt.show()


    # --------------------------------------------------
    # 1.6 canny detector

    # Load image
    # img = io.imread('iguana.png', as_grey=True)

    # Run Canny edge detector
    # edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
    # print(edges.shape)
    # plt.imshow(edges)
    # plt.axis('off')
    # plt.show()



    # --------------------------------------------------------
    # Part 2
    # --------------------------------------------------------
    # 2.1 edge detection
    # Load image
    img = io.imread('road.jpg', as_grey=True)

    # Run Canny edge detector
    edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)

    # plt.subplot(211)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title('Input Image')
    #
    # plt.subplot(212)
    # plt.imshow(edges)
    # plt.axis('off')
    # plt.title('Edges')
    # plt.show()

    # ----------------------------------------------------
    # 2.2 Extracting region of interest
    # Generate mask for ROI (Region of Interest)
    H, W = img.shape
    mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if i > (H / W) * j and i > -(H / W) * j + H:
                mask[i, j] = 1

    # Extract edges in ROI
    roi = edges * mask

    # plt.subplot(1, 2, 1)
    # plt.imshow(mask)
    # plt.title('Mask')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(roi)
    # plt.title('Edges in ROI')
    # plt.axis('off')
    # plt.show()

    # -------------------------------------------
    # 2.3 Fitting lines using Hough transform
    # Perform Hough transform on the ROI
    acc, rhos, thetas = hough_transform(roi)

    # Coordinates for right lane
    xs_right = []
    ys_right = []

    # Coordinates for left lane
    xs_left = []
    ys_left = []

    for i in range(20):
        idx = np.argmax(acc)
        r_idx = idx // acc.shape[1]
        t_idx = idx % acc.shape[1]
        acc[r_idx, t_idx] = 0  # Zero out the max value in accumulator

        rho = rhos[r_idx]
        theta = thetas[t_idx]

        # Transform a point in Hough space to a line in xy-space.
        a = - (np.cos(theta) / np.sin(theta))  # slope of the line
        b = (rho / np.sin(theta))  # y-intersect of the line

        # Break if both right and left lanes are detected
        if xs_right and xs_left:
            break

        if a < 0:  # Left lane
            if xs_left:
                continue
            xs = xs_left
            ys = ys_left
        else:  # Right Lane
            if xs_right:
                continue
            xs = xs_right
            ys = ys_right

        for x in range(img.shape[1]):
            y = a * x + b
            if y > img.shape[0] * 0.6 and y < img.shape[0]:
                xs.append(x)
                ys.append(int(round(y)))

    plt.imshow(img)
    plt.plot(xs_left, ys_left, linewidth=5.0)
    plt.plot(xs_right, ys_right, linewidth=5.0)
    plt.axis('off')
    plt.show()
