import random
import numpy as np
import os
from linalg import *
from imageManip import *

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def display(img):
    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()




if __name__ == "__main__":

    # Question 1.1
    M = np.arange(1,13).reshape(4,3)
    a = np.array([1, 1, 0])
    b = np.array([-1, 2, 5]).reshape(3,1)

    # END CODE HERE
    print('=====Question 1.1=========')
    print("M = \n", M)
    print("a = ", a)
    print("b = ", b)

    # Question 1.2
    print('=====Question 1.2=========')
    aDotB = dot_product(a, b)
    print(aDotB)

    # Question 1.3
    print('=====Question 1.3=========')
    ans = matrix_mult(M, a, b)
    print(ans)

    # Question 1.4
    print('=====Question 1.4=========')
    print(get_singular_values(M, 1))
    print(get_singular_values(M, 2))

    # Question 1.5
    print('=====Question 1.5=========')
    M = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    val, vec = get_eigen_values_and_vectors(M[:, :3], 1)
    print("Values = \n", val)
    print("Vectors = \n", vec)
    val, vec = get_eigen_values_and_vectors(M[:, :3], 2)
    print("Values = \n", val)
    print("Vectors = \n", vec)


    print()
    print('==========Part 2: Image Manipulation======================')
    # Question 2.1
    print('=====Question 2.1 =========')

    image1_path = './image1.jpg'
    image2_path = './image2.jpg'

    image1 = load(image1_path)
    image2 = load(image2_path)

    display(image1)
    display(image2)

    print('=====Question 2.2 =========')
    # Question 2.2
    new_image = change_value(image1)
    display(new_image)

    print('=====Question 2.3 =========')
    # Question 2.3
    grey_image = convert_to_grey_scale(image1)
    display(grey_image)

    print('=====Question 2.4 =========')
    # Question 2.4
    without_red = rgb_decomposition(image1, 'R')
    without_blue = rgb_decomposition(image1, 'B')
    without_green = rgb_decomposition(image1, 'G')

    display(without_red)
    display(without_blue)
    display(without_green)

    print('=====Question 2.5 =========')
    # Question 2.5
    image_l = lab_decomposition(image1, 'L')
    image_a = lab_decomposition(image1, 'A')
    image_b = lab_decomposition(image1, 'B')

    display(image_l)
    display(image_a)
    display(image_b)


    print('=====Question 2.6 =========')
    # Question 2.6
    image_h = hsv_decomposition(image1, 'H')
    image_s = hsv_decomposition(image1, 'S')
    image_v = hsv_decomposition(image1, 'V')

    display(image_h)
    display(image_s)
    display(image_v)


    print('=====Question 2.7 =========')
    # Question 2.7
    image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
    display(image_mixed)