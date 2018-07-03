import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io

from filters import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



def display(img, title='title'):
    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.title("Isn't he cute?")
    plt.show()



if __name__ == "__main__":
    # Open image as grayscale
    img = io.imread('dog.jpg', as_grey=True)
    # Show image
    # display(img, "Isn't he cute?")

    # Implement 1
    # conv_nested
    # kernel = np.array(
    #     [
    #         [1, 0, 1],
    #         [0, 0, 0],
    #         [1, 0, 1]
    #     ])
    #
    # # Create a test image: a white square in the middle
    # test_img = np.zeros((9, 9))
    # test_img[3:6, 3:6] = 1
    #
    # # Run your conv_nested function on the test image
    # test_output = conv_nested(test_img, kernel)
    #
    # # Build the expected output
    # expected_output = np.zeros((9, 9))
    # expected_output[2:7, 2:7] = 1
    # expected_output[4, 2:7] = 2
    # expected_output[2:7, 4] = 2
    # expected_output[4, 4] = 4
    #
    # # Plot the test image
    # plt.subplot(1, 3, 1)
    # plt.imshow(test_img)
    # plt.title('Test image')
    # plt.axis('off')
    #
    # # Plot your convolved image
    # plt.subplot(1, 3, 2)
    # plt.imshow(test_output)
    # plt.title('Convolution')
    # plt.axis('off')
    #
    # # Plot the exepected output
    # plt.subplot(1, 3, 3)
    # plt.imshow(expected_output)
    # plt.title('Exepected output')
    # plt.axis('off')
    # plt.show()
    #
    # # Test if the output matches expected output
    # assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."



    # implement 2
    kernel = np.array(
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])
    #
    # out = conv_nested(img, kernel)
    #
    # # Plot original image
    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # plt.title('Original')
    # plt.axis('off')
    #
    # out_m = out / 255
    #
    #
    # # Plot your convolved image
    # plt.subplot(2, 2, 2)
    # plt.imshow(out_m)
    # plt.title('Convolution')
    # plt.axis('off')
    #
    # # Plot your convolved image
    # plt.subplot(2, 2, 3)
    # plt.imshow(out)
    # plt.title('Convolution')
    # plt.axis('off')
    #
    # # Plot what you should get
    # solution_img = io.imread('convoluted_dog.jpg', as_grey=True)
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(solution_img)
    # plt.title('What you should get')
    # plt.axis('off')
    #
    # plt.show()


    # #Implement 3
    #
    # pad_width = 20  # width of the padding on the left and right
    # pad_height = 40  # height of the padding on the top and bottom
    #
    # padded_img = zero_pad(img, pad_height, pad_width)
    #
    # # Plot your padded dog
    # plt.subplot(1, 2, 1)
    # plt.imshow(padded_img)
    # plt.title('Padded dog')
    # plt.axis('off')
    #
    # # Plot what you should get
    # solution_img = io.imread('padded_dog.jpg', as_grey=True)
    # plt.subplot(1, 2, 2)
    # plt.imshow(solution_img)
    # plt.title('What you should get')
    # plt.axis('off')
    #
    # plt.show()


    # #Implement 4
    # t0 = time()
    # out_fast = conv_fast(img, kernel)
    # t1 = time()
    # out_nested = conv_nested(img, kernel)
    # t2 = time()
    #
    # # Compare the running time of the two implementations
    # print("conv_nested: took %f seconds." % (t2 - t1))
    # print("conv_fast: took %f seconds." % (t1 - t0))
    #
    # # Plot conv_nested output
    # plt.subplot(1,2,1)
    # plt.imshow(out_nested)
    # plt.title('conv_nested')
    # plt.axis('off')
    #
    # # Plot conv_fast output
    # plt.subplot(1,2,2)
    # plt.imshow(out_fast)
    # plt.title('conv_fast')
    # plt.axis('off')
    #
    # # Make sure that the two outputs are the same
    # if not (np.max(out_fast - out_nested) < 1e-10):
    #     print("Different outputs! Check your implementation.")


    # # Part 2
    # # Implement 2.1
    # Load template and image in grayscale
    img = io.imread('shelf.jpg')
    img_grey = io.imread('shelf.jpg', as_grey=True)
    temp = io.imread('template.jpg')
    temp_grey = io.imread('template.jpg', as_grey=True)
    #
    # # Perform cross-correlation between the image and the template
    # out = cross_correlation(img_grey, temp_grey)
    #
    # # Find the location with maximum similarity
    # y,x = (np.unravel_index(out.argmax(), out.shape))
    #
    # # Display product template
    # plt.figure(figsize=(25,20))
    # plt.subplot(3, 1, 1)
    # plt.imshow(temp)
    # plt.title('Template')
    # plt.axis('off')
    #
    # # Display cross-correlation output
    # plt.subplot(3, 1, 2)
    # plt.imshow(out)
    # plt.title('Cross-correlation (white means more correlated)')
    # plt.axis('off')
    #
    # # Display image
    # plt.subplot(3, 1, 3)
    # plt.imshow(img)
    # plt.title('Result (blue marker on the detected location)')
    # plt.axis('off')
    #
    # # Draw marker at detected location
    # plt.plot(x, y, 'bx', ms=40, mew=10)
    # plt.show()

    #Implement 2.2
    # Perform cross-correlation between the image and the template
    out = zero_mean_cross_correlation(img_grey, temp_grey)

    # Find the location with maximum similarity
    y, x = (np.unravel_index(out.argmax(), out.shape))

    # Display product template
    plt.figure(figsize=(30, 20))
    plt.subplot(3, 1, 1)
    plt.imshow(temp)
    plt.title('Template')
    plt.axis('off')

    # Display cross-correlation output
    plt.subplot(3, 1, 2)
    plt.imshow(out)
    plt.title('Cross-correlation (white means more correlated)')
    plt.axis('off')

    # Display image
    plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.title('Result (blue marker on the detected location)')
    plt.axis('off')

    # Draw marker at detcted location
    plt.plot(x, y, 'bx', ms=40, mew=10)
    plt.show()



    # Implement 2.3
    # Perform normalized cross-correlation between the image and the template

    # # Load image
    # img = io.imread('shelf_dark.jpg')
    # temp = io.imread('template.jpg')
    # img_grey = io.imread('shelf_dark.jpg', as_grey=True)
    # temp_grey = io.imread('template.jpg', as_grey=True)
    # out = normalized_cross_correlation(img_grey, temp_grey)
    #
    # # Find the location with maximum similarity
    # y, x = (np.unravel_index(out.argmax(), out.shape))
    #
    # # Display image
    # plt.imshow(img)
    # plt.title('Result (red marker on the detected location)')
    # plt.axis('off')
    #
    # # Draw marker at detcted location
    # plt.plot(x, y, 'rx', ms=25, mew=5)
    # plt.show()