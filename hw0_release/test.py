from skimage import io

im = io.imread('./image1.jpg')
print(im.shape)
io.imshow(im)
io.show()