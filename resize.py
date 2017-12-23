from scipy.misc import imread, imresize, imsave

x = imread("lcat.png")
x = imresize(x, (32, 32, 4))
imsave('cat.png', x)