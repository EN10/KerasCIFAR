from scipy.misc import imread, imresize, imsave
import sys

x = imread(sys.argv[1])         # filename as arg
x = imresize(x, (32, 32, 4))
imsave('cat.png', x)