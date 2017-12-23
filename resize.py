from scipy.misc import imresize, imsave
from PIL import Image
import sys

x = Image.open(sys.argv[1])
x = x.convert("RGB")
x = imresize(x, (32, 32, 3))
imsave('image.png', x)