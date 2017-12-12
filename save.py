from load_cifar import load_batch
import numpy as np

images , _ = load_batch()
images = np.reshape(images, (10000, 3, 32, 32))
img = np.transpose(images[26], (1, 2, 0))           #   from (3, 32, 32) to (32, 32, 3)

from PIL import Image
img = Image.fromarray(img, 'RGB')
img.save('cat.jpg')