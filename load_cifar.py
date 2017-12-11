import os               #   for wget and tar
import cPickle          #   load files
import numpy as np      #   arrays
import random           #   to select proportion of false data
import math             #   absolute value
random.seed(1)          #   set a seed so that the results are consistent

def download_extract():
    if not (os.path.isdir('cifar-10-batches-py') or os.path.isfile('cifar-10-python.tar.gz')):
        print 'no tar.gz or dir found so downloading tar.gz \n'
        os.system('wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    
    if not os.path.isdir('cifar-10-batches-py'):
        print 'no dir found so extracting tar.gz \n'
        os.system('tar -xvzf cifar-10-python.tar.gz')

def load_batch():
    path = 'cifar-10-batches-py/'
    file = 'data_batch_1'

    f = open(path+file, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    labels = dict['labels']
    imagearray = np.array(images)                           #   (10000, 3072)
    labelarray = np.array(labels)                           #   (10000,)
    
    return imagearray, labelarray

def train_test(imagearray, labelarray, train, test, classid):
    #   e.g. 200 train images from posistion 0 ++
    train_set_x, train_set_y    = n_images(imagearray, labelarray, classid, train, 0)
    #   e.g. 50 test images from posistion 9999 --
    test_set_x, test_set_y      = n_images(imagearray, labelarray, classid, test, -9999)

    return train_set_x, train_set_y, test_set_x, test_set_y
    
def n_images(imagearray, labelarray, classid, n, end):
    set_x = np.empty((n,3072))                              #   Create N images
    set_y = np.empty((1,n),dtype=np.int16)                  #   Create N labels

    i = j = 0
    while (j < n):                                          #   Select N images
        x = random.randint(0,1)
        
        if (labelarray[int(math.fabs(end+i))] == classid):  #   Selects Images from Class, end defines start or end
            set_x[j] = imagearray[int(math.fabs(end+i))]    #   end selects begining or end of array
            set_y[0,j] = 1                                  #   Set label
            j+=1
        elif (x % 2 == 0 and labelarray[i] != classid):     #   NOT Images from Class
            set_x[j] = imagearray[int(math.fabs(end+i))]
            set_y[0,j] = 0                                  #   Set label
            j+=1
        i+=1

    set_x = set_x.T                                         #   Reshape to (3072, N)
    set_x = set_x/255.                                      #   0-255 -> 0-1
    return set_x, set_y