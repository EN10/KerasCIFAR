# CIFAR Keras

Built upon [EN10 NumPy CIFAR](https://github.com/EN10/CIFAR)

The CIFAR-10 dataset consists of 60000 32x32 colour images in [10 classes](https://github.com/EN10/KerasCIFAR#classes), with 6000 images per class.  
There are 50000 training images and 10000 test images.  
The dataset is divided into five training batches and one test batch, each with 10000 images.

## Install

    sudo pip install -U pip
    sudo pip install tensorflow 
    sudo pip install h5py pillow 

[imsave](https://github.com/EN10/CIFAR/blob/master/README.md#compatability)



## Train

[train.py](https://github.com/EN10/KerasCIFAR/blob/master/train.py) based on 
[Keras Example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py) and 
[Keras in 30 seconds](https://keras.io/#getting-started-30-seconds-to-keras)   

    python train.py

See Also:
[MNIST CNN](https://github.com/EN10/KerasMNIST/blob/master/cnn.py)
    
## Predict

[predict.py](https://github.com/EN10/KerasCIFAR/blob/master/predict.py) based on 
[TFKpredict.py](https://github.com/EN10/KerasMNIST/blob/master/TFKpredict.py)

    python predict.py

## Utils

*   [resize.py](https://github.com/EN10/KerasCIFAR/blob/master/utils/resize.py) change image to 32 x 32 png 
*   [save.py](https://github.com/EN10/KerasCIFAR/blob/master/utils/save.py) save image from dataset to png
*   [load_cifar.py](https://github.com/EN10/KerasCIFAR/blob/master/utils/load_cifar.py) see below.

### Load CIFAR10 dataset
Keras Example [cifar10.py](https://github.com/keras-team/keras/blob/master/keras/datasets/cifar10.py) 
downloaded to `~/.keras` 341MB = 163MB tar.gz + 178MB extracted 

My Version [load_cifar.py](https://github.com/EN10/KerasCIFAR/blob/master/utils/load_cifar.py) based on  [NumPy load_cifar.py](https://github.com/EN10/CIFAR/blob/master/load_cifar.py):
* download and extract CIFAR.
* load batch into array
* test train split
* load images of classid

### Performance:

    wget https://github.com/EN10/BuildTF/raw/771df48529285c69ef760327121e996750b3916e/tensorflow-1.4.0-cp27-none-linux_x86_64.whl    
    sudo pip install --ignore-installed --upgrade tensorflow-1.4.0-cp27-none-linux_x86_64.whl

[FloydHub](https://github.com/EN10/FloydHub)

### Classes:

0 : airplane  
1 : automobile  
2 : bird  
3 : cat  
4 : deer  
5 : dog  
6 : frog  
7 : horse  
8 : ship  
9 : truck 

### Save != Open Bug

**DONT USE JPG!  USE PNG.**  
JPG Save array != JPG Open array  

check array shape:

    print x.shape

array to file for debugging:

    x.tofile('array.txt', "\n")