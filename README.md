# CIFAR Keras

Based on [EN10 NumPy CIFAR](https://github.com/EN10/CIFAR)

The CIFAR-10 dataset consists of 60000 32x32 colour images in [10 classes](https://github.com/EN10/KerasCIFAR#classes), with 6000 images per class.  
There are 50000 training images and 10000 test images.  
The dataset is divided into five training batches and one test batch, each with 10000 images.

[Keras load_cifar.py](https://github.com/EN10/KerasCIFAR/blob/master/load_cifar.py) based on  [NumPy load_cifar.py](https://github.com/EN10/CIFAR/blob/master/load_cifar.py):
* download and extract CIFAR.
* load batch into array
* test train split
* load images of classid

[keras.py](https://github.com/EN10/KerasCIFAR/blob/master/keras.py) based on [Keras in 30 seconds](https://keras.io/#getting-started-30-seconds-to-keras)

## Install

    sudo pip install -U pip
    sudo pip install tensorflow 
    sudo pip install h5py pillow 
    
## Train

    python keras.py

## Predict  (NOT WORKING!)
[predict.py](https://github.com/EN10/KerasCIFAR/blob/master/predict.py) based on 
[KerasInception.py](https://github.com/EN10/KerasInception/blob/master/KerasInception.py) and 
[TFKpredict.py](https://github.com/EN10/KerasMNIST/blob/master/TFKpredict.py)

### Performance:

    wget https://github.com/EN10/BuildTF/raw/771df48529285c69ef760327121e996750b3916e/tensorflow-1.4.0-cp27-none-linux_x86_64.whl    
    sudo pip install --ignore-installed --upgrade tensorflow-1.4.0-cp27-none-linux_x86_64.whl

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