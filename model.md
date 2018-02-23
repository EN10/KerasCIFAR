# Keras CIFAR

**Building image recognition with Keras and CIFAR**

#### Download CIFAR (15s):   
`tensorflow.python.keras.datasets.cifar10.load_data()`

#### MLP 0.3929 (15s):   

Dense 4 - Flatten - Dense 10 - Softmax  

Some ideas from [mnist_mlp](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)  

[MLP](https://github.com/EN10/KerasCIFAR/blob/master/example/mlp.py) Analysis:

* 4 input units seems optimial
* extra layer makes no significant improvement.
* No significant difference between optimiser `sgd` and `adam`.
* 2 epochs makes no significant improvement
* batch size 16 seems optimal. > faster but less accurate

#### Simple CNN 0.6026 (21s):   
Conv2D 32, 3, 3 - ReLU - Conv2D 32, 3, 3 - ReLU - Flatten - Dense 10 - Softmax  

Based on [cifar10_cnn](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)

Simple [CNN](https://github.com/EN10/KerasCIFAR/blob/master/example/cnn.py) Analysis:

* 3 Conv2D layers makes no significant improvement 
* 32 filters seems to be optimal. 64 filters gives 0.6188 accuracy in 32s
* `adam` optimiser improves accuracy over `sgd` (0.4437)
* 2 epochs makes no significant improvement
* batch size 32 seems optimal. > faster but less accurate

#### Notes

*Timings based on colab with GPU runtime.*