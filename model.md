# Keras CIFAR

**Building image recognition with Keras and CIFAR**

#### Download CIFAR (15s):   
`tensorflow.python.keras.datasets.cifar10.load_data()`

#### MLP 0.3929 (15s):   
Dense 4 - Flatten - Dense 10 - Softmax

MLP Analysis:

* 4 input units seems optimial
* extra layer makes no significant improvement.
* No significant difference between optimiser `sgd` and `adam`.
* 2 epochs makes no significant improvement
* batch size 16 seems optimal. > faster but less accurate

#### Simple CNN 0.6026 (21s):   
Dense 4 - Flatten - Dense 10 - Softmax

Simple CNN Analysis:

* `adam` optimiser improves accuracy over `sgd`

#### Notes

*Timings based on colab with GPU runtime.*