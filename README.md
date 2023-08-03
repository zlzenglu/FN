# FN
This is the source code for paper "Feature Noise Boosts DNN Generalization under Label Noise".

This project contains codes to run the baseline method, early stopping method and the proposed method on MNIST, FMNIST and miniImageNet. Experiments on other methods and datasets can be implemented by a similar approach.
The ADNI dataset we used in the paper is not provided because it is private and unpublished data. Please understand that we have no authority to disclose it.
You should specify the configuration file to run the experiments. 
e.g. running "python train.py -c 'config_mnist.json' " to run the experiment with default settings: 80% label noise and Gaussian feature noise on MNIST.

>Requirements

python==3.7.13

pytorch==1.12.0

torchvison==0.12.0

numpy==1.21.2

mlclf==0.2.14
