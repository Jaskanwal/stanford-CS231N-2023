# CS 231n: Deep Learning for Computer Vision
This repo documents my solutions for  assignments of the Stanford
[CS 231n (Deep Learning for Computer Vision - 2023)](http://cs231n.stanford.edu/) class. Setup details and the assignment descriptions can be found on the [course website](https://cs231n.github.io/). Video lectures covering most of the course content can be found here:
- [Stanford Convolutional Neural Networks for Visual Recognition (Spring 2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): Version of the class offered in 2017
- [Michigan EECS 498-007 Deep Learning for Computer Visison](https://www.youtube.com/watch?v=dJYGatp4SvA&t=1s): Covers some additional topics covered after 2017.

A brief summary of key concepts covered in different assignments is summarized below.

## Assignment 1: [Image Classification, kNN, SVM, Softmax, Fully Connected Neural Network](assignment1)
This assignment provides experience on multi-class image classification using kNN, SVM, Softmax and Fully connected neural networks using [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Losses and gradients for different models are derivered and are implemented using Numpy in a vectorized manner. Following are the key goals of this assignment:
- Understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages).
- Understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- Develop proficiency in writing efficient **vectorized** code with numpy.
- Implement and apply a k-Nearest Neighbor (**kNN**) classifier.
- Implement and apply a Multiclass Support Vector Machine (**SVM**) classifier.
- Implement and apply a **Softmax** classifier.
- Implement and apply a **Two layer neural network** classifier.
- Understand the differences and tradeoffs between these classifiers.
- Get a basic understanding of performance improvements from using **higher-level representations** as opposed to raw pixels, e.g. color histograms, Histogram of Oriented Gradient (HOG) features, etc.

## Assignment 2: [Fully Connected and Convolutional Nets, Batch Normalization, Dropout, Pytorch & Network Visualization](assignment2)
Following are the key goals of this assignment:
- Understand **Neural Networks** and how they are arranged in layered architectures.
- Understand and be able to implement (vectorized) **backpropagation** for fully connected, batch norm, layer norm, dropout, 
    among other, commonly used layers.
- Implement various **update rules** (SGD, RMSProp, Adam, etc.) used to optimize Neural Networks.
- Implement **Batch Normalization** and **Layer Normalization** for training deep networks.
- Implement **Dropout** to regularize networks.
- Understand the architecture of **Convolutional Neural Networks** and get practice with training them.
- Gain experience with a major deep learning framework, **PyTorch**.

The [linked blog post](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) 
was found to be very useful for understanding the backpropogarion of Batch norm and also motivated the implementation 
of layer norm and spatial group norm layers in this assignment.