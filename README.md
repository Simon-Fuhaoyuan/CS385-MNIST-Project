# CS385 MNIST Project

## Introduction

This is about the first project in SJTU CS385, Machine Learning.

We do the task of classifying [Mnist](http://yann.lecun.com/exdb/mnist/) dataset, which includes hand-written digits. There are 60,000 training samples and 10,000 test samples.

## Traditional Methods

The following are the traditional (not CNN-based) methods that I have implemented personally. Note that the Support Vector Machine method is implemented using [sklearn.svm](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) package.

* Logistic Regression
* Linear Discriminant Analysis
* Support Vector Machine
* Logistic Regression with Ridge Loss
* Logistic Regression with Lasso Loss
* Kernel-based Logistic Regression with Ridge Loss
* Kernel-based Logistic Regression with Lasso Loss

## CNN Architecture

The following are CNN architectures that I have implemented using [pytorch](https://pytorch.org) package.

* VGG11
* NaiveNet (which consists of 2 conv layers and two fc layers)

Keep updating...
