# Deep-Convolutional-Generative-Adversarial-Net-DCGAN-
A tensorflow implementation of ![Deep Convolutional Generative Adversarial Net (DCGAN)] (https://arxiv.org/pdf/1511.06434.pdf)

![DCGAN] (https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/net_structure.png)

I train and evaluate my model on CIFAR-10 Dataset. However, you can perform it on your own datasets

# Requirements
  python 2.7
  tensorflow
  cPickle
  numpy
  Scipy
  
# Usage

## Clone the repository
$ git clone https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-.git

## Download CIFAR-10 Dataset
download data from ![here] (https://www.cs.toronto.edu/~kriz/cifar.html)

## Train DCGAN 
$ cd ...(directory you place the repository)
$ python main.py --mode train

## Evaluate your trained model by sampling
$ python main.py --mode eval

# My results

## loss curve during training (for 20 epochs)
![loss_curve] (https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/loss_curve.png)

## sampled images from trained model
![sample_1] (https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/sample_1.png)

![sample_2] (https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/sample_2.png)

# Acknowledgements
Alec Radford, Luke Metz, Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR 2016.
