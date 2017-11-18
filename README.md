Deep-Convolutional-Generative-Adversarial-Net-DCGAN-
===
A tensorflow implementation of ![Deep Convolutional Generative Adversarial Net (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf)
<br>
![DCGAN](https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/net_structure.png)
<br>
I train and evaluate my model on CIFAR-10 Dataset. However, you can perform it on your own datasets
<br>
<br>
Requirements
---
  python 2.7<br>
  tensorflow<br>
  cPickle<br>
  numpy<br>
  Scipy<br>
  <br>
  <br>
Usage
---
<br>
# Clone the repository
$ git clone https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-.git
<br>
# Download CIFAR-10 Dataset
download data from ![here](https://www.cs.toronto.edu/~kriz/cifar.html)
<br>
<br>
My results
---
<br>
# loss curve during training (for 20 epochs)
![loss_curve](https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/loss_curve.png)
<br>
# sampled images from trained model
![sample_1](https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/sample_1.png)
<br>
![sample_2](https://github.com/Szy-Young/Deep-Convolutional-Generative-Adversarial-Net-DCGAN-/blob/master/sample_2.png)
<br>
<br>
Acknowledgements
---
Alec Radford, Luke Metz, Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR 2016.
