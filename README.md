Autoencoders
============

This repository is a Torch version of [Building Autoencoders in Keras](http://blog.keras.io/building-autoencoders-in-keras.html), but only containing code for reference - please refer to the original blog post for an explanation of autoencoders. Training hyperparameters have not been adjusted. The following models are implemented:

- **AE**: Fully-connected autoencoder
- **SparseAE**: Sparse autoencoder
- **DeepAE**: Deep (fully-connected) autoencoder
- **ConvAE**: Convolutional autoencoder
- **UpconvAE**: Upconvolutional autoencoder - also known by [several other names](https://github.com/torch/nn/blob/master/doc/convolution.md#spatialfullconvolution) *(bonus)*
- **DenoisingAE**: Denoising (convolutional) autoencoder [[1, 2]](#references)
- **CAE**: Contractive autoencoder *(bonus)* [[3]](#references)
- **Seq2SeqAE**: Sequence-to-sequence autoencoder
- **VAE**: Variational autoencoder [[4, 5]](#references)
- **CatVAE**: Categorical variational autoencoder *(bonus)* [[6, 7]](#references)
- **AAE**: Adversarial autoencoder *(bonus)* [[8]](#references)
- **WTA-AE**: Winner-take-all autoencoder *(bonus)* [[9]](#references)

Different models can be chosen using `th main.lua -model <modelName>`.

The *denoising* criterion can be used to replace the standard (autoencoder) *reconstruction* criterion by using the denoising flag. For example, a denoising AAE (DAAE) [[10]](#references) can be set up using `th main.lua -model AAE -denoising`. The corruption process is additive Gaussian noise *~ N(0, 0.5)*.

MCMC sampling [[10]](#references) can be used for VAEs, CatVAEs and AAEs with `th main.lua -model <modelName> -mcmc <steps>`. To see the effects of MCMC sampling with this simple setup it is best to choose a large standard deviation, e.g. `-sampleStd 5`, for the Gaussian distribution to draw the initial samples from.

Requirements
------------

The following luarocks packages are required:

- mnist
- dpnn (for DenoisingAE)
- rnn (for Seq2SeqAE)


Citation
--------

If you find this library useful and would like to cite it, the following would be appropriate:

```
@misc{Autoencoders,
  author = {Arulkumaran, Kai},
  title = {Kaixhin/Autoencoders},
  url = {https://github.com/Kaixhin/Autoencoders},
  year = {2016}
}
```

References
----------
[1] Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008, July). Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th international conference on Machine learning* (pp. 1096-1103). ACM.  
[2] Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P. A. (2010). Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *Journal of Machine Learning Research, 11*(Dec), 3371-3408.  
[3] Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). Contractive auto-encoders: Explicit invariance during feature extraction. In *Proceedings of the 28th international conference on machine learning (ICML-11)* (pp. 833-840).  
[4] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.  
[5] Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In *Proceedings of The 31st International Conference on Machine Learning* (pp. 1278-1286).  
[6] Jang, E., Gu, S., & Poole, B. (2016). Categorical Reparameterization with Gumbel-Softmax. *arXiv preprint arXiv:1611.01144*.  
[7] Maddison, C. J., Mnih, A., & Teh, Y. W. (2016). The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. *arXiv preprint arXiv:1611.00712*.  
[8] Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. *arXiv preprint arXiv:1511.05644*.  
[9] Makhzani, A., & Frey, B. J. (2015). Winner-take-all autoencoders. In *Advances in Neural Information Processing Systems* (pp. 2791-2799).  
[10] Arulkumaran, K., Creswell, A., & Bharath, A. A. (2016). Improving Sampling from Generative Autoencoders with Markov Chains. *arXiv preprint arXiv:1610.09296*.  
