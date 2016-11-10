Autoencoders
============

This repository is a Torch version of [Building Autoencoders in Keras](http://blog.keras.io/building-autoencoders-in-keras.html), but only containing code for reference - please refer to the original blog post for an explanation of autoencoders. Training hyperparameters have not been adjusted. The following models are implemented:

- **AE**: Fully-connected autoencoder
- **SparseAE**: Sparse autoencoder
- **DeepAE**: Deep (fully-connected) autoencoder
- **ConvAE**: Convolutional autoencoder
- **UpconvAE**: Upconvolutional autoencoder - also known by [several other names](https://github.com/torch/nn/blob/master/doc/convolution.md#spatialfullconvolution) *(bonus)*
- **DenoisingAE**: Denoising (convolutional) autoencoder
- **Seq2SeqAE**: Sequence-to-sequence autoencoder
- **VAE**: Variational autoencoder
- **CatVAE**: Categorical variational autoencoder *(bonus)*
- **AAE**: Adversarial autoencoder *(bonus)*

Different models can be chosen using `th main.lua -model <modelName>`.

MCMC sampling can be used for VAEs, CatVAEs and AAEs with `th main.lua -model <modelName> -mcmc <steps>`. To see the effects of MCMC sampling with this simple setup it is best to choose a large standard deviation, e.g. `-sampleStd 5`, for the Gaussian distribution to draw the initial samples from.

Requirements
------------

The following luarocks packages are required:

- mnist
- dpnn (for DenoisingAE)
- rnn (for Seq2SeqAE)
