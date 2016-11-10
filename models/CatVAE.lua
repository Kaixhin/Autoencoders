local nn = require 'nn'
require '../modules/Uniform'

local Model = {
  N = 30, -- Number of Gumbel-(Soft)Max distributions
  k = 10, -- Number of categories/classes
  tau = 1 -- Softmax temperature τ
}

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)

  -- Create encoder (inference model q, variational approximation for posterior p(z|x))
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, 128))
  self.encoder:add(nn.BatchNormalization(128))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(128, 64))
  self.encoder:add(nn.BatchNormalization(64))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(64, self.N * self.k)) -- Unnormalised log probabilities log(π)

  -- Create noise ε sample module
  local noiseModule = nn.Sequential()
  noiseModule:add(nn.Uniform(0, 1)) -- Sample from U(0, 1)
  -- Transform uniform sample to Gumbel sample
  noiseModule:add(nn.Log())
  noiseModule:add(nn.MulConstant(-1, true))
  noiseModule:add(nn.Log())
  noiseModule:add(nn.MulConstant(-1, true))

  -- Create sampler q(z) = G(z) = softmax((log(π) + ε)/τ) (reparametrization trick)
  local sampler = nn.Sequential()
  local samplerInternal = nn.ConcatTable()
  samplerInternal:add(nn.Identity()) -- Unnormalised log probabilities log(π)
  samplerInternal:add(noiseModule) -- Create noise ε
  sampler:add(samplerInternal)
  sampler:add(nn.CAddTable())
  self.temperature = nn.MulConstant(1 / self.tau, true) -- Temperature τ for softmax
  sampler:add(self.temperature)
  sampler:add(nn.View(-1, self.k)) -- Resize to work over k
  sampler:add(nn.SoftMax())
  sampler:add(nn.View(-1, self.N * self.k)) -- Resize back
  -- TODO: Hard sampling

  -- Create decoder (generative model p)
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(self.N * self.k, 64))
  self.decoder:add(nn.BatchNormalization(64))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(64, 128))
  self.decoder:add(nn.BatchNormalization(128))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(128, featureSize))
  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(sampler)
  self.autoencoder:add(self.decoder)
end

return Model
