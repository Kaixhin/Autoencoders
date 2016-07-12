local nn = require 'nn'
require 'dpnn'

local Model = {
  zSize = 2, -- Size of multivariate Gaussian Z
  epsilonStd = 0.01 -- Noise ε standard deviation
}

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)

  -- Create encoder (inference model q, variational approximation for posterior p(z|x))
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, 128))
  self.encoder:add(nn.BatchNormalization(128))
  self.encoder:add(nn.ReLU(true))
  -- Create latent Z parameter layer
  local zLayer = nn.ConcatTable()
  zLayer:add(nn.Linear(128, self.zSize)) -- Mean μ of Z
  zLayer:add(nn.Linear(128, self.zSize)) -- Log standard deviation σ of Z (diagonal covariance)
  self.encoder:add(zLayer) -- Add Z parameter layer

  -- Create noise ε sample module
  local epsilonModule = nn.Sequential()
  epsilonModule:add(nn.MulConstant(0)) -- Zero out whatever input (do not do inplace)
  epsilonModule:add(nn.WhiteNoise(0, self.epsilonStd)) -- Generate noise ε

  -- Create σε module
  local noiseModule = nn.Sequential()
  local noiseModuleInternal = nn.ConcatTable()
  noiseModuleInternal:add(nn.Exp()) -- Exponentiate log standard deviations
  noiseModuleInternal:add(epsilonModule) -- Sample noise
  noiseModule:add(noiseModuleInternal)
  noiseModule:add(nn.CMulTable())

  -- Create sampler q(z) = N(z; μ, σI) = μ + σε (reparametrization trick)
  local sampler = nn.Sequential()
  local samplerInternal = nn.ParallelTable()
  samplerInternal:add(nn.Identity()) -- Pass through μ 
  samplerInternal:add(noiseModule) -- Create noise σ * ε
  sampler:add(samplerInternal)
  sampler:add(nn.CAddTable())

  -- Create decoder (generative model p)
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(self.zSize, 128))
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
