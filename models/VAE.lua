local nn = require 'nn'
require 'dpnn'

local Model = {
  zSize = 2,
  epsilonStd = 0.01 -- Epsilon (noise) standard deviation
}

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, 128))
  self.encoder:add(nn.ReLU(true))
  -- Create Z parameter layer
  local zLayer = nn.ConcatTable()
  zLayer:add(nn.Linear(128, self.zSize)) -- Mean of Z
  zLayer:add(nn.Linear(128, self.zSize)) -- Log standard deviation of Z
  -- Add Z parameter layer
  self.encoder:add(zLayer)

  -- Create epsilon sample module
  local epsilonModule = nn.Sequential()
  epsilonModule:add(nn.MulConstant(0)) -- Zero out input (do not do inplace)
  epsilonModule:add(nn.WhiteNoise(0, self.epsilonStd)) -- Sample noise

  -- Create standard deviation + epsilon module
  local noiseModule = nn.Sequential()
  local noiseModuleInternal = nn.ConcatTable()
  noiseModuleInternal:add(nn.Exp()) -- Exponentiate log standard deviations
  noiseModuleInternal:add(epsilonModule) -- Sample noise
  noiseModule:add(noiseModuleInternal)
  noiseModule:add(nn.CMulTable())

  -- Create sampler
  local sampler = nn.Sequential()
  local samplerInternal = nn.ParallelTable()
  samplerInternal:add(nn.Identity()) -- Pass through means
  samplerInternal:add(noiseModule) -- Create noise
  sampler:add(samplerInternal)
  sampler:add(nn.CAddTable())

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(self.zSize, 128))
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
