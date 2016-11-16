local nn = require 'nn'
require '../modules/BatchTopK'

local Model = {
  k = 0.05, -- Sparsity
  features = 1000 -- Number of features
}

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, self.features))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.BatchTopK(self.k, true)) -- Extract k% top activations during training

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(self.features, featureSize))
  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model
