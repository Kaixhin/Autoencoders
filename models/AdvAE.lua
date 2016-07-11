local nn = require 'nn'

local Model = {
  zSize = 2
}

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, 128))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(128, self.zSize))

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
  self.autoencoder:add(self.decoder)
end

function Model:createAdversary()
  -- Create adversary
  self.adversary = nn.Sequential()
  self.adversary:add(nn.Linear(self.zSize, 1))
  self.adversary:add(nn.Sigmoid(true))
end

return Model
