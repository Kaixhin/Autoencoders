local nn = require 'nn'

local Model = {
  zSize = 2 --  -- Size of isotropic multivariate Gaussian Z
}

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)

  -- Create encoder (generator)
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, 128))
  self.encoder:add(nn.BatchNormalization(128))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(128, 64))
  self.encoder:add(nn.BatchNormalization(64))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(64, self.zSize)) -- Encoding distribution q(z|x) is a deterministic function of x
  -- Note that a Gaussian posterior (like VAE) or universal approximator posterior could be used, but deterministic q(z|x) works well

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(self.zSize, 64))
  self.decoder:add(nn.BatchNormalization(64))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(64, 128))
  self.decoder:add(nn.BatchNormalization(128))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(128, featureSize))
  self.decoder:add(nn.BatchNormalization(featureSize))
  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

function Model:createAdversary()
  -- Create adversary (discriminator)
  self.adversary = nn.Sequential()
  self.adversary:add(nn.Linear(self.zSize, 16))
  self.adversary:add(nn.BatchNormalization(16))
  self.adversary:add(nn.ReLU(true))
  self.adversary:add(nn.Linear(16, 1))
  self.adversary:add(nn.BatchNormalization(1))
  self.adversary:add(nn.Sigmoid(true))
end

return Model
