local nn = require 'nn'

local Model = {}

function Model:createAutoencoder(X)
  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, 1, X:size(2), X:size(3)))
  self.encoder:add(nn.SpatialConvolution(1, 16, 3, 3, 1, 1, 1, 1))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
  self.encoder:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
  self.encoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.SpatialUpSamplingNearest(2))
  self.decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.SpatialUpSamplingNearest(2))
  self.decoder:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 0, 0))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.SpatialUpSamplingNearest(2))
  self.decoder:add(nn.SpatialConvolution(16, 1, 3, 3, 1, 1, 1, 1))
  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model
