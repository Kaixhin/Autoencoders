local mnist = require 'mnist'
local nn = require 'nn'
local optim = require 'optim'
local image = require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local learningRate = 0.01
local batchSize = 600
local noiseFactor = 0.5

-- Load MNIST data
local data = mnist.traindataset().data:float():div(255)
local N = data:size(1)
local size = data:size(2) * data:size(3)

-- Create encoder
local encoder = nn.Sequential()
encoder:add(nn.View(-1, 1, data:size(2), data:size(3)))
encoder:add(nn.SpatialConvolution(1, 32, 3, 3, 1, 1, 1, 1))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
encoder:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- TODO: Find out padding - output is (8, 7, 7) with (1, 1) padding

-- Create decoder
local decoder = nn.Sequential()
decoder:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolution(32, 1, 3, 3, 1, 1, 1, 1))
decoder:add(nn.Sigmoid(true))

-- Create autoencoder
local ae = nn.Sequential()
ae:add(encoder)
ae:add(decoder)

-- Get params
local params, gradParams = ae:getParameters()

-- Create loss
local crit = nn.BCECriterion()

-- Train
local batch, xHat
local feval = function(x)
  if params ~= x then
    params:copy(x)
  end
  -- Zero gradients
  gradParams:zero()

  -- Create noisy inputs
  local batchNoisy = batch:clone()
  batchNoisy:add(noiseFactor, torch.Tensor(batch:size()):normal(0, 1))
  batchNoisy:clamp(0, 1)

  xHat = ae:forward(batchNoisy)
  local loss = crit:forward(xHat, batch)
  local gradLoss = crit:backward(xHat, batch)
  ae:backward(batchNoisy, gradLoss)

  return loss, gradParams
end

local optimParams = {learningRate = learningRate}
for n = 1, N, batchSize do
  -- Get batch of data
  batch = data:narrow(1, n, batchSize)
  local __, loss = optim.adam(feval, params, optimParams)
  print(loss[1])
end
