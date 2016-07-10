local mnist = require 'mnist'
local nn = require 'nn'
local optim = require 'optim'
local image = require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local learningRate = 0.01
local batchSize = 600
local codeLength = 32

-- Load MNIST data
local data = mnist.traindataset().data:float():div(255)
local N = data:size(1)
local size = data:size(2) * data:size(3)

-- Create encoder
local encoder = nn.Sequential()
encoder:add(nn.View(-1, size))
encoder:add(nn.Linear(size, codeLength))
encoder:add(nn.ReLU(true))

-- Create decoder
local decoder = nn.Sequential()
decoder:add(nn.Linear(codeLength, size))
decoder:add(nn.Sigmoid(true))
decoder:add(nn.View(data:size(2), data:size(3)))

-- Create autoencoder
local ae = nn.Sequential()
ae:add(encoder)
ae:add(decoder)

-- Get params
local params, gradParams = ae:getParameters()

-- Create loss
local crit = nn.BCECriterion()

-- Train
local batch
local feval = function(x)
  if params ~= x then
    params:copy(x)
  end
  -- Zero gradients
  gradParams:zero()

  local xHat = ae:forward(batch)
  local loss = crit:forward(xHat, batch)
  local gradLoss = crit:backward(xHat, batch)
  ae:backward(batch, gradLoss)

  return loss, gradParams
end

local optimParams = {learningRate = learningRate}
for n = 1, N, batchSize do
  -- Get batch of data
  batch = data:narrow(1, n, batchSize)
  local __, loss = optim.adam(feval, params, optimParams)
  print(loss[1])
end
