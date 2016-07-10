local mnist = require 'mnist'
local nn = require 'nn'
local optim = require 'optim'
local image = require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local learningRate = 0.01
local batchSize = 600
local hiddenSize = 128
local zSize = 2

-- Load MNIST data
local data = mnist.traindataset().data:float():div(255)
local N = data:size(1)
local size = data:size(2) * data:size(3)

-- Create encoder
local encoder = nn.Sequential()
encoder:add(nn.View(-1, size))
encoder:add(nn.Linear(size, hiddenSize))
encoder:add(nn.ReLU(true))
encoder:add(nn.Linear(hiddenSize, zSize))

-- Create decoder
local decoder = nn.Sequential()
decoder:add(nn.Linear(zSize, hiddenSize))
decoder:add(nn.ReLU(true))
decoder:add(nn.Linear(hiddenSize, size))
decoder:add(nn.Sigmoid(true))
decoder:add(nn.View(data:size(2), data:size(3)))

-- Create autoencoder
local ae = nn.Sequential()
ae:add(encoder)
ae:add(decoder)

-- Create adversary
local adversary = nn.Sequential()
adversary:add(nn.Linear(zSize, 1))
adversary:add(nn.Sigmoid(true))

-- Get params
local params, gradParams = ae:getParameters()
local advParams, gradAdvParams = adversary:getParameters()

-- Create loss
local crit = nn.BCECriterion()

-- Train
local batch, aeLoss
local advFeval = function(x)
  if advParams ~= x then
    advParams:copy(x)
  end
  -- Zero gradients
  gradParams:zero()
  gradAdvParams:zero()

  -- Reconstruction phase
  local xHat = ae:forward(batch)
  aeLoss = crit:forward(xHat, batch)
  local gradLoss = crit:backward(xHat, batch)
  ae:backward(batch, gradLoss)

  -- Regularization phase (real sample ~ N(0, 1))
  local real = torch.Tensor(batchSize, zSize):normal(0, 1)
  local pred = adversary:forward(real)
  local lossReal = crit:forward(pred, torch.ones(batchSize))
  local gradLossReal = crit:backward(pred, torch.ones(batchSize))
  adversary:backward(real, gradLossReal)
  -- Regularization phase (fake sample ~ encoder)
  pred = adversary:forward(encoder.output)
  local lossFake = crit:forward(pred, torch.zeros(batchSize))
  local gradLossFake = crit:backward(pred, torch.zeros(batchSize))
  local gradFake = adversary:backward(encoder.output, gradLossFake)

  return lossFake + lossReal, gradAdvParams
end

local feval = function(x)
  if params ~= x then
    params:copy(x)
  end

  -- Minimax on fake sample
  local lossMinimax = crit:forward(adversary.output, torch.ones(batchSize))
  local gradLossMinimax = crit:backward(adversary.output, torch.ones(batchSize))
  local gradMinimax = adversary:updateGradInput(encoder.output, gradLossMinimax) -- No need to calculate grad wrt net parameters
  encoder:backward(batch, gradMinimax)

  return aeLoss + lossMinimax, gradParams
end


local optimParams = {learningRate = learningRate}
local advOptimParams = {learningRate = learningRate}
for n = 1, N, batchSize do
  -- Get batch of data
  batch = data:narrow(1, n, batchSize)
  local __, loss = optim.adam(advFeval, advParams, advOptimParams)
  __, loss = optim.adam(feval, params, optimParams)
  print(loss[1])
end
