local mnist = require 'mnist'
local nn = require 'nn'
local optim = require 'optim'
local image = require 'image'
require 'dpnn'

torch.setdefaulttensortype('torch.FloatTensor')

local learningRate = 0.01
local batchSize = 600
local hiddenSize = 128
local zSize = 2
local epsilonStd = 0.01 -- Epsilon (noise) standard deviation

-- Load MNIST data
local data = mnist.traindataset().data:float():div(255)
local N = data:size(1)
local size = data:size(2) * data:size(3)

-- Create encoder
local encoder = nn.Sequential()
encoder:add(nn.View(-1, size))
encoder:add(nn.Linear(size, hiddenSize))
encoder:add(nn.ReLU(true))
-- Create Z parameter layer
local zLayer = nn.ConcatTable()
zLayer:add(nn.Linear(hiddenSize, zSize)) -- Mean of Z
zLayer:add(nn.Linear(hiddenSize, zSize)) -- Log standard deviation of Z
-- Add Z parameter layer
encoder:add(zLayer)

-- Create epsilon sample module
local epsilonModule = nn.Sequential()
epsilonModule:add(nn.MulConstant(0)) -- Zero out input (do not do inplace)
epsilonModule:add(nn.WhiteNoise(0, epsilonStd)) -- Sample noise

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
local decoder = nn.Sequential()
decoder:add(nn.Linear(zSize, hiddenSize))
decoder:add(nn.ReLU(true))
decoder:add(nn.Linear(hiddenSize, size))
decoder:add(nn.Sigmoid(true))
decoder:add(nn.View(data:size(2), data:size(3)))

-- Create autoencoder
local ae = nn.Sequential()
ae:add(encoder)
ae:add(sampler)
ae:add(decoder)
-- Set to training mode for nn.WhiteNoise()
ae:training()

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

  -- Optimize KL-Divergence between encoder output and prior N(0, 1)
  local q = encoder.output
  local std = torch.exp(q[2])
  local KLLoss = -0.5 * torch.mean(1 + q[2] - torch.pow(q[2], 2) - std)
  loss = loss + KLLoss
  local gradKLLoss = {q[1], 0.5*(std - 1)}
  encoder:backward(batch, gradKLLoss)

  return loss, gradParams
end

local optimParams = {learningRate = learningRate}
for n = 1, N, batchSize do
  -- Get batch of data
  batch = data:narrow(1, n, batchSize)
  local __, loss = optim.adam(feval, params, optimParams)
  print(loss[1])
end
