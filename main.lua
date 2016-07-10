-- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local image = require 'image'

-- Set up Torch
print('Setting up')
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

-- Shared hyperparameters
local learningRate = 0.001
local batchSize = 600
local epochs = 10

-- Load MNIST data
local XTrain = mnist.traindataset().data:float():div(255) -- Normalise to [0, 1]
local XTest = mnist.testdataset().data:float():div(255)
local N = XTrain:size(1)

-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-model', 'AE', 'Model: AE|SparseAE|CAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AdvAE')
local opt = cmd:parse(arg)

-- Create model
local Model = require ('models/' .. opt.model)
local autoencoder = Model:createAutoencoder(XTrain)

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()

-- Create loss
local criterion = nn.BCECriterion()

-- Create optimiser function evaluation
local x -- Minibatch
local feval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()

  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

  return loss, gradTheta
end

-- Train
print('Training')
autoencoder:training()
local optimParams = {learningRate = learningRate}
local __, loss

for epoch = 1, epochs do
  print('Epoch ' .. epoch .. '/' .. epochs)
  for n = 1, N, batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, batchSize)

    -- Optimise
    __, loss = optim.adam(feval, theta, optimParams)
  end
end

-- Test
print('Testing')
autoencoder:evaluate()
x = XTest:narrow(1, 1, 10)
local xHat = autoencoder:forward(x)

-- Plot reconstructions
image.save('reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))
