-- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'

-- Set up Torch
print('Setting up')
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)


-- Load MNIST data
local XTrain = mnist.traindataset().data:float():div(255) -- Normalise to [0, 1]
local XTest = mnist.testdataset().data:float():div(255)
local N = XTrain:size(1)

-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-model', 'AE', 'Model: AE|SparseAE|CAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AdvAE')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-epochs', 10, 'Training epochs')
local opt = cmd:parse(arg)
opt.batchSize = 600 -- Currently only set up for divisors of N

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
local optimParams = {learningRate = opt.learningRate}
local __, loss
local losses = {}

for epoch = 1, opt.epochs do
  print('Epoch ' .. epoch .. '/' .. opt.epochs)
  for n = 1, N, opt.batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, opt.batchSize)

    -- Optimise
    __, loss = optim.adam(feval, theta, optimParams)
    losses[#losses + 1] = loss[1]
  end
end

-- Plot training curve
gnuplot.pngfigure('Training.png')
gnuplot.plot('', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-')
gnuplot.ylabel('Loss')
gnuplot.xlabel('Batch #')
gnuplot.plotflush()

-- Test
print('Testing')
autoencoder:evaluate()
x = XTest:narrow(1, 1, 10)
local xHat = autoencoder:forward(x)

-- Plot reconstructions
image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))
