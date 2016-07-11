-- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available

-- Set up Torch
print('Setting up')
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
end

-- Load MNIST data
local XTrain = mnist.traindataset().data:float():div(255) -- Normalise to [0, 1]
local XTest = mnist.testdataset().data:float():div(255)
local N = XTrain:size(1)
if cuda then
  XTrain = XTrain:cuda()
  XTest = XTest:cuda()
end

-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-model', 'AE', 'Model: AE|SparseAE|CAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AdvAE')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-epochs', 10, 'Training epochs')
local opt = cmd:parse(arg)
opt.batchSize = 150 -- Currently only set up for divisors of N

-- Create model
local Model = require ('models/' .. opt.model)
Model:createAutoencoder(XTrain)
local autoencoder = Model.autoencoder
if cuda then
  autoencoder:cuda()
  -- Use cuDNN if available
  if hasCudnn then
    cudnn.convert(autoencoder, cudnn)
  end
end

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()

-- Create loss
local criterion = nn.BCECriterion()
if cuda then
  criterion:cuda()
end

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

  if opt.model == 'VAE' then
    local encoder = Model.encoder
    -- Optimize KL-Divergence between encoder output and prior N(0, 1)
    local q = encoder.output
    local std = torch.exp(q[2])
    local KLLoss = -0.5 * torch.mean(1 + q[2] - torch.pow(q[2], 2) - std)
    loss = loss + KLLoss
    local gradKLLoss = {q[1], 0.5*(std - 1)}
    encoder:backward(x, gradKLLoss)
  end

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
x = XTest:narrow(1, 1, 10)
local xHat
if opt.model == 'DenoisingAE' then
  xHat = autoencoder:forward(x)

  -- Extract noised version from denoising AE
  local xNoise = autoencoder:findModules('nn.WhiteNoise')
  x = xNoise[1].output
else
  autoencoder:evaluate()
  xHat = autoencoder:forward(x)
end
-- Plot reconstructions
image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))
