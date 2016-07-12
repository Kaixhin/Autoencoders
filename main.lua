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
cmd:option('-model', 'AE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AdvAE')
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

-- Create adversary (if needed)
local adversary
if opt.model == 'AdvAE' then
  Model:createAdversary()
  adversary = Model.adversary
  if cuda then
    adversary:cuda()
    -- Use cuDNN if available
    if hasCudnn then
      cudnn.convert(adversary, cudnn)
    end
  end
end

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()
local thetaAdv, gradThetaAdv
if opt.model == 'AdvAE' then
  thetaAdv, gradThetaAdv = adversary:getParameters()
end

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
  if opt.model == 'AdvAE' then
    gradThetaAdv:zero()
  end

  -- Reconstruction phase
  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

  -- Regularization phase
  if opt.model == 'VAE' then
    local encoder = Model.encoder
    -- Optimize KL-Divergence between encoder output and prior N(0, 1)
    local q = encoder.output
    local std = torch.exp(q[2])
    local KLLoss = -0.5 * torch.mean(1 + q[2] - torch.pow(q[2], 2) - std)
    loss = loss + KLLoss
    local gradKLLoss = {q[1], 0.5*(std - 1)}
    encoder:backward(x, gradKLLoss)
  elseif opt.model == 'AdvAE' then
    local encoder = Model.encoder
    local real = torch.Tensor(opt.batchSize, Model.zSize):normal(0, 1):typeAs(XTrain)
    local YReal = torch.ones(opt.batchSize):typeAs(XTrain)
    local YFake = torch.zeros(opt.batchSize):typeAs(XTrain)

    -- Train adversary on real sample ~ N(0, 1)
    local pred = adversary:forward(real)
    local lossReal = criterion:forward(pred, YReal)
    local gradLossReal = criterion:backward(pred, YReal)
    adversary:backward(real, gradLossReal)

    -- Train adversary on fake sample ~ encoder
    pred = adversary:forward(encoder.output)
    local lossFake = criterion:forward(pred, YFake)
    advLoss = lossReal + lossFake
    local gradLossFake = criterion:backward(pred, YFake)
    local gradFake = adversary:backward(encoder.output, gradLossFake)

    -- Minimax on fake sample
    local lossMinimax = criterion:forward(adversary.output, YReal)
    loss = loss + lossMinimax
    local gradLossMinimax = criterion:backward(adversary.output, YReal)
    local gradMinimax = adversary:updateGradInput(encoder.output, gradLossMinimax) -- Do not calculate grad wrt adversary parameters
    encoder:backward(x, gradMinimax)
  end

  return loss, gradTheta
end

local advFeval = function(params)
  if thetaAdv ~= params then
    thetaAdv:copy(params)
  end

  return advLoss, gradThetaAdv
end

-- Train
print('Training')
autoencoder:training()
local optimParams = {learningRate = opt.learningRate}
local advOptimParams = {learningRate = opt.learningRate}
local __, loss
local losses, advLosses = {}, {}

for epoch = 1, opt.epochs do
  print('Epoch ' .. epoch .. '/' .. opt.epochs)
  for n = 1, N, opt.batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, opt.batchSize)

    -- Optimise
    __, loss = optim.adam(feval, theta, optimParams)
    losses[#losses + 1] = loss[1]

    -- Train adversary
    if opt.model == 'AdvAE' then
      __, loss = optim.adam(advFeval, thetaAdv, advOptimParams)     
      advLosses[#advLosses + 1] = loss[1]
    end
  end
end

-- Plot training curve(s)
local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
if opt.model == 'AdvAE' then
  plots[#plots + 1] = {'Adversary', torch.linspace(1, #advLosses, #advLosses), torch.Tensor(advLosses), '-'}
end
gnuplot.pngfigure('Training.png')
gnuplot.plot(table.unpack(plots))
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

-- Plot samples
if opt.model == 'VAE' or opt.model == 'AdvAE' then
  local decoder = Model.decoder
  local n = 15
  local height, width = XTest:size(2), XTest:size(3)
  local samples = torch.Tensor(n * height, n * width):typeAs(XTest)
  local std = 1

  -- Sample n points within [-14, 14] standard deviations of N(0, 1)
  for i = 1, 15  do
    for j = 1, 15 do
      local sample = torch.Tensor({2 * i * std - 16 * std, 2 * j * std - 16 * std}):typeAs(XTest):view(1, 2) -- Minibatch of 1 for batch normalisation
      samples[{{(i-1) * height + 1, i * height}, {(j-1) * width + 1, j * height}}] = decoder:forward(sample)
    end
  end
  image.save('Samples.png', samples)
end
