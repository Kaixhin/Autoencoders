-- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
require 'dpnn'

-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-cpu', false, 'CPU only (useful if GPU memory is too low)')
cmd:option('-model', 'AE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|CatVAE|AAE|WTA-AE')
cmd:option('-learningRate', 0.0001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 20, 'Training epochs')
cmd:option('-denoising', false, 'Use denoising criterion')
cmd:option('-mcmc', 0, 'MCMC samples')
cmd:option('-sampleStd', 1, 'Standard deviation of Gaussian distribution to sample from')
local opt = cmd:parse(arg)
opt.batchSize = 60 -- Currently only set up for divisors of N
if opt.cpu then
  cuda = false
end
if opt.model == 'DenoisingAE' then
  opt.denoising = false -- Disable "extra" denoising
end

-- Set up Torch
print('Setting up')
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

-- Create model
local Model = require ('models/' .. opt.model)
Model:createAutoencoder(XTrain)
if opt.denoising then
  Model.autoencoder:insert(nn.WhiteNoise(0, 0.5), 1) -- Add noise during training
end
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
if opt.model == 'AAE' then
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
if opt.model == 'AAE' then
  thetaAdv, gradThetaAdv = adversary:getParameters()
end

-- Create loss
local criterion = nn.BCECriterion()
local softmax = nn.SoftMax() -- Softmax for CatVAE KL divergence
if cuda then
  criterion:cuda()
  softmax:cuda()
end

-- Create optimiser function evaluation
local x -- Minibatch
local feval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()
  if opt.model == 'AAE' then
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
  if opt.model == 'Seq2SeqAE' then
    -- Clamp RNN gradients to prevent exploding gradients
    gradTheta:clamp(-10, 10)
  elseif opt.model == 'VAE' then
    -- Optimise Gaussian KL divergence between inference model and prior: DKL[q(z|x)||N(0, σI)] = log(σ2/σ1) + ((σ1^2 - σ2^2) + (μ1 - μ2)^2) / 2σ2^2
    local nElements = xHat:nElement()
    local mean, logVar = table.unpack(Model.encoder.output)
    local var = torch.exp(logVar)
    local KLLoss = 0.5 * torch.sum(torch.pow(mean, 2) + var - logVar - 1)
    KLLoss = KLLoss / nElements -- Normalise loss (same normalisation as BCECriterion)
    loss = loss + KLLoss
    local gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}  -- Normalise gradient of loss (same normalisation as BCECriterion)
    Model.encoder:backward(x, gradKLLoss)
  elseif opt.model == 'CatVAE' then
    -- Optimise KL divergence between inference model and prior
    local nElements = xHat:nElement()
    local z = softmax:forward(Model.encoder.output:view(-1, Model.k)) + 1e-9 -- Improve numerical stability
    local logZ = torch.log(z)
    local KLLoss = torch.sum(z:cmul(logZ - math.log(1 / Model.k)))
    KLLoss = KLLoss / nElements -- Normalise loss (same normalisation as BCECriterion)
    local gradKLLoss = softmax:backward(Model.encoder.output:view(-1, Model.k), math.log(1 / Model.k) - logZ - 1):view(-1, Model.N * Model.k)
    gradKLLoss = gradKLLoss / nElements -- Normalise gradient of loss (same normalisation as BCECriterion)
    loss = loss + KLLoss
    Model.encoder:backward(x, gradKLLoss)
    
    -- Anneal temperature τ
    Model.tau = math.max(Model.tau - 0.0002, 0.5)
    Model.temperature.constant_scalar = 1 / Model.tau
  elseif opt.model == 'AAE' then
    local real = torch.Tensor(opt.batchSize, Model.zSize):normal(0, 1):typeAs(XTrain) -- Real samples ~ N(0, 1)
    local YReal = torch.ones(opt.batchSize):typeAs(XTrain) -- Labels for real samples
    local YFake = torch.zeros(opt.batchSize):typeAs(XTrain) -- Labels for generated samples

    -- Train adversary to maximise log probability of real samples: max_D log(D(x))
    local pred = adversary:forward(real)
    local realLoss = criterion:forward(pred, YReal)
    local gradRealLoss = criterion:backward(pred, YReal)
    adversary:backward(real, gradRealLoss)

    -- Train adversary to minimise log probability of fake samples: max_D log(1 - D(G(x)))
    pred = adversary:forward(Model.encoder.output)
    local fakeLoss = criterion:forward(pred, YFake)
    advLoss = realLoss + fakeLoss
    local gradFakeLoss = criterion:backward(pred, YFake)
    local gradFake = adversary:backward(Model.encoder.output, gradFakeLoss)

    -- Train encoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
    local minimaxLoss = criterion:forward(pred, YReal) -- Technically use max_G max_D log(D(G(x))) for same fixed point, stronger initial gradients
    loss = loss + minimaxLoss
    local gradMinimaxLoss = criterion:backward(pred, YReal)
    local gradMinimax = adversary:updateGradInput(Model.encoder.output, gradMinimaxLoss) -- Do not calculate gradient wrt adversary parameters
    Model.encoder:backward(x, gradMinimax)
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
    __, loss = optim[opt.optimiser](feval, theta, optimParams)
    losses[#losses + 1] = loss[1]

    -- Train adversary
    if opt.model == 'AAE' then
      __, loss = optim[opt.optimiser](advFeval, thetaAdv, advOptimParams)     
      advLosses[#advLosses + 1] = loss[1]
    end
  end

  -- Plot training curve(s)
  local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
  if opt.model == 'AAE' then
    plots[#plots + 1] = {'Adversary', torch.linspace(1, #advLosses, #advLosses), torch.Tensor(advLosses), '-'}
  end
  gnuplot.pngfigure('Training.png')
  gnuplot.plot(table.unpack(plots))
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Batch #')
  gnuplot.plotflush()

  -- Permute data
  XTrain = XTrain:index(1, torch.randperm(XTrain:size(1)):long())
end

-- Test
print('Testing')
x = XTest:narrow(1, 1, 10)
local xHat
if opt.model == 'DenoisingAE' then
  -- Normally this should be switched to evaluation mode, but this lets us extract the noised version
  xHat = autoencoder:forward(x)
  -- Extract noised version from denoising AE
  x = Model.noiser.output
else
  autoencoder:evaluate()
  xHat = autoencoder:forward(x)
end

-- Plot reconstructions
image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))

if opt.model == 'AE' or opt.model == 'SparseAE' or opt.model == 'WTA-AE' then
  -- Plot filters
  image.save('Weights.png', image.toDisplayTensor(Model.decoder:findModules('nn.Linear')[1].weight:view(x:size(3), x:size(2), Model.features):transpose(1, 3), 1, math.floor(math.sqrt(Model.features))))
end

if opt.model == 'VAE' or opt.model == 'AAE' then
  if opt.denoising then
    autoencoder:training() -- Retain corruption process
  end

  -- Plot interpolations
  local height, width = XTest:size(2), XTest:size(3)
  local interpolations = torch.Tensor(15 * height, 15 * width):typeAs(XTest)
  local step = 0.05 -- Use small steps in dense region of 2D Gaussian; TODO: Move to spherical interpolation?

  -- Sample 15 x 15 points
  for i = 1, 15  do
    for j = 1, 15 do
      local sample = torch.Tensor({2 * i * step - 16 * step, 2 * j * step - 16 * step}):typeAs(XTest):view(1, 2) -- Minibatch of 1 for batch normalisation
      interpolations[{{(i-1) * height + 1, i * height}, {(j-1) * width + 1, j * width}}] = Model.decoder:forward(sample)
    end
  end
  image.save('Interpolations.png', interpolations)

  -- Plot samples
  local output = Model.decoder:forward(torch.Tensor(15 * 15, 2):normal(0, opt.sampleStd):typeAs(XTest)):clone()
  
  -- Perform MCMC sampling
  for m = 0, opt.mcmc do
    -- Save samples
    if m == 0 then
      image.save('Samples.png', image.toDisplayTensor(Model.decoder.output, 0, 15))
    else
      image.save('Samples (MCMC step ' .. m .. ').png', image.toDisplayTensor(Model.decoder.output, 0, 15))
    end

    -- Forward again
    autoencoder:forward(output)
  end
elseif opt.model == 'CatVAE' then
  if opt.denoising then
    autoencoder:training() -- Retain corruption process
  end

  -- Plot "interpolations"
  local height, width = XTest:size(2), XTest:size(3)
  local interpolations = torch.Tensor(Model.N * height, Model.k * width):typeAs(XTest)
  
  for n = 1, Model.N do
    for k = 1, Model.k do
      local sample = torch.zeros(Model.N, Model.k):typeAs(XTest)
      sample[{{}, {1}}] = 1 -- Start with first dimension "set"
      sample[n] = 0 -- Zero out distribution
      sample[n][k] = 1 -- "Set" cluster
      interpolations[{{(n-1) * height + 1, n * height}, {(k-1) * width + 1, k * width}}] = Model.decoder:forward(sample:view(1, Model.N * Model.k)) -- Minibatch of 1 for batch normalisation
    end
  end
  image.save('Interpolations.png', interpolations)

  -- Plot samples
  local samples = torch.Tensor(15 * 15 * Model.N, Model.k):bernoulli(1 / Model.k):typeAs(XTest):view(15 * 15, Model.N * Model.k)
  local output = Model.decoder:forward(samples):clone()
  
  -- Perform MCMC sampling
  for m = 0, opt.mcmc do
    -- Save samples
    if m == 0 then
      image.save('Samples.png', image.toDisplayTensor(Model.decoder.output, 0, 15))
    else
      image.save('Samples (MCMC step ' .. m .. ').png', image.toDisplayTensor(Model.decoder.output, 0, 15))
    end

    -- Forward again
    autoencoder:forward(output)
  end
end
