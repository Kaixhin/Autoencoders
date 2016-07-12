local nn = require 'nn'
require 'rnn'

local Model = {
  cellSizes = {256, 256}, -- Number of LSTM cells
  encLSTMs = {},
  decLSTMs = {}
}

-- Copy encoder cell and output to decoder LSTM
function Model:forwardConnect()
  for l = 1, #Model.decLSTMs do
    Model.decLSTMs[l].userPrevOutput = Model.encLSTMs[l].output[self.seqLen]
    Model.decLSTMs[l].userPrevCell = Model.encLSTMs[l].cell[self.seqLen]
  end
end

-- Copy decoder gradients to encoder LSTM
function Model:backwardConnect()
  for l = 1, #Model.encLSTMs do
    Model.encLSTMs[l].userNextGradCell = Model.decLSTMs[l].userGradPrevCell
    Model.encLSTMs[l].gradPrevOutput = Model.decLSTMs[l].userGradPrevOutput
  end
end

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)
  self.seqLen = X:size(2) -- Treat rows as a sequence

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #Model.cellSizes do
    local inputSize = l == 1 and X:size(3) or Model.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, Model.cellSizes[l])
    self.encoder:add(self.encLSTMs[l])
  end

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #Model.cellSizes do
    local inputSize = l == 1 and X:size(3) or Model.cellSizes[l - 1]
    self.decLSTMs[l] = nn.SeqLSTM(inputSize, Model.cellSizes[l]):remember('eval') -- Retain hidden state on consecutive calls to forward during evaluation
    self.decoder:add(self.decLSTMs[l])
  end
  self.decoder:add(nn.Sequencer(nn.Linear(Model.cellSizes[#Model.cellSizes], X:size(3)))) -- Reconstruct columns
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  self.decoder:add(nn.Sigmoid(true))

  -- Create dummy container for getParameters (no other way to combine storage pointers)
  self.dummyContainer = nn.Sequential()
  self.dummyContainer:add(self.encoder)
  self.dummyContainer:add(self.decoder)

  -- Create autoencoder wrapper
  self.autoencoder = {
    parent = self
  }
  
  -- Create CUDA wrapper
  function self.autoencoder:cuda()
    self.parent.encoder:cuda()
    self.parent.decoder:cuda()
  end

  -- Create replace wrapper
  function self.autoencoder:replace(fn)
    return nil -- cuDNN require contiguous inputs so do not convert
  end

  -- Create getParameters wrapper
  function self.autoencoder:getParameters()
    return self.parent.dummyContainer:getParameters()
  end

  -- Create training wrapper
  function self.autoencoder:training()
    self.parent.encoder:training()
    self.parent.decoder:training()
  end

  -- Create evaluate wrapper
  function self.autoencoder:evaluate()
    self.parent.encoder:evaluate()
    self.parent.decoder:evaluate()
  end

  -- Create forward wrapper
  function self.autoencoder:forward(x)
    local encOut = self.parent.encoder:forward(x)
    self.parent:forwardConnect()

    -- Use target vector in training, sample from self in evaluate
    if self.parent.decoder.train then
      -- Shift decoder input sequence by one step forward
      local decInSeq = x:clone()
      decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
      decInSeq[{{}, {1}, {}}]:zero() -- Start from vector of zeros
      return self.parent.decoder:forward(decInSeq)
    else
      local decOut = torch.zeros(x:size()):typeAs(x)
      local decOutPartial = torch.zeros(x:size(1), 1, x:size(3)):typeAs(x) -- Start from vector of zeros

      -- Feed outputs back into self
      self.parent.decoder:forget() -- Clear hidden state
      for t = 1, self.parent.seqLen do
        decOutPartial = self.parent.decoder:forward(decOutPartial)
        decOut[{{}, {t}, {}}] = decOutPartial
      end
      return decOut
    end
  end
  
  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    -- Shift decoder input sequence by one step forward
    local decInSeq = x:clone()
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero()
    self.parent.decoder:backward(decInSeq, gradLoss)
    self.parent:backwardConnect()
    
    local zeroTensor = torch.Tensor(x:size(2), x:size(1), Model.cellSizes[#Model.cellSizes]):typeAs(x) -- seqlen x batch
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

return Model
