local nn = require 'nn'
require 'rnn'

local Model = {}

-- Copy encoder cell and output to decoder LSTM
function Model:forwardConnect(encLSTM, decLSTM)
  decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[self.seqLen])
  decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[self.seqLen])
end

-- Copy decoder gradients to encoder LSTM
function Model:backwardConnect(encLSTM, decLSTM)
  encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
  encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end

function Model:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)
  self.seqLen = X:size(2) -- Treat rows as a sequence

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.SplitTable(1, 2)) -- Works for both online and batch mode
  self.encLSTM = nn.FastLSTM(X:size(3), 64, self.seqLen) -- Truncated BPTT up to sequence length
  self.encoder:add(nn.Sequencer(self.encLSTM))
  self.encoder:add(nn.SelectTable(-1)) -- Get last output

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.SplitTable(1, 2))
  self.decLSTM = nn.FastLSTM(X:size(3), 64, self.seqLen)
  self.decoder:add(nn.Sequencer(self.decLSTM))--:remember()) -- Does not forget on forward passes (so can be used online; requires manual forget calls)
  self.decoder:add(nn.Sequencer(nn.Linear(64, X:size(3)))) -- Reconstruct columns
  self.decoder:add(nn.JoinTable(1, 1)) -- Combine columns
  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

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
    self.parent.forwardConnect(self.parent, self.parent.encLSTM, self.parent.decLSTM)
    return self.parent.decoder:forward(x) -- TODO: Change input
  end
  
  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    self.parent.decoder:backward(x, gradLoss) -- TODO: Change input
    self.parent.backwardConnect(self.parent, self.parent.encLSTM, self.parent.decLSTM)
    local zeroTensor = torch.Tensor(x:size(1), 64):typeAs(x)
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

return Model
