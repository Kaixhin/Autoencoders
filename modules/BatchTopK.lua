local nn = require 'nn'

local BatchTopK, parent = torch.class('nn.BatchTopK', 'nn.Module')

function BatchTopK:__init(k, flag)
  parent.__init(self)
  self.k = k or 1
  self.flag = flag or false
end

function BatchTopK:updateOutput(input)
  if self.train then
    local nDim = input:nDimension()
    assert(nDim == 2, 'Error: Must be a minibatch')
    local batchSize = input:size(1)
    local kSize = math.ceil(self.k * batchSize)

    self.output:resizeAs(input):zero() -- Zero outputs
    self.buffer, self.indices = torch.topk(input, kSize, 1, self.flag)
    self.output:scatter(1, self.indices, self.buffer) -- Retain top k%
  else
    self.output = input -- Pass all inputs during evaluation
  end
  
  return self.output
end

function BatchTopK:updateGradInput(input, gradOutput)
  if self.train then
    self.gradInput:resizeAs(input):zero() -- Zero gradients
    self.gradInput:scatter(1, self.indices, gradOutput:gather(1, self.indices)) -- Allocate gradients according to indices
  else
    self.gradInput = gradOutput
  end
  
  return self.gradInput
end
