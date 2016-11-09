local nn = require 'nn'

local Gaussian, parent = torch.class('nn.Gaussian', 'nn.Module')

function Gaussian:__init(mean, stdv)
  parent.__init(self)
  self.mean = mean or 0
  self.stdv = stdv or 1
end

function Gaussian:updateOutput(input)
  self.output:resizeAs(input)
  self.output:normal(self.mean, self.stdv)
  return self.output
end

function Gaussian:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  return self.gradInput
end
