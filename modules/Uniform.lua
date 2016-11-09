local nn = require 'nn'

local Uniform, parent = torch.class('nn.Uniform', 'nn.Module')

function Uniform:__init(a, b)
  parent.__init(self)
  self.a = a or 0
  self.b = b or 1
end

function Uniform:updateOutput(input)
  self.output:resizeAs(input)
  self.output:uniform(self.a, self.b)
  return self.output
end

function Uniform:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  return self.gradInput
end
