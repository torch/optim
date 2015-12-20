require 'torch'
require 'optim'


require 'rosenbrock'
require 'l2'

x = torch.Tensor(4):fill(0)
config = {maxEval=10000, sigma=0.5}
x,fx,i=optim.cmaes(rosenbrock,x, config)
