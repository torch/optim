require 'lab'
dofile('rosenbrock.lua')
dofile('l2.lua')
dofile('cg.lua')


x = torch.Tensor(2):fill(0)

x,fx=cg(rosenbrock,x)
