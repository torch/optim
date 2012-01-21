require 'lab'
require 'optim'
require 'plot'
dofile 'rosenbrock.lua'
dofile 'l2.lua'

x = torch.Tensor(2):fill(0)
x,fx,i=optim.lbfgs(rosenbrock,x,{maxIter=100, verbose=true, learningRate=1e-1})

print()
print('Rosenbrock test')
print()
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,#fx do print(i,fx[i]); end
