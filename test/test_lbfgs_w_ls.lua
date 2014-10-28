require 'torch'
require 'optim'

require 'rosenbrock'
require 'l2'

print('--- batch test w/ line search ---')

x = torch.Tensor(2):fill(0)
x,fx,i=optim.lbfgs(rosenbrock,x,{maxIter=100, lineSearch=optim.lswolfe})

print()
print('Rosenbrock test')
print()
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,#fx do print(i,fx[i]); end
print()
print()
