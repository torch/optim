require 'torch'
require 'optim'

require 'rosenbrock'
require 'l2'

x = torch.Tensor(2):fill(0)
config = {maxEval=10000, sigma=0.5, ftarget=0.00001, verb_disp=0}
x,fx,i=optim.cmaes(rosenbrock,x,config)


print('Rosenbrock test')
print()
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,#fx do print(i,fx[i]); end
print()
print()
