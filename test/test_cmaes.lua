require 'torch'
require 'optim'

require 'rosenbrock'
require 'l2'

-- 10-D rosenbrock
x = torch.Tensor(10):fill(0)
config = {maxEval=10000, sigma=0.5, verb_disp=0}

-- will take some time
x,fx,i=optim.cmaes(rosenbrock,x,config)


print('Rosenbrock test')
print()
-- approx 6500 function evals expected
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,#fx do print(i,fx[i]); end
print()
print()