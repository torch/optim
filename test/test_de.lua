require 'torch'
require 'optim'

require 'rosenbrock'
require 'l2'


-- 10-D rosenbrock
x = torch.Tensor(2):fill(0)
config = {popsize=50, scaleFactor=0.5, crossoverRate=0.9, maxFEs=3000}

-- will take some time
x,fx=optim.de(rosenbrock,x,config)


print('Rosenbrock test')
print()
-- approx 6500 function evals expected
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,config.maxFEs do print(i,fx[i]); end
print()
print()
