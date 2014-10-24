require 'torch'
require 'optim'

require 'rosenbrock'
require 'l2'

print('--- regular batch test ---')

x = torch.Tensor(2):fill(0)
x,fx,i=optim.lbfgs(rosenbrock,x,{maxIter=100, learningRate=1e-1})

print()
print('Rosenbrock test')
print()
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,#fx do print(i,fx[i]); end
print()
print()

print('--- stochastic test ---')

x = torch.Tensor(2):fill(0)
fx = {}
config = {learningRate=1e-1, maxIter=1}
for i = 1,100 do
	x,f=optim.lbfgs(rosenbrock,x,config)
	table.insert(fx,f[1])
end

print()
print('Rosenbrock test')
print()
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,#fx do print(i,fx[i]); end
