require 'torch'
require 'optim'

require 'rosenbrock'
require 'l2'


x = torch.Tensor(2):fill(0)
x,fx,i=optim.cg(rosenbrock,x,{maxIter=50})

print()
print('Rosenbrock test: compare with http://www.gatsby.ucl.ac.uk/~edward/code/minimize/example.html')
print()
print('Number of function evals = ',i)
print('x=');print(x)
print('fx=')
for i=1,#fx do print(i,fx[i]); end
