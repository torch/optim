require 'torch'
require 'optim'
require 'adadelta'
require 'rosenbrock'
require 'l2'

x = torch.Tensor(2):fill(0)
fx = {}
state = {}
config = {p=0.9, eps=1e-8, learningRate=1e-3}
for i = 1,10001 do
	x,f=optim.adadelta(rosenbrock,x,config,state)
	if (i-1)%1000 == 0 then
		table.insert(fx,f[1])
	end
end

print()
print('Rosenbrock test')
print()
print('x=');print(x)
print('fx=')
for i=1,#fx do print((i-1)*1000+1,fx[i]); end
