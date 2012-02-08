require 'torch'
require 'optim'

dofile 'rosenbrock.lua'
dofile 'l2.lua'

x = torch.Tensor(2):fill(0)
fx = {}

config = {learningRate=1e-3}
for i = 1,10001 do
	x,f=optim.sgd(rosenbrock,x,config)
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
