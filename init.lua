
require 'torch'

optim = {}

-- optimizations
torch.include('optim', 'sgd.lua')
torch.include('optim', 'cg.lua')
torch.include('optim', 'asgd.lua')
torch.include('optim', 'fista.lua')
torch.include('optim', 'lbfgs.lua')

-- tools 
torch.include('optim', 'ConfusionMatrix.lua')
torch.include('optim', 'Logger.lua')
