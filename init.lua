
require 'torch'

optim = {}

-- optimizations
torch.include('optim', 'sgd.lua')
torch.include('optim', 'cg.lua')
torch.include('optim', 'asgd.lua')
torch.include('optim', 'nag.lua')
torch.include('optim', 'fista.lua')
torch.include('optim', 'lbfgs.lua')
torch.include('optim', 'adagrad.lua')
torch.include('optim', 'rprop.lua')
torch.include('optim', 'adam.lua')
torch.include('optim', 'rmsprop.lua')
torch.include('optim', 'adadelta.lua')

-- line search functions
torch.include('optim', 'lswolfe.lua')

-- helpers
torch.include('optim', 'polyinterp.lua')
torch.include('optim', 'checkgrad.lua')

-- tools
torch.include('optim', 'ConfusionMatrix.lua')
torch.include('optim', 'Logger.lua')
