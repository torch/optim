
require 'torch'

optim = {}

torch.include('optim', 'sgd.lua')
torch.include('optim', 'cg.lua')
torch.include('optim', 'asgd.lua')
torch.include('optim', 'fista.lua')
torch.include('optim', 'sparsecoding.lua')
