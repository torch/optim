package = "optim"
version = "1.0.4-1"

source = {
   url = "git://github.com/torch/optim",
   tag = "1.0.4-0"
}

description = {
   summary = "An optimization library for Torch.",
   detailed = [[
      This package contains several optimization routines for Torch.   
  ]],
   homepage = "https://github.com/torch/optim",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "sys >= 1.0",
   "paths"
}

build = {
   type= "builtin",
   modules = {

      ["optim.init"] = "init.lua",

      -- optimization algorithms
      ["optim.adagrad"] = "adagrad.lua",
      ["optim.asgd"] = "asgd.lua",
      ["optim.cg"] = "cg.lua",
      ["optim.fista"] = "fista.lua",
      ["optim.lbfgs"] = "lbfgs.lua",
      ["optim.rprop"] = "rprop.lua",
      ["optim.sgd"] = "sgd.lua",

      -- line search tools
      ["optim.lswolfe"] = "lswolfe.lua",
      ["optim.polyinterp"] = "polyinterp.lua",

      -- general tools
      ["optim.Logger"] = "Logger.lua",
      ["optim.ConfusionMatrix"] = "ConfusionMatrix.lua"
   }

   install = {
      lua = {
         ["optim.doc.README"] = "doc/README.md",
      }
   }
}
