package = "optim"
version = "1.0-0"

source = {
   url = "git://github.com/koraykv/optim",
   tag = "1.0-0"
}

description = {
   summary = "An optimization library for Torch.",
   detailed = [[
This package contains several optimization routines for Torch.   
  ]],
   homepage = "https://github.com/koraykv/optim",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "sys >= 1.0",
}

build = {
   type = "cmake",
   variables = {
      LUAROCKS_PREFIX = "$(PREFIX)"
   }
}
