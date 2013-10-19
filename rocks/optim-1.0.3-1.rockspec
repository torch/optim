package = "optim"
version = "1.0.3-1"

source = {
   url = "git://github.com/torch/optim",
   tag = "1.0.3-1"
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
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
