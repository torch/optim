
package = "optim"
version = "1.0-1"

source = {
   url = "optim-1.0-1.tgz"
}

description = {
   summary = "This package contains several optimization routines for Torch7.",
   detailed = [[
         This package contains several optimization routines for Torch7.
   ]],
   homepage = "",
   license = "MIT/X11" -- or whatever you like
}

dependencies = {
   "lua >= 5.1",
   "torch"
}

build = {
   type = "cmake",

   variables = {
      CMAKE_INSTALL_PREFIX = "$(PREFIX)"
   }
}
