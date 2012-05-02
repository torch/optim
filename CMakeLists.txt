
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(src)
FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(optim "${src}" "${luasrc}")
ADD_TORCH_DOK(dok optim "Machine Learning" "Optimization" 3.2)
