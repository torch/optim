
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
FIND_PACKAGE(Torch REQUIRED)

SET(src)
FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(optim "${src}" "${luasrc}")
#ADD_TORCH_DOK(dok optim "Machine Learning" "Optimization" 3.2)
