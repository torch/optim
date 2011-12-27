cmake_minimum_required(VERSION 2.8)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

IF(${CMAKE_PROJECT_NAME} STREQUAL "Torch")
  SET(TORCH_LIBRARIES TH luaT)
  SET(TORCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  SET(TORCH_TH TH)
  SET(TORCH_luaT luaT)

  SET(INSTALL_PREFIX /${Torch_INSTALL_LUA_PATH_SUBDIR}/optim)
  SET(CINSTALL_PREFIX /${Torch_INSTALL_LUA_CPATH_SUBDIR})
  INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})
  SET(LIBS ${TORCH_LIBRARIES} ${LUA_LIBRARIES})
ELSE(${CMAKE_PROJECT_NAME} STREQUAL "Torch")
  # infer path for Torch7
  string (REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" TORCH_PREFIX "${CMAKE_INSTALL_PREFIX}" )
  message (STATUS "Found Torch7, installed in: " ${TORCH_PREFIX})
  
  find_package (Torch REQUIRED)
  SET(INSTALL_PREFIX /lua/optim)
  SET(CINSTALL_PREFIX /lib)
ENDIF(${CMAKE_PROJECT_NAME} STREQUAL "Torch")

include_directories (${TORCH_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})

install_files(${INSTALL_PREFIX} init.lua)
install_files(${INSTALL_PREFIX} sgd.lua)
install_files(${INSTALL_PREFIX} asgd.lua)
install_files(${INSTALL_PREFIX} cg.lua)
install_files(${INSTALL_PREFIX} fista.lua)
install_files(${INSTALL_PREFIX} sparsecoding.lua)
