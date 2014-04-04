cmake_minimum_required (VERSION 2.6) 
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

SET(CMAKE_INSTALL_PREFIX /opt/software/mt-imager)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")


set(BOOST_ROOT /opt/software/boost)
set(CASACORE_ROOT_DIR /opt/software/casacore)
set(GAFW_PIL_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/code/gafw/pil)
set(GAFW_TOOLS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/code/gafw ${CMAKE_CURRENT_LIST_DIR}/code/gafw/tools ${CMAKE_CURRENT_LIST_DIR}/code/gafw/tools/CSVWriter ${CMAKE_CURRENT_LIST_DIR}/code/gafw/tools/CppProperties)
set(GAFW_IMPL_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/code/gafw/Implimentation/GeneralImplimentation ${CMAKE_CURRENT_LIST_DIR}/code/gafw/Implimentation/GPUImplimentation ${CMAKE_CURRENT_LIST_DIR}/code/gafw/Implimentation/StandardOperators)

add_subdirectory(code)

