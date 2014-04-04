cmake_minimum_required (VERSION 2.6)
project(gafw-pil)
FILE(GLOB MyCSources *.cpp)
add_library(gafw SHARED ${MyCSources} )
