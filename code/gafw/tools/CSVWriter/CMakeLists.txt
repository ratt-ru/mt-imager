cmake_minimum_required (VERSION 2.6)
project(CSVWriter)
FILE(GLOB MyCSources *.cpp)
add_library(CSVWriter SHARED ${MyCSources} )
