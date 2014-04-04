cmake_minimum_required (VERSION 2.6)
project(gafw-genimpl)
FILE(GLOB MyCSources *.cpp)
include_directories(../pil ../tools/CppProperties)

add_library(gafw-genimpl SHARED ${MyCSources} )
target_link_libraries(gafw-genimpl gafw)
install(TARGETS gafw-genimpl LIBRARY DESTINATION lib)