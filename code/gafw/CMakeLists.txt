cmake_minimum_required (VERSION 2.6) 
project (gafw)
find_package(CUDA REQUIRED)
find_package(Boost 1.53 REQUIRED COMPONENTS regex thread)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(pil tools/CppProperties tools/CSVWriter Implimentation/GeneralImplimentation Implimentation/GPUImplimentation Implimentation/StandardOperators)
FILE(GLOB ToolsCppSources tools/CSVWriter/*.cpp tools/CppProperties/*.cpp)
FILE(GLOB PILCppSources pil/*.cpp)
FILE(GLOB ImplCppSources Implimentation/GeneralImplimentation/*.cpp Implimentation/GPUImplimentation/*.cpp Implimentation/StandardOperators/*.cpp)
FILE(GLOB ImplCuSources Implimentation/StandardOperators/*.cu)

set(CUDA_NVCC_FLAGS -lineinfo;--maxrregcount=32;-arch=sm_20;--ptxas-options=-v;-Xcompiler;-fPIC)
cuda_compile(ImplNVCCCompiled ${ImplCuSources} )

add_library(gafw-tools SHARED ${ToolsCppSources} )
target_link_libraries(gafw-tools ${Boost_LIBRARIES})

add_library(gafw-pil SHARED ${PILCppSources} )

add_library(gafw-impl SHARED ${ImplCppSources} ${ImplNVCCCompiled})
target_link_libraries(gafw-impl gafw-pil ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})


install(TARGETS gafw-tools LIBRARY DESTINATION lib)
install(TARGETS gafw-pil LIBRARY DESTINATION lib)
install(TARGETS gafw-impl LIBRARY DESTINATION lib)


