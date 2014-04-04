cmake_minimum_required (VERSION 2.6) 

FILE(GLOB MyCSources *.cpp components/*.cpp components/ConvolutionFunctionsGenerator/*.cpp components/statistics/*.cpp components/VisibilityManager/*.cpp operators/*.cpp)

find_package(CUDA REQUIRED)
find_package(Boost 1.53 REQUIRED COMPONENTS system thread chrono)
find_package(CasaCore REQUIRED)
find_package(CfitsIO REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CASACORE_INCLUDE_DIR} ${CASACORE_INCLUDE_DIR}/casacore)

include_directories(components/ components/ConvolutionFunctionsGenerator components/statistics components/VisibilityManager operators .)
include_directories(${GAFW_PIL_INCLUDE_DIR} ${GAFW_TOOLS_INCLUDE_DIR} ${GAFW_IMPL_INCLUDE_DIR} )
FILE(GLOB MyCUSources operators/*.cu)
set(CUDA_NVCC_FLAGS -lineinfo;--maxrregcount=32;-arch=sm_30;--ptxas-options=-v;-Xcompiler;-fPIC)
cuda_compile(CompiledCuda ${MyCUSources} )
set (CMAKE_CXX_FLAGS "-DDEFAULT_CONF_FILE=${CMAKE_INSTALL_PREFIX}/conf/mtimager.properties")

add_library(mt-imager-lib SHARED ${MyCSources} ${CompiledCuda} )
target_link_libraries(mt-imager-lib gafw-tools gafw-pil gafw-impl)
target_link_libraries(mt-imager-lib log4cxx)
target_link_libraries(mt-imager-lib ${CASACORE_LIBRARIES}) 
target_link_libraries(mt-imager-lib ${Boost_LIBRARIES})
target_link_libraries(mt-imager-lib ${CFITSIO_LIBRARY})

install(TARGETS mt-imager-lib LIBRARY DESTINATION lib)