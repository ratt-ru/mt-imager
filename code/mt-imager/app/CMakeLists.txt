cmake_minimum_required (VERSION 2.6) 
set (Boost_debug ON)
project(mt-imager-app)
find_package(CasaCore REQUIRED)
find_package(CUDA REQUIRED)
FIND_PACKAGE(Boost 1.53 REQUIRED)
find_package(Log4cxx REQUIRED)
find_package(CfitsIO REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CASACORE_INCLUDE_DIR} ${CASACORE_INCLUDE_DIR}/casacore)
include_directories(${LOG4CXX_INCLUDE_DIR})
include_directories(../lib ../lib/components ../lib/components/modules ../lib/components/statistics ../lib/components/VisibilityManager ../lib/components/ConvolutionFunctionsGenerator ../lib/operators)
#include_directories(../../gafw/StandardOperators ../lib ../../gafw/pil ../../gafw/tools/CppProperties ../../gafw/GeneralImplimentation ../../gafw/GPUImplimentation ../../gafw )
include_directories(${GAFW_PIL_INCLUDE_DIR} ${GAFW_TOOLS_INCLUDE_DIR} ${GAFW_IMPL_INCLUDE_DIR} )

FILE(GLOB MyCPPSources *.cpp)
add_executable(mt-imager  ${MyCPPSources} ) 

target_link_libraries(mt-imager ${LOG4CXX_LIBRARIES})
target_link_libraries(mt-imager mt-imager-lib)
target_link_libraries(mt-imager gafw-tools gafw-pil gafw-impl)
target_link_libraries(mt-imager ${CASACORE_LIBRARIES})
target_link_libraries(mt-imager ${CUDA_LIBRARIES})
target_link_libraries(mt-imager ${CFITSIO_LIBRARY})

install(TARGETS mt-imager RUNTIME DESTINATION bin)




