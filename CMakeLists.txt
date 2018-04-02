cmake_minimum_required(VERSION 2.8.3)
project(cublas_examples)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)
find_package(CUDA 8.0 )
SET (CUDA_ENABLE ON CACHE BOOL "enable NVCC compiler")
message(STATUS "CUDA_ENABLE is ${CUDA_ENABLE}")
MESSAGE(STATUS "CUDA_FOUND is ${CUDA_FOUND}")
MESSAGE(STATUS "CUDA_VERSION_STRING is ${CUDA_VERSION_STRING}")
MESSAGE(STATUS "CUDA_TOOLKIT_ROOT_DIR is ${CUDA_TOOLKIT_ROOT_DIR}")
MESSAGE(STATUS "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
MESSAGE(STATUS "CUDA_LIBRARIES is ${CUDA_LIBRARIES}")
MESSAGE(STATUS "CUDA_CUBLAS_LIBRARIES is ${CUDA_CUBLAS_LIBRARIES}")
MESSAGE(STATUS "CUDA_VERBOSE_BUILD is ${CUDA_VERBOSE_BUILD}")
SET (CUDA_VERBOSE_BUILD OFF CACHE BOOL "nvcc verbose" FORCE)
SET (CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets CACHE STRING "suppress deprecated warnings" FORCE)
MESSAGE(STATUS "CUDA_VERBOSE_BUILD is ${CUDA_VERBOSE_BUILD}")
MESSAGE(STATUS "CUDA_NVCC_HOST_COMPILER is ${CUDA_NVCC_HOST_COMPILER}")
MESSAGE(STATUS "CUDA_PROPAGATE_HOST_FLAGS is ${CUDA_PROPAGATE_HOST_FLAGS}")
#find_package(Eigen REQUIRED)
#message(STATUS "Found eigen3:")
#message(STATUS "  include: ${EIGEN_INCLUDE_DIRS}")
#message(STATUS "  definitions: ${EIGEN_DEFINITIONS}")

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  #LIBRARIES markov_cuda
  #CATKIN_DEPENDS CUDA
  #DEPENDS CUDA
)

###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

if(${CUDA_FOUND} AND ${CUDA_ENABLE})
  add_definitions(-DCUDA_ENABLE)
  SET(HELLO_WORLD_CUDA hello_world_cuda)
  cuda_add_library(${HELLO_WORLD_CUDA}
    src/cuda_example/hello_world.cu
  )
  target_link_libraries(${HELLO_WORLD_CUDA}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
  )
else(${CUDA_FOUND} AND ${CUDA_ENABLE})
  SET(HELLO_WORLD_CUDA "")
endif(${CUDA_FOUND} AND ${CUDA_ENABLE})

message(STATUS "HELLO_WORLD_CUDA is ${HELLO_WORLD_CUDA}")

add_library(hello_world
  src/cuda_example/hello_world.cpp
)
target_link_libraries(hello_world
  ${catkin_LIBRARIES}
  ${HELLO_WORLD_CUDA}
)

add_executable(cuda_example
  src/cuda_example/cuda_example.cpp
)
target_link_libraries(cuda_example
  ${catkin_LIBRARIES}
  hello_world
)

cuda_add_executable(cublas_example
  src/cublas_example/cublas_example.cu
)
target_link_libraries(cublas_example
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

cuda_add_executable(dot
  src/cublas_example/dot.cu
)
target_link_libraries(dot
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS cuda_example cublas_example
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_cublas_examples.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
