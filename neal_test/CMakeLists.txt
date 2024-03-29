SET(PROJECT_NAME svo_test)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 3.16.3)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Set build flags. Set IS_ARM on odroid board as environment variable
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add catkin and required ROS packages
FIND_PACKAGE(catkin REQUIRED COMPONENTS 
  roscpp
  cmake_modules
  nav_msgs
  std_msgs
  visualization_msgs
  sensor_msgs
  image_transport
  cv_bridge_1
  tf
  svo
  svo_msgs
  vikit_common
  vikit_ros
)

# Add plain cmake packages 
FIND_PACKAGE(OpenCV 4.9 REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread)

# Describe catkin Project
catkin_package(
  DEPENDS Eigen OpenCV Sophus Boost
  CATKIN_DEPENDS roscpp nav_msgs std_msgs visualization_msgs sensor_msgs image_transport cv_bridge_1 tf svo svo_msgs vikit_common vikit_ros
  INCLUDE_DIRS include
  LIBRARIES svo_visualizer
)

# Include dirs
INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

# Set link libraries
LIST(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/thirdParty/DBoW2/lib/libDBoW2.so
)

# # Create visualizer library
# ADD_LIBRARY(svo_visualizer src/visualizer.cpp)
# TARGET_LINK_LIBRARIES(svo_visualizer ${LINK_LIBS})
                                 
# # Create Executables                      
# ADD_EXECUTABLE(vo src/vo_node.cpp)
# TARGET_LINK_LIBRARIES(vo svo_visualizer)
# ADD_EXECUTABLE(benchmark src/benchmark_node.cpp)
# TARGET_LINK_LIBRARIES(benchmark svo_visualizer)

ADD_EXECUTABLE(neal_test src/initial.cpp)
TARGET_LINK_LIBRARIES(neal_test ${LINK_LIBS})
