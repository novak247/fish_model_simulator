cmake_minimum_required(VERSION 3.5)
project(fish_model_simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-Wall)
add_definitions(-Wextra)
add_definitions(-fno-omit-frame-pointer)

set(CATKIN_DEPENDENCIES
  message_generation
  message_runtime
  roscpp
  mrs_msgs
  nav_msgs
  std_msgs
  mrs_lib
  )

set(EXECUTABLES
  filtration
  fish_model
  )

find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_DEPENDENCIES}

  )

generate_messages(DEPENDENCIES
  mrs_msgs
  std_msgs
  sensor_msgs
  )

find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

catkin_package(
  # INCLUDE_DIRS include
  CATKIN_DEPENDS ${CATKIN_DEPENDENCIES}
  LIBRARIES ${LIBRARIES}
  DEPENDS Eigen
  )

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenGL_INCLUDE_DIRS}
  ${LIBGBM_INCLUDE_DIRS}
  )

add_executable(filtration
  src/filtration.cpp
  )

add_executable(fish_model
  src/fish_model.cpp
  )

add_dependencies(filtration
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  )

target_link_libraries(filtration
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  )

add_dependencies(fish_model
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  )

target_link_libraries(fish_model
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  )
