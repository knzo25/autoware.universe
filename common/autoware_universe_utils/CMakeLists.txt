cmake_minimum_required(VERSION 3.14)
project(autoware_universe_utils)

option(BUILD_EXAMPLES "Build examples" OFF)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Boost REQUIRED)

find_package(fmt REQUIRED)

include_directories(SYSTEM
  include
  ${PCL_COMMON_INCLUDE_DIRS}
)

add_library(autoware_universe_utils SHARED
  src/geometry/geometry.cpp
  src/geometry/pose_deviation.cpp
  src/geometry/boost_polygon_utils.cpp
  src/geometry/random_convex_polygon.cpp
  src/geometry/gjk_2d.cpp
  src/math/sin_table.cpp
  src/math/trigonometry.cpp
  src/ros/msg_operation.cpp
  src/ros/marker_helper.cpp
  src/ros/logger_level_configure.cpp
  src/system/backtrace.cpp
  src/system/time_keeper.cpp
)

target_include_directories(autoware_universe_utils PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

ament_target_dependencies(autoware_universe_utils
  autoware_internal_msgs
  autoware_perception_msgs
  autoware_planning_msgs
  autoware_vehicle_msgs
  builtin_interfaces
  diagnostic_msgs
  geometry_msgs
  logging_demo
  pcl_conversions
  rclcpp
  tf2
  tf2_geometry_msgs
  tier4_debug_msgs
  tier4_planning_msgs
  unique_identifier_msgs
  visualization_msgs
)

target_link_libraries(autoware_universe_utils
  fmt::fmt
  ${PCL_COMMON_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_targets(export_${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_ros REQUIRED)

  file(GLOB_RECURSE test_files test/**/*.cpp)

  ament_add_ros_isolated_gtest(test_autoware_universe_utils ${test_files})

  target_link_libraries(test_autoware_universe_utils
    autoware_universe_utils
  )
endif()

if(BUILD_EXAMPLES)
  message(STATUS "Building examples")
  file(GLOB_RECURSE example_files examples/*.cpp)

  foreach(example_file ${example_files})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    target_link_libraries(${example_name}
      autoware_universe_utils
    )
    install(TARGETS ${example_name}
      DESTINATION lib/${PROJECT_NAME}
    )
  endforeach()
endif()

ament_auto_package()
