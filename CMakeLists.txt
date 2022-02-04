cmake_minimum_required(VERSION 3.0.0)
project(opencv-tut-cmake VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv-tut-cmake src/main.cpp)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
