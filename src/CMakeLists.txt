cmake_minimum_required(VERSION 2.8)
project(lab5)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(panoramic_generator panoramic_generator.cpp PanoramicImage.cpp PanoramicImage.h panoramic_utils.h)

target_link_libraries(panoramic_generator ${OpenCV_LIBS})
