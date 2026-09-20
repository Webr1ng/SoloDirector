cmake_minimum_required(VERSION 3.20)
project(solo_director_camera_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(camera_bridge_core
    src/camera_backend.cpp
    src/mock_camera_backend.cpp
    src/insta360_camera_backend.cpp
)
target_include_directories(camera_bridge_core PUBLIC include)
target_compile_features(camera_bridge_core PUBLIC cxx_std_17)

add_executable(solo_director_camera_bridge src/main.cpp)
target_link_libraries(solo_director_camera_bridge PRIVATE camera_bridge_core)
target_compile_features(solo_director_camera_bridge PRIVATE cxx_std_17)
