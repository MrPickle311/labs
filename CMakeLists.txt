cmake_minimum_required (VERSION 3.12)
project (lab3)

#enable memory sanitizer
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

find_package(Armadillo REQUIRED)
set(CXX_STANDARD 17)
include_directories(${ARMADILLO_INCLUDE_DIRS} include)
add_executable(lab3 main.cpp)
target_link_libraries(lab3 ${ARMADILLO_LIBRARIES})


