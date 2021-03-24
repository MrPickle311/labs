cmake_minimum_required (VERSION 3.12)
project (lab3)

#enable memory sanitizer and some linker settings
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address" "-O1" "-lblas" "-llapack" "-larmadillo")

find_package(Armadillo REQUIRED)
set(CXX_STANDARD 17)

link_directories(src)
add_subdirectory(src)

include_directories(${ARMADILLO_INCLUDE_DIRS} include)
add_executable(lab3 main.cpp)

target_link_libraries(lab3 ${ARMADILLO_LIBRARIES} EquationSolvers -pthread ${LAPACK_LIBRARIES})


