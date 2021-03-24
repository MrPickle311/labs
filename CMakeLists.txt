cmake_minimum_required (VERSION 3.12)
project (lab3)

#enable memory sanitizer
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address" "-O1" "-lblas" "-llapack" "-larmadillo")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address" "-O1" "-lblas" "-llapack" "-larmadillo")

#SET(CMAKE_CXX_FLAGS "-O1" "-lblas" "-llapack" "-larmadillo")
#SET(CMAKE_EXE_LINKER_FLAGS  "-lblas" "-llapack" )


set(BLA_VENDOR OpenBLAS)
find_package(LAPACK)

if(LAPACK_FOUND)
    message("lapack found")
endif()

find_package(Armadillo REQUIRED)
set(CXX_STANDARD 17)

link_directories(src)
add_subdirectory(src)

include_directories(${ARMADILLO_INCLUDE_DIRS} include)
add_executable(lab3 main.cpp)

#add_definitions(-DARMA_DONT_USE_WRAPPER)

target_link_libraries(lab3 ${ARMADILLO_LIBRARIES} EquationSolvers -pthread ${LAPACK_LIBRARIES})


