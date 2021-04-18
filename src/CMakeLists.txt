set(CXX_STANDARD 17)
add_library(EquationSolvers STATIC
            GradientSolvers.cpp
            IterativeSolvers.cpp
            Solver.cpp
            MatrixStream.cpp
           )           
target_link_libraries(EquationSolvers ${ARMADILLO_LIBRARIES} -pthread ${LAPACK_LIBRARIES})