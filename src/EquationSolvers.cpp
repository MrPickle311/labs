#include "../include/EquationSolvers.hpp"

// template<typename MatrixType,typename VectorType>
// SharedResources<MatrixType,VectorType>::
// SharedResources(MatrixType const& preconditioner,
//                      MatrixType const& cooficient_matrix,
//                      VectorType const& right_side_vector,
//                      VectorType const& solutions_vector):
//         preconditioner_{preconditioner},
//         cooficient_matrix_{cooficient_matrix},
//         right_side_vector_{right_side_vector},
//         solutions_vector_{right_side_vector}
// {}


// template<typename MatrixType,typename VectorType>
// bool Solver<MatrixType,VectorType>::vectorHasBadLength() const
// {
//     return right_side_vector_.n_elem != cooficient_matrix_.n_rows;
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::checkSystemValid() const
// {
//     if(!cooficient_matrix_.is_square())
//         throwError(std::string{"Passed matrix is not square !"});
//     if(vectorHasBadLength())
//         throwError(std::string{"Vector length != matrix size !"});
// }

// template<typename MatrixType,typename VectorType>
// bool Solver<MatrixType,VectorType>::isStartVectorEmpty() const
// {
//     return solutions_vector_.empty();
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::fillStartVectorWithZero()
// {
//     solutions_vector_.resize(right_side_vector_.n_elem);
//     for(auto&& x_i : solutions_vector_)
//         x_i = 0;
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::initSolver()
// {
//     setPreconditioner();
//     if(isDiagZeroVector())
//         throwError(std::string{"Diagonal of cooficients is zero vector !"});
// }

// template<typename MatrixType,typename VectorType>
// bool Solver<MatrixType,VectorType>::isFirstIteration() const
// {
//     return iteration_ == 0;
// }

// template<typename MatrixType,typename VectorType>
// arma::mat Solver<MatrixType,VectorType>::getSubstep() const
// {
//     return right_side_vector_ - cooficient_matrix_ * solutions_vector_;
// }

// template<typename MatrixType,typename VectorType>
// arma::mat Solver<MatrixType,VectorType>::getStep() const
// {
//     return preconditioner_ * getSubstep();
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::iterationEngine()
// {
//     if(dynamic_relaxing_flag_ == true)
//         modifyRelax();
//     solutions_vector_ += relax_ *  getStep();
// }

// template<typename MatrixType,typename VectorType>
// double Solver<MatrixType,VectorType>::getNumerator() const
// {
//     return arma::dot(arma::trans( getSubstep()), cooficient_matrix_ *  getStep());
// }

// template<typename MatrixType,typename VectorType>
// double Solver<MatrixType,VectorType>::getEnumerator() const
// {
//     return arma::dot(arma::trans(cooficient_matrix_ *  getStep()), 
//                                  cooficient_matrix_ *  getStep());
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::modifyRelax()
// {
//     relax_ =  getNumerator() / getEnumerator();
// }

// template<typename MatrixType,typename VectorType>
// bool Solver<MatrixType,VectorType>::isDiagZeroVector() const
// {
//     for(auto&& e: preconditioner_)
//         if(e != 0) return false;
//     return true;
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::throwError(std::string msg) const noexcept(false) 
// {
//     throw std::runtime_error(msg);
// }

// template<typename MatrixType,typename VectorType>
// Solver<MatrixType,VectorType>::Solver(arma::mat const& cooficient_matrix,
//                arma::colvec const& right_side_vector,
//                bool enable_dynamic_relax ,
//                double start_relax,
//                arma::colvec const& start_positions_vector):
//         cooficient_matrix_{cooficient_matrix},
//         right_side_vector_{right_side_vector},
//         dynamic_relaxing_flag_{enable_dynamic_relax},
//         preconditioner_{},
//         iteration_{0},
//         relax_{start_relax}
// {
//     checkSystemValid();
//     if(isStartVectorEmpty())
//         fillStartVectorWithZero();
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::operator() ()
// {
//     if(isFirstIteration())
//         initSolver();
//     iterationEngine();
//     ++iteration_;
// }

// template<typename MatrixType,typename VectorType>
// bool Solver<MatrixType,VectorType>::hasGoodPrecision(double left,double right,double precision)
// {
//     return fabs(left-right) < precision;
// }

// template<typename MatrixType,typename VectorType>
// bool Solver<MatrixType,VectorType>::isInsufficientPrecision(double precision,arma::colvec const& temp)
// {
//     size_t vector_size {temp.n_elem};
//     size_t precise_solutions {0};
//     for(size_t i{0}; i < vector_size ; ++i)
//         if(hasGoodPrecision(temp.at(i,0),solutions_vector_.at(i,0),precision))
//             ++precise_solutions;
//     if(precise_solutions == vector_size)
//         return false; // sufficient solutions
//     return true;//bad solutions
// }

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::operator() (size_t count)
// {
//     for(size_t i{0}; i < count; ++i)
//         (*this)();
// } 

// template<typename MatrixType,typename VectorType>
// void Solver<MatrixType,VectorType>::operator() (double precision)
// {
//     arma::colvec temp {solutions_vector_};
//     (*this)();
//     while (isInsufficientPrecision(precision,temp))
//     {
//         temp = solutions_vector_;
//         (*this)();
//     }
// }

// template<typename MatrixType,typename VectorType>
// arma::colvec Solver<MatrixType,VectorType>::getSolutions()
// {
//     return solutions_vector_;
// }

// template<typename MatrixType,typename VectorType>
// size_t Solver<MatrixType,VectorType>::getIteration() const
// {
//     return iteration_;
// }

// template<typename MatrixType,typename VectorType>
// void JacobiSolver<MatrixType,VectorType>::setPreconditioner()
// {
//     //get diagonal of matrix
//     preconditioner_ =  arma::diagmat(cooficient_matrix_).i();
// }

// template<typename MatrixType,typename VectorType>
// void GaussSeidelSolver<MatrixType,VectorType>::setPreconditioner()
// {
//     //get lower triangular parts
//     preconditioner_ = arma::trimatl(cooficient_matrix_,0);
//     preconditioner_ = arma::inv(preconditioner_);
// }

