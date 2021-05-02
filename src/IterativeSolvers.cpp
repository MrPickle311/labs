#include "../include/IterativeSolvers.hpp"

//ConvergenceChecker//

bool ConvergenceChecker::isConvergence()
{
    Matrix D  { arma::diagmat(res_->cooficient_matrix_) };
    Matrix L  { arma::trimatl(res_->cooficient_matrix_,-1) };
    Matrix U  { arma::trimatu(res_->cooficient_matrix_,1) };

    Matrix B = -1 * (D + L).i() * U;

    arma::cx_vec eigen_values ;
    arma::eig_gen(eigen_values,B);

    for(size_t i{0}; i < eigen_values.n_cols;++i)
        if(std::abs(eigen_values.at(i)) >= 1)
            return false;
        
    return true;
}

//SystemMonitor//


void SystemMonitor::fillStartVectorWithZero()
{
    res_->solutions_vector_.resize(res_->right_side_vector_.n_elem);
    for(auto&& x_i : res_->solutions_vector_)
        x_i = 0;
}

bool SystemMonitor::isDiagZeroVector() const
{
    for(auto&& e: res_->preconditioner_)
        if(e != 0) return false;
    return true;
}

void SystemMonitor::checkSystemValid() const
{
    if(!res_->cooficient_matrix_.is_square())
        throwError(std::string{"Passed matrix is not square !"});
    if(vectorHasBadLength())
        throwError(std::string{"Vector length != matrix size !"});
}



//IterativeSolver//

void IterativeSolver::initSolver()
{
    setPreconditioner();
    if(system_monitor_.isDiagZeroVector())
        system_monitor_.throwError(std::string{"Diagonal of cooficients is zero vector !"});
}

void IterativeSolver::iterationEngine()
{
    if(dynamic_relaxing_flag_ == true)
        modifyRelax();
    res_->solutions_vector_ += relax_ *  relax_modifier_.getStep();
}

IterativeSolver::IterativeSolver(Matrix const& cooficient_matrix,
                                 Vector const& right_side_vector,
                                 bool   enable_dynamic_relax,
                                 double start_relax,
                                 Vector const& start_positions_vector):
        Solver{cooficient_matrix,
                right_side_vector,
                start_positions_vector,
                start_relax
                },
        relax_modifier_{},
        system_monitor_{},
        convergence_checker_{},
        dynamic_relaxing_flag_{enable_dynamic_relax},
        iteration_{0}
{
    relax_modifier_.setRes(res_);
    system_monitor_.setRes(res_);
    convergence_checker_.setRes(res_);

    system_monitor_.checkSystemValid();
    
    if(!convergence_checker_.isConvergence())
        throw std::logic_error{"Matrix is not convergence!\n"};
    if(system_monitor_.isStartVectorEmpty())
        system_monitor_.fillStartVectorWithZero();
        
}

//JacobiSolver//

void JacobiSolver::setPreconditioner()
{
    //get diagonal of matrix
    this->res_->preconditioner_ =  arma::diagmat(this->res_->cooficient_matrix_).i();
}



//GaussSeidelSolver//


void GaussSeidelSolver::setPreconditioner()
{
    //get lower triangular parts
    this->res_->preconditioner_ = arma::trimatl(this->res_->cooficient_matrix_,0);
    this->res_->preconditioner_ = arma::inv(this->res_->preconditioner_);
}