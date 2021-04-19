#include "../include/GradientSolvers.hpp"

//GradientConvergenceChecker//


bool GradientConvergenceChecker::isPositivelyDefined()
{
    for(size_t i{0}; i < res_->cooficient_matrix_.n_cols; ++i)
    {
        if( arma::norm(res_->cooficient_matrix_.col(i)) == 0)
            return false;
        if( arma::dot( arma::trans( res_->cooficient_matrix_.col(i) ) , res_->cooficient_matrix_ * res_->cooficient_matrix_.col(i) ) <= 0 )
            return false;
    }
    return true;
}

void GradientConvergenceChecker::checkConvergence()
{
    //if(!isSymetric())
    //    throw std::logic_error("Matrix is not symmetric!\n");
        
    if(!isPositivelyDefined())
        throw std::logic_error("Matrix is not positive defined!\n");
}  



//GradientSolver//

void GradientSolver::initSolver()
{
    initRMainVector();

    if(use_preconditioner_)
    {
        initPreconditioner();
        setupVectors(z_new_vector_);
    }
    else setupVectors(r_main_vector_);
}

void GradientSolver::iterationEngine()//very ugly 
{
    computeHelperMatrix();
    computeRelax();
    computeSolutionsVector();
    computeRMainVector();

    if(use_preconditioner_)
    {
        computeZNewVector();
        computeRNewVector(z_new_vector_);
    }
    else computeRNewVector(r_main_vector_);
        
        
    computeBeta();

    if(use_preconditioner_)
        computePVector(z_new_vector_);
    else computePVector(r_main_vector_);
        
    r_old_vector_ = r_main_vector_;

    if(use_preconditioner_)
        z_old_vector_ = z_new_vector_;
}

GradientSolver::GradientSolver(Matrix cooficient_matrix,
                               Vector right_side_vector,
                               bool use_preconditioner,
                               double start_relax,
                               Vector start_position_vector):
        Solver{cooficient_matrix,
               right_side_vector,
               start_position_vector,
               start_relax
               },
        checker_{},
        r_main_vector_{},
        r_old_vector_{},
        r_new_vector_{},
        beta_cooficient_{0},
        helper_vector_{},
        use_preconditioner_{use_preconditioner}
{
    checker_.setRes(res_);

    checker_.checkConvergence();
}