#pragma once
#include <armadillo>
#include <string>
#include <cmath>
#include <memory>
#include "EquationSolvers.hpp"

class GradientConvergenceChecker:
    public SupportObject
{
public:
    using SupportObject::SupportObject;
private:
    inline bool isSymetric()
    {
        return res_->cooficient_matrix_.is_symmetric();
    }
    inline bool isPositivelyDefined()
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
public:
    void checkConvergence()
    {
        if(!isSymetric())
            throw std::logic_error("Matrix is not symmetric!\n");
        
        if(!isPositivelyDefined())
            throw std::logic_error("Matrix is not positive defined!\n");
    }  
};

class GradientSolver:
    public Solver
{
private:
    GradientConvergenceChecker checker_;

    Vector helper_vector_;

    Vector r_main_vector_;//r_i
    Vector r_old_vector_;//r_i+1
    Vector r_new_vector_;

    Vector p_vector_;//p

    double beta_cooficient_;

    bool   use_preconditioner_;

    Vector z_old_vector_;
    Vector z_new_vector_;
private:
    inline void initPreconditioner()
    {
        std::cout << "xdxxd\n";
        z_old_vector_ = res_->cooficient_matrix_.i() * r_main_vector_;
        z_new_vector_ = z_old_vector_;
    }
    inline void setupVectors(Vector const& init_vec)
    {
        p_vector_ = init_vec;
        r_old_vector_ = arma::trans(r_main_vector_) * init_vec;
    }
    void initSolver()
    {
        r_main_vector_ = res_->right_side_vector_ - res_->cooficient_matrix_ * res_->solutions_vector_;

        if(use_preconditioner_)
        {
            initPreconditioner();
            setupVectors(z_new_vector_);
        }
        else setupVectors(r_main_vector_);
    }
    inline void computeHelperMatrix()
    {
        helper_vector_ = res_->cooficient_matrix_ * p_vector_;
    }
    inline void computeRelax()
    {
        relax_ =  arma::norm(r_old_vector_)   / 
                  arma::norm(arma::trans(p_vector_) * helper_vector_) ;
    }
    inline void computeSolutionsVector()
    {
        res_->solutions_vector_ += relax_ * p_vector_; 
    }
    inline void computeRMainVector()
    {
        r_main_vector_ -=  relax_ * helper_vector_; 
    }
    inline void computeZNewVector()
    {
        z_new_vector_ = res_->cooficient_matrix_.i() * r_main_vector_;
    }
    inline void computeRNewVector(Vector const& compute_vec)
    {
        r_new_vector_ = arma::trans(r_main_vector_) * compute_vec;
    }
    inline void computeBeta()
    {
        beta_cooficient_ = arma::norm(r_new_vector_) / arma::norm(r_old_vector_);
    }
    inline void computePVector(Vector const& comp_vec)
    {
        p_vector_ = comp_vec + beta_cooficient_ * p_vector_;
    }
    inline void iterationEngine()
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
public:
    GradientSolver(Matrix cooficient_matrix,
                   Vector right_side_vector,
                   bool use_preconditioner = false,
                   Vector start_position_vector = {}):
        Solver{cooficient_matrix,
               right_side_vector,
               start_position_vector
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
};
