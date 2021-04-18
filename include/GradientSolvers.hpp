#pragma once

#include "Solver.hpp"

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
    bool isPositivelyDefined();
public:
    void checkConvergence();
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
        z_old_vector_ = res_->cooficient_matrix_.i() * r_main_vector_;
        z_new_vector_ = z_old_vector_;
    }
    inline void setupVectors(Vector const& init_vec)
    {
        p_vector_ = init_vec;
        r_old_vector_ = arma::trans(r_main_vector_) * init_vec;
    }
    inline void initRMainVector()
    {
        r_main_vector_ = res_->right_side_vector_ - res_->cooficient_matrix_ * res_->solutions_vector_;
    }

    void initSolver();
    
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

    void iterationEngine();

public:
    GradientSolver(Matrix cooficient_matrix,
                   Vector right_side_vector,
                   bool use_preconditioner = false,
                   double start_relax = 1.0,
                   Vector start_position_vector = {});
};
