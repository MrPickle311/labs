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

class GradientSolver
{
private:
    CommonResources res_;
    GradientConvergenceChecker checker_;
    PrecisionChekcer precision_checker_;

    Vector helper_vector_;

    Vector r_main_vector_;//r_i
    Vector r_old_vector_;//r_i+1
    Vector r_new_vector_;

    Vector p_vector_;//p

    size_t iteration_;
    double relax_;
    double beta_cooficient_;
private:
    inline bool isFirstIteration() const
    {
        return iteration_ == 0;
    }
    inline void initSolver()
    {
        r_main_vector_ = res_->right_side_vector_ - res_->cooficient_matrix_ * res_->solutions_vector_;
        p_vector_ = r_main_vector_;
        r_old_vector_ = arma::trans(r_main_vector_) * r_main_vector_;
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
    inline void computeRNewVector()
    {
        r_new_vector_ = arma::trans(r_main_vector_) * r_main_vector_;
    }
    inline void computeBeta()
    {
        beta_cooficient_ = arma::norm(r_new_vector_) / arma::norm(r_old_vector_);
    }
    inline void computePVector()
    {
        p_vector_ = r_main_vector_ + beta_cooficient_ * p_vector_;
    }
    inline void iterationEngine()
    {
        computeHelperMatrix();
        computeRelax();
        computeSolutionsVector();
        computeRMainVector();
        computeRNewVector();
        computeBeta();
        computePVector();
        r_old_vector_ = r_main_vector_;
    }
public:
    GradientSolver(Matrix cooficient_matrix,Vector right_side_vector,
                   Vector start_position_vector = {}):
            res_{ new SharedResources{ cooficient_matrix,
                                       right_side_vector,
                                       start_position_vector}
                },
            checker_{},
            r_main_vector_{},
            r_old_vector_{},
            r_new_vector_{},
            iteration_{0},
            relax_{1},
            beta_cooficient_{0},
            precision_checker_{},
            helper_vector_{}
    {
        checker_.setRes(res_);
        precision_checker_.setRes(res_);

        checker_.checkConvergence();
    }
    void operator() ()
    {
        if(isFirstIteration());
            initSolver();
        iterationEngine();
        ++iteration_;
    }
    void operator() (size_t count) // several iterations 
    {
        for(size_t i{0}; i < count; ++i)
            (*this)();
    } 
    double operator() (double precision)//counting while all |x_i+1 - x_i| < precision  
    {
        auto start = std::chrono::steady_clock::now();
        Vector temp {res_->solutions_vector_};
        (*this)();
        while (precision_checker_.isInsufficientPrecision(precision,temp))
        {
            temp = res_->solutions_vector_;
            (*this)();
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> seconds {end - start};
        return seconds.count();
    }

    arma::colvec getResults()
    {
        return res_->solutions_vector_;
    }

    size_t getIteration() const
    {
        return iteration_;
    }

};
