#pragma once

#include <armadillo>
#include <string>
#include <cmath>
#include <memory>
#include "EquationSolvers.hpp"

using Matrix = arma::mat;
using Vector = arma::colvec;
using CommonResources = std::shared_ptr<SharedResources<Matrix,Vector>>;

class ConvergenceChecker
{
private:
    CommonResources res_;
private:
    inline bool isSymetric()
    {
        return arma::approx_equal(res_->cooficient_matrix_,arma::trans(res_->cooficient_matrix_),"absdiff",0.1);
    }
    inline bool isPositivelyDefined()
    {
        for(size_t i{0}; i < res_->cooficient_matrix_.n_cols; ++i)
            if( arma::dot( arma::trans( res_->cooficient_matrix_.col(i) ) , res_->cooficient_matrix_ * res_->cooficient_matrix_.col(i) <= 0 ) )
                return false;
        return true;
    }
public:
    ConvergenceChecker(){}
    void setRes(CommonResources res)
    {
        res_ = std::move(res);
    }
    bool isConvergence()
    {
        return isSymetric() && isPositivelyDefined();
    }  
};

class GradientSolver
{
private:
    CommonResources res_;
    ConvergenceChecker checker_;
    Vector working_vector_;
    size_t iteration_;
    double relax_;
    RelaxModifier<Matrix,Vector> relax_modifier_;
public:
    GradientSolver(arma::mat cooficient_matrix,arma::colvec right_side_vector,
                   arma::colvec start_position_vector = {}):
            res_{ new SharedResources<Matrix,Vector>{ cooficient_matrix,
                                                      right_side_vector,
                                                      start_position_vector}
                },
            checker_{},
            working_vector_{},
            iteration_{0},
            relax_{1},
            relax_modifier_{}
    {
        checker_.setRes(res_);
        relax_modifier_.setRes(res_);

        if(!checker_.isConvergence())
            throw std::logic_error{"Matrix is not convergence"};
    }
    void operator() ()
    {

    }
    void operator() (double precision)
    {

    }
    arma::colvec getResults()
    {
        return res_->solutions_vector_;
    }

};