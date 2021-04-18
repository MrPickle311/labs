#pragma once

#include "Solver.hpp"

class ConvergenceChecker:
    public SupportObject
{
public:
    using SupportObject::SupportObject;

    bool isConvergence();
};

class SystemMonitor:
    public SupportObject
{
public:
    using SupportObject::SupportObject;

    inline bool vectorHasBadLength() const
    {
        return res_->right_side_vector_.n_elem != res_->cooficient_matrix_.n_rows;
    }

    inline bool isStartVectorEmpty() const
    {
        return res_->solutions_vector_.empty();
    }

    void fillStartVectorWithZero();
    bool isDiagZeroVector() const;

    inline void throwError(std::string msg) const noexcept(false)
    {
        throw std::logic_error(msg);
    }

    void checkSystemValid() const;
};

class RelaxModifier:
    public SupportObject
{
public:
    using SupportObject::SupportObject;

    inline Matrix getSubstep() const
    {
        return res_->right_side_vector_ - res_->cooficient_matrix_ * res_->solutions_vector_;
    }
    inline Matrix getStep() const
    {
        return res_->preconditioner_ * getSubstep();
    }
    inline double getNumerator() const
    {
        return arma::dot(arma::trans( getSubstep()), res_->cooficient_matrix_ *  getStep());
    }
    inline double getEnumerator() const
    {
        return arma::dot(arma::trans(res_->cooficient_matrix_ *  getStep()), 
                                     res_->cooficient_matrix_ *  getStep());
    }
    inline double getRelax()
    {
        return getNumerator() / getEnumerator();
    }
};

class IterativeSolver:
    public Solver
{
private:
    bool                         dynamic_relaxing_flag_; 
    size_t                       iteration_;
    RelaxModifier                relax_modifier_;
    SystemMonitor                system_monitor_;
    ConvergenceChecker           convergence_checker_;
private:
    void initSolver();

    void iterationEngine();

    inline void modifyRelax()
    {
        relax_ = relax_modifier_.getRelax(); 
    }
protected:
    virtual void setPreconditioner() = 0; // hook 
public:
    //ugly constructor , but it works 
    explicit IterativeSolver(Matrix const& cooficient_matrix,
                             Vector const& right_side_vector,
                             bool   enable_dynamic_relax = false,
                             double start_relax = 1.0,
                             Vector const& start_positions_vector = {});
};

//JacobiSolver and GaussSeidelSolver also have a dynamic MINRES relax modification

class JacobiSolver:
    public IterativeSolver
{
protected:
    virtual void setPreconditioner();
public:
    using IterativeSolver::IterativeSolver;
};

class GaussSeidelSolver:
    public IterativeSolver
{
protected:
    virtual void setPreconditioner();
public:
    using IterativeSolver::IterativeSolver;
};


