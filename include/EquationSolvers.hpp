#pragma once
#include <armadillo>
#include <string>
#include <cmath>
#include <iostream>

class Solver
{
private:
    bool dynamic_relaxing_flag_; 
    size_t iteration_;
    arma::colvec right_side_vector_;
    arma::colvec solutions_vector_;
    double relax_;
protected:
    arma::mat cooficient_matrix_;
    arma::mat preconditioner_;
private:
    inline bool vectorHasBadLength() const;
    void checkSystemValid() const;
    inline bool isStartVectorEmpty() const;
    void fillStartVectorWithZero();
    void initSolver();
    inline bool isFirstIteration() const;
    inline void iterationEngine();
    bool isDiagZeroVector() const;
    inline void throwError(std::string msg) const noexcept(false);
    bool isDiagZeroVector() const;
    inline void throwError(std::string msg) const noexcept(false);
    inline arma::mat getStep();
    inline arma::mat getSubstep();
    inline void modifyRelax();
    inline arma::mat getNumerator() const;
protected:
    virtual void setPreconditioner() = 0;
public:
    Solver(arma::mat const& cooficient_matrix,
           arma::colvec const& right_side_vector,
           bool enable_dynamic_relax = false,
           double start_relax = 1.0,
           arma::colvec const& start_positions_vector = {});
    //iteration
    void operator() ();
    inline bool hasGoodPrecision(double left,double right,double precision);
    bool isInsufficientPrecision(double precision,arma::colvec const& temp);
    void operator() (size_t count);
    void operator() (double precision);
    arma::colvec getSolutions();
};

//JacobiSolver and GaussSeidelSolver also have a dynamic MINRES relax modification

class JacobiSolver:
    public Solver
{
protected:
    virtual void setPreconditioner();
public:
    using Solver::Solver;
};

class GaussSeidelSolver:
    public Solver
{
protected:
    virtual void setPreconditioner();
public:
    using Solver::Solver;
};
