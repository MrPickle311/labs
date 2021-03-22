#pragma once
#include <armadillo>
#include <string>
#include <cmath>

class HelperObject
{
protected:
    arma::mat    const& preconditioner_;
    arma::mat    const& cooficient_matrix_;
    arma::colvec const& right_side_vector_;
    arma::colvec const& solutions_vector_;
public:
    HelperObject(arma::mat const& preconditioner,
                 arma::mat const& cooficient_matrix,
                 arma::colvec const& right_side_vector,
                 arma::colvec const& solutions_vector):
        preconditioner_{preconditioner},
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        solutions_vector_{right_side_vector}
    {}
};

class SystemChecker:
    protected HelperObject
{
public:
    using HelperObject::HelperObject;
};

class RelaxModifier:
    protected HelperObject
{
public:
    using HelperObject::HelperObject;
};

class Solver
{
private:
    bool dynamic_relaxing_flag_; 
    size_t iteration_;
    arma::colvec right_side_vector_;
    arma::colvec solutions_vector_;
    double relax_;

    RelaxModifier relax_modifier_;
    SystemChecker sys_checker_;
protected:
    arma::mat cooficient_matrix_;
    arma::mat preconditioner_;
private:
    inline bool vectorHasBadLength() const;
    inline bool isStartVectorEmpty() const;
    void fillStartVectorWithZero();
    bool isDiagZeroVector() const;
    inline void throwError(std::string msg) const noexcept(false);

     void checkSystemValid() const;

    void initSolver();
    inline bool isFirstIteration() const;
    inline void iterationEngine();
    
    inline arma::mat getStep() const;
    inline arma::mat getSubstep() const; 
    inline double getNumerator() const;
    inline double getEnumerator() const;

    inline void modifyRelax();
    
    inline bool hasGoodPrecision(double left,double right,double precision);
    bool isInsufficientPrecision(double precision,arma::colvec const& temp);
protected:
    virtual void setPreconditioner() = 0; // hook 
public:
    Solver(arma::mat const& cooficient_matrix,
           arma::colvec const& right_side_vector,
           bool enable_dynamic_relax = false,
           double start_relax = 1.0,
           arma::colvec const& start_positions_vector = {});
    void operator() (); // one iteration
    void operator() (size_t count); // several iterations 
    void operator() (double precision);//counting while all |x_i+1 - x_i| < precision  
    arma::colvec getSolutions();
    size_t getIteration() const;
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
