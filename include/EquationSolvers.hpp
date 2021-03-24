#pragma once
#include <armadillo>
#include <string>
#include <cmath>
#include <tbb/concurrent_vector.h>

using DenseMatrix = arma::mat;
using SpareMatrix = arma::sp_mat;
using DenseVector = arma::colvec;
using SparseVector = arma::sp_colvec;

template<typename MatrixType,typename VectorType>
class DenseSharedResources
{
protected:
    MatrixType     preconditioner_;
    MatrixType     cooficient_matrix_;
    VectorType     right_side_vector_;
    VectorType     solutions_vector_;
public:
    DenseSharedResources(){}
    DenseSharedResources(MatrixType const& preconditioner,
                         MatrixType const& cooficient_matrix,
                         VectorType const& right_side_vector,
                         VectorType const& solutions_vector);
};

template<typename MatrixType,typename VectorType>
class SystemChecker
{
private:
   
};

template<typename MatrixType,typename VectorType>
class RelaxModifier
{

};

class Solver
{
private:
    DenseSharedResources<DenseMatrix,SparseVector> xd;
    bool dynamic_relaxing_flag_; 
    size_t iteration_;
    double relax_;
protected:
    arma::colvec right_side_vector_;
    arma::colvec solutions_vector_;
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
