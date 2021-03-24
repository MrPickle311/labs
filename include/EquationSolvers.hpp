#pragma once
#include <armadillo>
#include <string>
#include <cmath>
#include <tbb/concurrent_vector.h>
#include <memory>

using DenseMatrix = arma::mat;
using SparseMatrix = arma::sp_mat;
using DenseVector = arma::colvec;
using SparseVector = arma::sp_colvec;

template<typename MatrixType,typename VectorType>
struct SharedResources
{
    MatrixType     preconditioner_;
    MatrixType     cooficient_matrix_;
    VectorType     right_side_vector_;
    VectorType     solutions_vector_;
    SharedResources(){}
    SharedResources(MatrixType const& preconditioner,
                         MatrixType const& cooficient_matrix,
                         VectorType const& right_side_vector,
                         VectorType const& solutions_vector):
        preconditioner_{preconditioner},
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        solutions_vector_{right_side_vector}
{}
};

template<typename MatrixType,typename VectorType>
class SystemChecker
{
private:
   std::shared_ptr<SharedResources<MatrixType,VectorType>> res_;
};

template<typename MatrixType,typename VectorType>
class RelaxModifier
{
private:
   std::shared_ptr<SharedResources<MatrixType,VectorType>> res_;
};

template<typename MatrixType,typename VectorType>
class Solver
{
private:
    bool dynamic_relaxing_flag_; 
    size_t iteration_;
    double relax_;
protected:
    arma::colvec right_side_vector_;
    arma::colvec solutions_vector_;
    arma::mat cooficient_matrix_;
    arma::mat preconditioner_;
private:
    inline bool vectorHasBadLength() const
    {
        return right_side_vector_.n_elem != cooficient_matrix_.n_rows;
    }
    inline bool isStartVectorEmpty() const
    {
        return solutions_vector_.empty();
    }
    void fillStartVectorWithZero()
    {
        solutions_vector_.resize(right_side_vector_.n_elem);
        for(auto&& x_i : solutions_vector_)
            x_i = 0;
    }
    bool isDiagZeroVector() const
    {
        for(auto&& e: preconditioner_)
            if(e != 0) return false;
        return true;
    }
    inline void throwError(std::string msg) const noexcept(false)
    {
        throw std::runtime_error(msg);
    }
    void checkSystemValid() const
    {
        if(!cooficient_matrix_.is_square())
            throwError(std::string{"Passed matrix is not square !"});
        if(vectorHasBadLength())
            throwError(std::string{"Vector length != matrix size !"});
    }
    inline arma::mat getSubstep() const
    {
        return right_side_vector_ - cooficient_matrix_ * solutions_vector_;
    }
    inline arma::mat getStep() const
    {
        return preconditioner_ * getSubstep();
    }
    void initSolver()
    {
        setPreconditioner();
        if(isDiagZeroVector())
            throwError(std::string{"Diagonal of cooficients is zero vector !"});
    }
    inline bool isFirstIteration() const
    {
        return iteration_ == 0;
    }
    inline void iterationEngine()
    {
        if(dynamic_relaxing_flag_ == true)
            modifyRelax();
        solutions_vector_ += relax_ *  getStep();
    }
    
    inline double getNumerator() const
    {
        return arma::dot(arma::trans( getSubstep()), cooficient_matrix_ *  getStep());
    }
    inline double getEnumerator() const
    {
        return arma::dot(arma::trans(cooficient_matrix_ *  getStep()), 
                                 cooficient_matrix_ *  getStep());
    }
    inline void modifyRelax()
    {
        relax_ =  getNumerator() / getEnumerator();
    }
    
    inline bool hasGoodPrecision(double left,double right,double precision)
    {
        return fabs(left-right) < precision;
    }
    bool isInsufficientPrecision(double precision,arma::colvec const& temp)
    {
        size_t vector_size {temp.n_elem};
        size_t precise_solutions {0};
        for(size_t i{0}; i < vector_size ; ++i)
            if(hasGoodPrecision(temp.at(i,0),solutions_vector_.at(i,0),precision))
                ++precise_solutions;
        if(precise_solutions == vector_size)
            return false; // sufficient solutions
        return true;//bad solutions
    }
protected:
    virtual void setPreconditioner() = 0; // hook 
public:
    Solver(arma::mat const& cooficient_matrix,
           arma::colvec const& right_side_vector,
           bool enable_dynamic_relax = false,
           double start_relax = 1.0,
           arma::colvec const& start_positions_vector = {}):
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        dynamic_relaxing_flag_{enable_dynamic_relax},
        preconditioner_{},
        iteration_{0},
        relax_{start_relax}
    {
        checkSystemValid();
        if(isStartVectorEmpty())
            fillStartVectorWithZero();
    }
    void operator() () // one iteration
    {
        if(isFirstIteration())
            initSolver();
        iterationEngine();
        ++iteration_;
    }

    void operator() (size_t count) // several iterations 
    {
        for(size_t i{0}; i < count; ++i)
            (*this)();
    } 
    void operator() (double precision)//counting while all |x_i+1 - x_i| < precision  
    {
        arma::colvec temp {solutions_vector_};
        (*this)();
        while (isInsufficientPrecision(precision,temp))
        {
            temp = solutions_vector_;
            (*this)();
        }
    }
    arma::colvec getSolutions()
    {
        return solutions_vector_;
    }
    size_t getIteration() const
    {
        return iteration_;
    }
};

//JacobiSolver and GaussSeidelSolver also have a dynamic MINRES relax modification

template<typename MatrixType,typename VectorType>
class JacobiSolver:
    public Solver<DenseMatrix,DenseVector>
{
protected:
    virtual void setPreconditioner()
    {
        //get diagonal of matrix
        preconditioner_ =  arma::diagmat(cooficient_matrix_).i();
}
public:
    using Solver<DenseMatrix,DenseVector>::Solver;
};

template<typename MatrixType,typename VectorType>
class GaussSeidelSolver:
    public Solver<DenseMatrix,DenseVector>
{
protected:
    virtual void setPreconditioner()
    {
        //get lower triangular parts
        preconditioner_ = arma::trimatl(cooficient_matrix_,0);
        preconditioner_ = arma::inv(preconditioner_);
    }
public:
    using Solver<DenseMatrix,DenseVector>::Solver;
};
