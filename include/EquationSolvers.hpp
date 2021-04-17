#pragma once
#include <armadillo>
#include <string>
#include <cmath>

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
    SharedResources(MatrixType const& cooficient_matrix,
                    VectorType const& right_side_vector,
                    VectorType const& solutions_vector):
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        solutions_vector_{right_side_vector}
    {}
};

template<typename MatrixType,typename VectorType>
class MatrixConvergenceChecker
{
private:
    std::shared_ptr<SharedResources<MatrixType,VectorType>> res_;
public:
    MatrixConvergenceChecker(){}
    void setRes(std::shared_ptr<SharedResources<MatrixType,VectorType>>  res)
    {
         res_ = std::move(res);
    }
    bool isConvergence()
    {
        size_t sum = 0;
        //i cannot use std::accumulate bcs this matrix does not provide matrx.begin() ,matrix.end()...
        for(size_t i = 0 ; i < res_->cooficient_matrix_.n_rows; ++i)
            for(size_t j = 0 ; j < res_->cooficient_matrix_.n_rows; ++j)
                if(j != i)
                    sum += res_->cooficient_matrix_.at(i,j);
        
        for(size_t i = 0 ; i < res_->cooficient_matrix_.n_rows; ++i)
            if(fabs(res_->cooficient_matrix_.at(i,i)) <= sum )
                return false;
        return true;
    }
};

template<typename MatrixType,typename VectorType>
class SystemMonitor
{
private:
   std::shared_ptr<SharedResources<MatrixType,VectorType>> res_;
public:
    SystemMonitor(){}
    void setRes(std::shared_ptr<SharedResources<MatrixType,VectorType>>  res)
    {
         res_ = std::move(res);
    }
    inline bool vectorHasBadLength() const
    {
        return res_->right_side_vector_.n_elem != res_->cooficient_matrix_.n_rows;
    }
    inline bool isStartVectorEmpty() const
    {
        return res_->solutions_vector_.empty();
    }
    void fillStartVectorWithZero()
    {
        res_->solutions_vector_.resize(res_->right_side_vector_.n_elem);
        for(auto&& x_i : res_->solutions_vector_)
            x_i = 0;
    }
    bool isDiagZeroVector() const
    {
        for(auto&& e: res_->preconditioner_)
            if(e != 0) return false;
        return true;
    }
    inline void throwError(std::string msg) const noexcept(false)
    {
        throw std::logic_error(msg);
    }
    void checkSystemValid() const
    {
        if(!res_->cooficient_matrix_.is_square())
            throwError(std::string{"Passed matrix is not square !"});
        if(vectorHasBadLength())
            throwError(std::string{"Vector length != matrix size !"});
    }
};

template<typename MatrixType,typename VectorType>
class RelaxModifier
{
private:
   std::shared_ptr<SharedResources<MatrixType,VectorType>> res_;
public:
    RelaxModifier(){}
    void setRes(std::shared_ptr<SharedResources<MatrixType,VectorType>>  res)
    {
        res_ = std::move(res);
    }
    inline MatrixType getSubstep() const
    {
        return res_->right_side_vector_ - res_->cooficient_matrix_ * res_->solutions_vector_;
    }
    inline MatrixType getStep() const
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

template<typename MatrixType,typename VectorType>
class Solver
{
private:
    bool dynamic_relaxing_flag_; 
    size_t iteration_;
    double relax_;
    RelaxModifier<MatrixType,VectorType>                relax_modifier_;
    SystemMonitor<MatrixType,VectorType>                system_monitor_;
    MatrixConvergenceChecker<MatrixType,VectorType>     convergence_checker_;
protected:
    std::shared_ptr<SharedResources<MatrixType,VectorType>> res_;
private:
    void initSolver()
    {
        setPreconditioner();
        if(system_monitor_.isDiagZeroVector())
            system_monitor_.throwError(std::string{"Diagonal of cooficients is zero vector !"});
    }
    inline bool isFirstIteration() const
    {
        return iteration_ == 0;
    }
    inline void iterationEngine()
    {
        if(dynamic_relaxing_flag_ == true)
            modifyRelax();
        res_->solutions_vector_ += relax_ *  relax_modifier_.getStep();
    }
    inline void modifyRelax()
    {
        relax_ = relax_modifier_.getRelax(); 
    }
    inline bool hasGoodPrecision(double left,double right,double precision)
    {
        return fabs(left-right) < precision;
    }
    //i need to optimize this below function
    bool isInsufficientPrecision(double precision,VectorType const& temp)
    {
        size_t vector_size {temp.n_elem};
        size_t precise_solutions {0};
        for(size_t i{0}; i < vector_size ; ++i)
            if(hasGoodPrecision(temp.at(i,0),res_->solutions_vector_.at(i,0),precision))
                ++precise_solutions;
        if(precise_solutions == vector_size)
            return false; // sufficient solutions
        return true;//bad solutions
    }
protected:
    virtual void setPreconditioner() = 0; // hook 
public:
    //ugly constructor , but it works 
    explicit Solver(MatrixType const& cooficient_matrix,
           VectorType const& right_side_vector,
           bool enable_dynamic_relax = false,
           double start_relax = 1.0,
           VectorType const& start_positions_vector = {}):
        iteration_{0},
        relax_{start_relax},
        res_{new SharedResources<MatrixType,VectorType>
                                    {cooficient_matrix,
                                    right_side_vector,
                                    start_positions_vector}},
        relax_modifier_{},
        system_monitor_{},
        convergence_checker_{}
    {
        relax_modifier_.setRes(res_);
        system_monitor_.setRes(res_);
        convergence_checker_.setRes(res_);

        system_monitor_.checkSystemValid();
        if(!convergence_checker_.isConvergence())
            throw std::logic_error{"Matrix is not convergence!\n"};
        if(system_monitor_.isStartVectorEmpty())
            system_monitor_.fillStartVectorWithZero();
        
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
        VectorType temp {res_->solutions_vector_};
        (*this)();
        while (isInsufficientPrecision(precision,temp))
        {
            temp = res_->solutions_vector_;
            (*this)();
        }
    }
    VectorType getSolutions()
    {
        return res_->solutions_vector_;
    }
    size_t getIteration() const
    {
        return iteration_;
    }
};

//JacobiSolver and GaussSeidelSolver also have a dynamic MINRES relax modification

template<typename MatrixType,typename VectorType>
class JacobiSolver:
    public Solver<MatrixType,VectorType>
{
protected:
    virtual void setPreconditioner()
    {
        //get diagonal of matrix
        this->res_->preconditioner_ =  arma::diagmat(this->res_->cooficient_matrix_).i();
    }
public:
    using Solver<MatrixType,VectorType>::Solver;
};

template<typename MatrixType,typename VectorType>
class GaussSeidelSolver:
    public Solver<MatrixType,VectorType>
{
protected:
    virtual void setPreconditioner()
    {
        //get lower triangular parts
        this->res_->preconditioner_ = arma::trimatl(this->res_->cooficient_matrix_,0);
        this->res_->preconditioner_ = arma::inv(this->res_->preconditioner_);
    }
public:
    using Solver<MatrixType,VectorType>::Solver;
};
