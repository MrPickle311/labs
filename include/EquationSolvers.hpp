#pragma once
#include <armadillo>
#include <string>
#include <cmath>
#include <memory>
#include <chrono>

using Matrix = arma::mat;
using Vector = arma::colvec;

struct SharedResources
{
    Matrix     preconditioner_;
    Matrix     cooficient_matrix_;
    Vector     right_side_vector_;
    Vector     solutions_vector_;
    SharedResources(Matrix const& cooficient_matrix,
                    Vector const& right_side_vector,
                    Vector const& solutions_vector):
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        solutions_vector_{right_side_vector},
        preconditioner_{}
    {}
};

using CommonResources = std::shared_ptr<SharedResources>;

class SupportObject
{
protected:
    CommonResources res_;
public:
    SupportObject(){}
    void setRes(CommonResources  res)
    {
         res_ = std::move(res);
    }
};

class ConvergenceChecker:
    public SupportObject
{
public:
    using SupportObject::SupportObject;

    bool isConvergence()
    {
        Matrix D  { arma::diagmat(res_->cooficient_matrix_) };
        Matrix L  { arma::trimatl(res_->cooficient_matrix_,-1) };
        Matrix U  { arma::trimatu(res_->cooficient_matrix_,1) };

        Matrix B = -1 * (D + L).i() * U;

        arma::cx_vec eigen_values ;
        arma::eig_gen(eigen_values,B);

        for(size_t i{0}; i < eigen_values.n_cols;++i)
            if(std::abs(eigen_values.at(i)) >= 1)
                return false;
        
        return true;
    }
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

class PrecisionChekcer:
    public SupportObject
{
public:
    using SupportObject::SupportObject;

    inline bool hasGoodPrecision(double left,double right,double precision)
    {
        return ( fabs(left-right) / left ) < precision;
    }
    //i need to optimize this below function
    bool isInsufficientPrecision(double precision,Vector const& temp)
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
};

class Solver
{
protected:
    CommonResources res_; 
    PrecisionChekcer precision_checker_;
    size_t iteration_;
    double relax_;
public:
    Solver(Matrix cooficient_matrix,
           Vector right_side_vector,
           Vector start_position_vector = {},
           double start_relax = 1.0):
        res_{ new SharedResources{ cooficient_matrix,
                                   right_side_vector,
                                   start_position_vector}
            },        
        iteration_{0},    
        precision_checker_{},
        relax_{start_relax}
    {
        precision_checker_.setRes(res_);
    }       
    inline bool isFirstIteration() const
    {
        return iteration_ == 0;
    }

    virtual void initSolver() = 0;
    virtual void iterationEngine() = 0;
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

class IterationSolver:
    public Solver
{
private:
    bool                         dynamic_relaxing_flag_; 
    size_t                       iteration_;
    RelaxModifier                relax_modifier_;
    SystemMonitor                system_monitor_;
    ConvergenceChecker           convergence_checker_;
private:
    void initSolver()
    {
        setPreconditioner();
        if(system_monitor_.isDiagZeroVector())
            system_monitor_.throwError(std::string{"Diagonal of cooficients is zero vector !"});
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
protected:
    virtual void setPreconditioner() = 0; // hook 
public:
    //ugly constructor , but it works 
    explicit IterationSolver(Matrix const& cooficient_matrix,
                             Vector const& right_side_vector,
                             bool   enable_dynamic_relax = false,
                             double start_relax = 1.0,
                             Vector const& start_positions_vector = {}):
        Solver{cooficient_matrix,
                right_side_vector,
                start_positions_vector,
                start_relax
                },
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
};

//JacobiSolver and GaussSeidelSolver also have a dynamic MINRES relax modification

class JacobiSolver:
    public IterationSolver
{
protected:
    virtual void setPreconditioner()
    {
        //get diagonal of matrix
        this->res_->preconditioner_ =  arma::diagmat(this->res_->cooficient_matrix_).i();
    }
public:
    using IterationSolver::IterationSolver;
};

class GaussSeidelSolver:
    public IterationSolver
{
protected:
    virtual void setPreconditioner()
    {
        //get lower triangular parts
        this->res_->preconditioner_ = arma::trimatl(this->res_->cooficient_matrix_,0);
        this->res_->preconditioner_ = arma::inv(this->res_->preconditioner_);
    }
public:
    using IterationSolver::IterationSolver;
};
