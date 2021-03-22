#pragma once
#include <armadillo>
#include <string>
#include <cmath>
#include <iostream>
/*
TODO:
1. Zaimplementować metodę Jacobiego DONE
2. Zaimplementować podrelaksację dla MJ DONE
3. Zaimplementować metodę Gaussa-Seidla i SOR
4. Zaimplementować MINRES
5. Przeanalizować wyniki:
• Ile iteracji potrzebuje każda z metod?
• Jak wpływa na zbieżność współczynnik podrelaksacji?
*/

class Solver
{
private:
    size_t iteration_;
    arma::colvec right_side_vector_;
    arma::colvec solutions_vector_;
    double relax_;
protected:
    arma::mat cooficient_matrix_;
    arma::mat preconditioner_;
private:
    inline bool vectorHasBadLength() const
    {
        return right_side_vector_.n_elem != cooficient_matrix_.n_rows;
    }
    void checkSystemValid() const
    {
        if(!cooficient_matrix_.is_square())
            throwError(std::string{"Passed matrix is not square !"});
        if(vectorHasBadLength())
            throwError(std::string{"Vector length != matrix size !"});
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
    void initSolver()
    {
        setPreconditioner();
    }
    inline bool isFirstIteration() const
    {
        return iteration_ == 0;
    }
    //x_n+1 = Mx + Nb
    inline void iterationEngine()
    {
        solutions_vector_ += relax_ * preconditioner_ * ( right_side_vector_ - cooficient_matrix_ * solutions_vector_);
    }
protected:
    virtual void setPreconditioner() = 0;
    virtual void setCooficientMatrix() = 0;
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
public:
    Solver(arma::mat const& cooficient_matrix,
           arma::colvec const& right_side_vector,
           double relax = 1.0,
           arma::colvec const& start_positions_vector = {}):
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        preconditioner_{},
        iteration_{0},
        relax_{relax}
    {
        checkSystemValid();
        if(isStartVectorEmpty())
            fillStartVectorWithZero();
    }
    //iteration
    void operator() ()
    {
        if(isFirstIteration())
            initSolver();
        iterationEngine();
        ++iteration_;
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
    void operator() (size_t count)
    {
        for(size_t i{0}; i < count; ++i)
            (*this)();
    } 
    void operator() (double precision)
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
};

class JacobiSolver:
    public Solver
{
protected:
    virtual void setCooficientMatrix()
    {
        
    }
    virtual void setPreconditioner()
    {
        //get diagonal of matrix
        preconditioner_ =  arma::diagmat(cooficient_matrix_).i();
        if(isDiagZeroVector())
            throwError(std::string{"Diagonal of cooficients is zero vector !"});
    }
public:
    using Solver::Solver;
};

class GaussSeidelSolver:
    public Solver
{
protected:
    virtual void setCooficientMatrix()
    {

    }
    virtual void setPreconditioner()
    {
        //get lower triangular parts
        preconditioner_ = arma::trimatl(cooficient_matrix_,0);
    }
public:
    using Solver::Solver;
};

