#pragma once
#include <armadillo>
#include <string>
/*
TODO:
1. Zaimplementować metodę Jacobiego
2. Zaimplementować podrelaksację dla MJ
3. Zaimplementować metodę Gaussa-Seidla i SOR
4. Zaimplementować MINRES
5. Przeanalizować wyniki:
• Ile iteracji potrzebuje każda z metod?
• Jak wpływa na zbieżność współczynnik podrelaksacji?
*/

class JacobiSolver
{
private:
    arma::mat cooficient_matrix_;
    arma::colvec right_side_vector_;
    arma::colvec solutions_vector_;
    arma::mat diagonal_;
    size_t iteration_;
private:
    void throwError(std::string msg) const noexcept(false) 
    {
        throw std::runtime_error(msg);
    }
    bool isDiagZeroVector(arma::mat  const& diagonal) const
    {
        for(auto&& e: diagonal)
            if(e != 0) return false;
        return true;
    } 
    void checkSystemValid() const
    {
        if(cooficient_matrix_.n_rows != cooficient_matrix_.n_cols)
            throwError(std::string{"Passed matrix is not square !"});
        if(right_side_vector_.n_elem != cooficient_matrix_.n_rows)
            throwError(std::string{"Vector length != matrix size !"});
        if(isDiagZeroVector(diagonal_))
            throwError(std::string{"Diagonal of cooficients is zero vector !"});
    }
    bool isStartVectorEmpty() const
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
        //decomposition to 2 triangles
        arma::mat upper_triangle {arma::trimatu(cooficient_matrix_,1)};
        arma::mat lower_triangle {arma::trimatl(cooficient_matrix_,-1)};
        
        //invert diagonal matrix
        diagonal_ = diagonal_.i();

        //create a new cooficient matrix
        cooficient_matrix_ = -1*diagonal_ * ( upper_triangle + lower_triangle );
    }
    bool isFirstIteration() const
    {
        return iteration_ == 0;
    }
    //x_n+1 = Mx + Nb
    void iterationEngine()
    {
        //in a single iteration vector of solutions is constant
        arma::colvec temp {solutions_vector_};
        solutions_vector_ = cooficient_matrix_ * solutions_vector_ + 
        // for(auto&& x_i : temp)
        // {
        //     for()
        // }
    }
public:
    JacobiSolver(arma::mat const& cooficient_matrix,
                 arma::colvec const& right_side_vector,
                 arma::colvec const& start_positions_vector = {}):
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        diagonal_{cooficient_matrix.diag()},
        iteration_{0}
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
        
        ++iteration_;
    }
    void operator() (double accuracy)
    {
        
    }
    arma::colvec getSolutions()
    {
        return solutions_vector_;
    }
};