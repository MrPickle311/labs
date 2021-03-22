#include "../include/EquationSolvers.hpp"

bool Solver::vectorHasBadLength() const
{
    return right_side_vector_.n_elem != cooficient_matrix_.n_rows;
}

void Solver::checkSystemValid() const
{
    if(!cooficient_matrix_.is_square())
        throwError(std::string{"Passed matrix is not square !"});
    if(vectorHasBadLength())
        throwError(std::string{"Vector length != matrix size !"});
}

bool Solver::isStartVectorEmpty() const
{
    return solutions_vector_.empty();
}

void Solver::fillStartVectorWithZero()
{
    solutions_vector_.resize(right_side_vector_.n_elem);
    for(auto&& x_i : solutions_vector_)
        x_i = 0;
}

void Solver::initSolver()
{
    setPreconditioner();
    if(isDiagZeroVector())
        throwError(std::string{"Diagonal of cooficients is zero vector !"});
}

bool Solver::isFirstIteration() const
{
    return iteration_ == 0;
}

arma::mat Solver::getSubstep() const
{
    return right_side_vector_ - cooficient_matrix_ * solutions_vector_;
}

arma::mat Solver::getStep() const
{
    return preconditioner_ * getSubstep();
}

void Solver::iterationEngine()
{
    if(dynamic_relaxing_flag_ == true)
        modifyRelax();
    solutions_vector_ += relax_ *  getStep();
}

double Solver::getNumerator() const
{
    return arma::dot(arma::trans( getSubstep()), cooficient_matrix_ *  getStep());
}
double Solver::getEnumerator() const
{
    return arma::dot(arma::trans(cooficient_matrix_ *  getStep()), 
                                 cooficient_matrix_ *  getStep());
}

void Solver::modifyRelax()
{
    relax_ =  getNumerator() / getEnumerator();
}

bool Solver::isDiagZeroVector() const
{
    for(auto&& e: preconditioner_)
        if(e != 0) return false;
    return true;
}

void Solver::throwError(std::string msg) const noexcept(false) 
{
    throw std::runtime_error(msg);
}

Solver::Solver(arma::mat const& cooficient_matrix,
               arma::colvec const& right_side_vector,
               bool enable_dynamic_relax ,
               double start_relax,
               arma::colvec const& start_positions_vector):
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

void Solver::operator() ()
{
    if(isFirstIteration())
        initSolver();
    iterationEngine();
    ++iteration_;
}

bool Solver::hasGoodPrecision(double left,double right,double precision)
{
    return fabs(left-right) < precision;
}

bool Solver::isInsufficientPrecision(double precision,arma::colvec const& temp)
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

void Solver::operator() (size_t count)
{
    for(size_t i{0}; i < count; ++i)
        (*this)();
} 

void Solver::operator() (double precision)
{
    arma::colvec temp {solutions_vector_};
    (*this)();
    while (isInsufficientPrecision(precision,temp))
    {
        temp = solutions_vector_;
        (*this)();
    }
}

arma::colvec Solver::getSolutions()
{
    return solutions_vector_;
}

size_t Solver::getIteration() const
{
    return iteration_;
}

void JacobiSolver::setPreconditioner()
{
    //get diagonal of matrix
    preconditioner_ =  arma::diagmat(cooficient_matrix_).i();
}

void GaussSeidelSolver::setPreconditioner()
{
    //get lower triangular parts
    preconditioner_ = arma::trimatl(cooficient_matrix_,0);
    preconditioner_ = arma::inv(preconditioner_);
}

