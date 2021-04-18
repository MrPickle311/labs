#include "../include/Solver.hpp"

//SharedResources//

SharedResources::SharedResources(Matrix const& cooficient_matrix,
                                 Vector const& right_side_vector,
                                 Vector const& solutions_vector):
        cooficient_matrix_{cooficient_matrix},
        right_side_vector_{right_side_vector},
        solutions_vector_{right_side_vector},
        preconditioner_{}
    {}




//SupportObject//

void SupportObject::setRes(CommonResources  res)
{
         res_ = std::move(res);
}



//PrecisionChekcer//

bool PrecisionChekcer::isInsufficientPrecision(double precision,Vector const& temp)
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



//Solver//

Solver::Solver(Matrix cooficient_matrix,
               Vector right_side_vector,
               Vector start_position_vector,
               double start_relax):
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

void Solver::operator() ()
{
    if(isFirstIteration());
        initSolver();
    iterationEngine();
    ++iteration_;
}

void Solver::operator() (size_t count)
{
    for(size_t i{0}; i < count; ++i)
        (*this)();
} 

double Solver::operator() (double precision)

{
    auto start = std::chrono::steady_clock::now();
    
    (*this)();
    Vector temp {res_->solutions_vector_};
    
    
    while (precision_checker_.isInsufficientPrecision(precision,temp))
    {
        temp = res_->solutions_vector_;
        (*this)();
        //std::cout << arma::norm(res_->solutions_vector_ - temp) << " : NORM\n";
    }
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> seconds {end - start};
   
    return seconds.count();
}

Vector Solver::getResults()
{
    return res_->solutions_vector_;
}

size_t Solver::getIteration() const
{
    return iteration_;
}