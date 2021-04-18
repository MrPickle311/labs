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
    if(arma::norm(res_->solutions_vector_ - temp) >= precision)
        return true;

    return false;//bad solutions
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
        relax_{start_relax},
        timeout_occured_{false}
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

double Solver::operator() (double precision,double timeout)

{
    auto start = std::chrono::steady_clock::now();

    Vector temp {res_->solutions_vector_};
    timeout_occured_ = false;

    (*this)();

    while (precision_checker_.isInsufficientPrecision(precision,temp))
    {
        std::chrono::duration<double> tim {std::chrono::steady_clock::now() - start};
        if (timeout < tim.count())
        {
            timeout_occured_ = true;
            break;
        }
        temp = res_->solutions_vector_;
        (*this)();
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

bool Solver::getTimeoutState() const
{
    return timeout_occured_;
}