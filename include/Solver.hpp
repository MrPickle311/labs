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
                    Vector const& solutions_vector);
};

using CommonResources = std::shared_ptr<SharedResources>;

class SupportObject
{
protected:
    CommonResources res_;
public:
    SupportObject(){}
    void setRes(CommonResources  res);
};

class PrecisionChekcer:
    public SupportObject
{
public:
    using SupportObject::SupportObject;

    inline bool hasGoodPrecision(double left,double right,double precision)
    {
        return ( fabs( left - right ) / left ) < precision;
    }
    
    bool isInsufficientPrecision(double precision,Vector const& temp);
};

class Solver
{
protected:
    CommonResources res_; 
    PrecisionChekcer precision_checker_;
    size_t iteration_;
    double relax_;
    bool timeout_occured_;
public:
    Solver(Matrix cooficient_matrix,
           Vector right_side_vector,
           Vector start_position_vector = {},
           double start_relax = 1.0);
    inline bool isFirstIteration() const
    {
        return iteration_ == 0;
    }

    virtual void initSolver() = 0; // hook 
    virtual void iterationEngine() = 0; // hook 

    void operator() ();
    void operator() (size_t count); // several iterations 

    double operator() (double precision,double timeout = 0.1);//counting while all |x_i+1 - x_i| / x_i+1 < precision

    Vector getResults();

    size_t getIteration() const;

    bool getTimeoutState() const;
};

