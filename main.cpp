#include <iostream>
#include <armadillo>
#include "include/EquationSolvers.hpp"
using namespace std;
using namespace arma;

int main()
{
    mat A(4, 4);
    //manual
    //row-col 
    A.at(0,0) = 4;
    A.at(1,0) = -1;
    A.at(2,0) = 0.2;
    A.at(3,0) = 0;
    A.at(0,1) = -1;
    A.at(1,1) = 5;
    A.at(2,1) = 1;
    A.at(3,1) = -2;
    A.at(0,2) = -0.2;
    A.at(1,2) = 0;
    A.at(2,2) = 10;
    A.at(3,2) = -1;
    A.at(0,3) = 2;
    A.at(1,3) = -2;
    A.at(2,3) = -1;
    A.at(3,3) = 4;
    //
    colvec V(4);
    //manual
    V.at(0,0) = 30;
    V.at(1,0) = 0;
    V.at(2,0) = -10;
    V.at(3,0) = 5;
    //
    cout << A << endl;
    cout << V << endl;
    cout << arma::trimatl(A,0) << endl;

    colvec start(4);
    start.at(0,0) = 1;
    start.at(1,0) = 1;
    start.at(2,0) = 1;
    start.at(3,0) = 1;


    GaussSeidelSolver<SparseMatrix,SparseVector> j{A,V,true};
    j();
    j((size_t)10);
    cout << j.getSolutions() << endl; 
    
    JacobiSolver<DenseMatrix,DenseVector> g{A,V};
    g();
    g();
    cout << g.getSolutions() << endl; 

    mat U  = trimatu(A);
    mat L  = trimatl(A);

    mat UU = trimatu(A,  1);  // omit the main diagonal
    mat LL = trimatl(A, -1);  // omit the main diagonal
    cout << A << endl;
    cout << LL << endl;
    cout << UU << endl;
    mat C(4, 5, fill::randu);
    mat B(4, 5, fill::randu);
    
    cout << C*B.t() << endl;
    cout << A.diag() << endl;
    cout << diagmat(A);
    return 0;
}
