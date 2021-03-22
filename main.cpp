#include <iostream>
#include <armadillo>
#include "include/EquationSolvers.hpp"
using namespace std;
using namespace arma;

int main()
{
    mat A(5, 5, fill::randu);
    colvec V(5,fill::randu);
    JacobiSolver j{A,V};

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
    
    return 0;
}
