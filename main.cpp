#include <iostream>
#include "include/EquationSolvers.hpp"
#include "include/MesLib.h"
using namespace std;
using namespace arma;

template<typename MatrixType>
void assembly(MatrixType& K_glob, int mx, int my,int n, int node_matrix_size)
{
	//Assemblacja + "umasowienie" macierzy
	for (size_t elx{ 0 }; elx < mx; ++elx)
		for (size_t ely{ 0 }; ely < my; ++ely) //dla ka¿dego wêz³a...
			for (size_t dim1{ 0 }; dim1 < node_matrix_size; ++dim1)
				for (size_t dim2{ 0 }; dim2 < node_matrix_size; ++dim2)
					K_glob.at(DOF(elx, ely, dim1, mx), DOF(elx, ely, dim2, mx)) += K[dim1][dim2];
	// redystrybucja elementów lokalnych macierzy do macierz globalnej
	// z wykorzystaniem DOF

	for (size_t i{ 0 }; i < n; ++i)
		for (size_t j{ 0 }; j < n; ++j)
			K_glob.at(i, j) *= Md;
}

void freeAllNodes(std::vector<bool>& fix)
{
	//na pocz¹tku ca³y uk³ad bêdzie uwolniony
	for (auto&& elem : fix)
		elem = false;
}

void blockNodes(std::vector<bool>& fix,int mx)
{
	//wprowadzam trzy podpory
	fix[0] = true;
	fix[1] = true;
	fix[mx] = true;
	fix[mx + 1] = true;
	fix[2 * mx] = true;
	fix[2 * mx + 1] = true;
}

template<typename VectorType>
void setForces(VectorType& forces, int n)
{
	for (size_t i{ 0 }; i < n; ++i)
		if (i % 2 != 0)
			forces[i] = -400000000;
}

template<typename MatrixType,typename VectorType>
void applyLocksToMatrix(MatrixType& K_glob, std::vector<bool>& fix, VectorType& forces,int n)
{
	//Narzuæ warunki brzegowe tzn. zerujê si³y w wêz³ach podporowych oraz
	//modyfikujê odpowiednio macierz
	for (size_t i{ 0 }; i < n; ++i)//dla ka¿dej kolumny
		if (fix[i])// sprawd ,czy wêze³ utwierdzony
		{
			for (size_t j{ 0 }; j < n; ++j)
				K_glob.at(i, j) = 0;//zerowanie kolumn
			K_glob.at(i, i) = 1;//na diagonali 1 
			forces[i] = 0;
		}
}

template<typename VectorType>
void resizeAll(VectorType& forces, std::vector<bool>& fix,int n)
{
	// alokacja pamiêci
	fix.resize(n);
}

//JEŚLI COŚ NIE DZIAŁA ,TO ZAMIEŃ NA DENSITY
int main()
{
    std::cout << "MES_START\n";

    //wymiary siatki
	int const mx{ 10 };
	int const my{ 20 };

    size_t node_matrix_size{ 8 };	//rozmiar macierzy K

    int  n{ 2 * (mx + 1) * (my + 1) };
    
    
    colvec shifts{};	//wektor przesuniêæ
	colvec forces(n);	//wektor si³
	std::vector<bool> fix{};

    resizeAll(forces, fix, n);
    freeAllNodes(fix);
    blockNodes(fix, mx);

    DenseMatrix K_glob(n,n);
   
    assembly(K_glob, mx, my, n, node_matrix_size);
    setForces(forces, n);
	applyLocksToMatrix(K_glob, fix, forces, n);

    //JacobiSolver<DenseMatrix,colvec> solver{K_glob,forces};
    //GaussSeidelSolver<DenseMatrix,colvec> solver{K_glob,forces,false};
    
    //solver();
    //solver((size_t)10);
    //solver(0.1);
    //shifts = solver.getSolutions();
    //std::cout << solver.getIteration() << std::endl;
    
    // col_i * A_
    //A = A^T
    //x^T *(skalar) A *(wektor) x > 0 

    std::cout << "MES_END\n\n\n";

    std::cout << "wartosci wlasne\n\n\n";

    mat BD(3,3);

    BD.at(0,0) = 3;
    BD.at(0,1) = 0;
    BD.at(0,2) = 3;
    BD.at(1,0) = 3;
    BD.at(1,1) = 2;
    BD.at(1,2) = 1;
    BD.at(2,0) = 4;
    BD.at(2,1) = 0;
    BD.at(2,2) = 2;

    std::cout << BD << std::endl;

    cx_vec eigval;

    eig_gen(eigval,BD); 

    std::cout <<  std::abs(eigval.at(0)) << std::endl;

    std::cout << eigval << std::endl;
    std::cout << "end_wartosci_wlasne\n\n\n";

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

    JacobiSolver<DenseMatrix,DenseVector> g{A,V,true};
    g();
    //g((size_t)10);
    g(0.001);
    cout << g.getSolutions() << endl; 
    cout << "Liczba iteracji : " << g.getIteration() << std::endl; 
    // GaussSeidelSolver<SparseMatrix,SparseVector> j{A,V,false};
    // j();
    // j((size_t)10);
    // cout << j.getSolutions() << endl; 
    
    

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
