//------------------------------- Bardzo pomocne funkcje
#include <vector>
#include <iostream>

int P(int x, int y, int z,int const mx)
{
	return (y*(mx + 1) + x) * 2 + z;
}

int Q(int x, int y,int const mx)
{
	return y*mx + x;
}

int DOF(int elidx, int elidy, int locdofid,int const mx)
{
	if (locdofid < 4)
		return 2 * (elidy*(mx + 1) + elidx) + locdofid;
	else if (locdofid == 4 || locdofid == 5)
		return 2 * ((elidy + 1)*(mx + 1) + elidx - 1) + locdofid;
	else if (locdofid == 6 || locdofid == 7)
		return 2 * ((elidy + 1)*(mx + 1) + elidx - 3) + locdofid;
}

const double skala = 1.0;

//------------------------------- MACIERZ SZTYWNOSCI ELEMENTU --------------------------
const double E = 210*pow(10,9);
const double nu = 0.3;
const double elems[] = {
	0, 1. / 2. - nu / 6., 1. / 8. + nu / 8., -1. / 4. - nu / 12., -1. / 8. + 3.*nu / 8.,
	-1. / 4. + nu / 12., -1. / 8. - nu / 8., nu / 6., 1. / 8. - 3.*nu / 8.
};
const double Md = E / (1 - nu*nu);

const double K[8][8] = { { elems[1], elems[2], elems[3], elems[4], elems[5], elems[6], elems[7], elems[8] },
{ elems[2], elems[1], elems[8], elems[7], elems[6], elems[5], elems[4], elems[3] },
{ elems[3], elems[8], elems[1], elems[6], elems[7], elems[4], elems[5], elems[2] },
{ elems[4], elems[7], elems[6], elems[1], elems[8], elems[3], elems[2], elems[5] },
{ elems[5], elems[6], elems[7], elems[8], elems[1], elems[2], elems[3], elems[4] },
{ elems[6], elems[5], elems[4], elems[3], elems[2], elems[1], elems[8], elems[7] },
{ elems[7], elems[4], elems[5], elems[2], elems[3], elems[8], elems[1], elems[6] },
{ elems[8], elems[3], elems[2], elems[5], elems[4], elems[7], elems[6], elems[1] } };

//------------------------------- MACIERZ MASOWA ELEMENTU --------------------------
const double rho = 1;
const double Mm = rho;

const double M[8][8] = { { 4, 0, 2, 0, 1, 0, 2, 0 },
{ 0, 4, 0, 2, 0, 1, 0, 2 },
{ 2, 0, 4, 0, 2, 0, 1, 0 },
{ 0, 2, 0, 4, 0, 2, 0, 1 },
{ 1, 0, 2, 0, 4, 0, 2, 0 },
{ 0, 1, 0, 2, 0, 4, 0, 2 },
{ 2, 0, 1, 0, 2, 0, 4, 0 },
{ 0, 2, 0, 1, 0, 2, 0, 4 } };

