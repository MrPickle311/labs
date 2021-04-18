#pragma once
#include <armadillo>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

struct EquationPackage
{
    arma::mat cooficient_matrix_;
    arma::mat right_side_vector_;
};

class MatrixStream
{
    std::filesystem::path eq_path_;
public:
    MatrixStream(std::filesystem::path relative_eq_path);

    friend EquationPackage& operator << (EquationPackage& matrix_package, 
                                         MatrixStream matrix_stream)
    {
        std::ifstream eq_file;

        eq_file.open(matrix_stream.eq_path_);

        if(!eq_file.is_open())
            std::runtime_error{"Bad directory!"};

        eq_file.seekg(0);

        size_t matrix_size = 2;
        eq_file >> matrix_size;


        matrix_package.cooficient_matrix_.clear();
        matrix_package.cooficient_matrix_.resize(matrix_size,matrix_size);

        for(size_t i{0}; i < matrix_size ; ++i)
            for(size_t j{0}; j < matrix_size ; ++j)
                eq_file >> matrix_package.cooficient_matrix_.at(i,j);
        
        matrix_package.right_side_vector_.clear();
        matrix_package.right_side_vector_.resize(matrix_size);

        for(size_t i{0}; i < matrix_size ; ++i)
            eq_file >> matrix_package.right_side_vector_.at(i);

        return matrix_package;
    }
};

