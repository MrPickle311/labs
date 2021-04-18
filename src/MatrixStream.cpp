#include "../include/MatrixStream.hpp"


MatrixStream::MatrixStream(std::filesystem::path relative_eq_path):
        eq_path_{relative_eq_path}
{       
    if(!std::filesystem::exists(eq_path_))
        std::runtime_error{"Bad directory!"};
}
