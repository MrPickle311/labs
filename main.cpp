#include <iostream>
#include "include/GradientSolvers.hpp"
#include "include/IterativeSolvers.hpp"
#include "include/MatrixStream.hpp"
#include <vector>

using namespace std;
using namespace arma;

struct TestData
{
    double              time_;
    size_t              iterations_;
    std::string         name_;
    bool                dynamic_relax_enabled_ = false;
    double              start_relax_ = 1.0; 
    bool                timeout_occured;
};

TestData start_benchmark(Solver* solver,
                         std::string test_name,
                         double norm_precision,
                         double timeout = 0.1,
                         bool dynamic_relax = false,
                         double start_relax_ = 1.0,
                         bool show_results = false)
{
    TestData data;
    data.name_              = test_name;
    data.time_              = (*solver)(norm_precision,timeout) * 1000;
    data.iterations_        = solver->getIteration();
    data.timeout_occured    = solver->getTimeoutState();
    data.start_relax_       = start_relax_;

    std::cout << "Method: " << data.name_  << std::endl;
    std::cout << "Time of solving : " << data.time_ << " ms" << std::endl;
    
    if(show_results)
        std::cout << "Results:\n" << solver->getResults() << std::endl;
    
    if(dynamic_relax)
        std::cout << "Dynamic relax enabled with value " << start_relax_ << std::endl;

    std::cout << "Iterations count: " << data.iterations_ << std::endl;

    if(data.timeout_occured)
        std::cout << " **** TIMEOUT ****" << std::endl;
    std::cout << std::endl;
    return data;
}

std::vector<TestData> gradient_benchmark_set(EquationPackage const& pack,
                                             double norm_precision,
                                             double min_relax_value  = 0.99 ,
                                             double max_relax_value = 1.01,
                                             double timeout = 0.1)
{
    std::vector<TestData> data;

    //without preconditiner
    for(double relax{min_relax_value}; relax < max_relax_value ; relax += 0.01)
    {
        GradientSolver grad{pack.cooficient_matrix_,pack.right_side_vector_,false,relax};
        data.push_back(start_benchmark(&grad,"Gradient method without preconditioner",norm_precision,timeout,true,relax));
    }

    //with preconditioner
    for(double relax{min_relax_value}; relax < max_relax_value ; relax += 0.01)
    {
        GradientSolver grad{pack.cooficient_matrix_,pack.right_side_vector_,true,relax};
        data.push_back(start_benchmark(&grad,"Gradient method with preconditioner",norm_precision,timeout,true,relax));
    }

    return data;
}

std::vector<TestData> jacobi_benchmark_set(EquationPackage const& pack,
                                                           double norm_precision,
                                                           double min_relax_value  = 0.99 ,
                                                           double max_relax_value = 1.01,
                                                           double timeout = 0.1)
{
    std::vector<TestData> data;

    //without dynamic relax
    for(double relax{min_relax_value}; relax < max_relax_value ; relax += 0.01)
    {
        JacobiSolver jacobi{pack.cooficient_matrix_,pack.right_side_vector_};
        data.push_back(start_benchmark(&jacobi,"Jacobi method without dynamic relax",norm_precision,timeout));
    }

    //with dynamic relax
    for(double relax{min_relax_value}; relax < max_relax_value ; relax += 0.01)
    {
        JacobiSolver jacobi{pack.cooficient_matrix_,pack.right_side_vector_,false,relax};
        data.push_back(start_benchmark(&jacobi,"Jacobi method with dynamic relax",norm_precision,timeout,true,relax));
    }

    return data;
}

std::vector<TestData> gauss_siedel_benchmark_set(EquationPackage const& pack,double norm_precision,
                                                                 double min_relax_value  = 0.99 ,
                                                                 double max_relax_value = 1.01,
                                                                 double timeout = 0.1)
{
    std::vector<TestData> data;

    //without dynamic relax
    for(double relax{min_relax_value}; relax < max_relax_value ; relax += 0.01)
    {
        GaussSeidelSolver gauss{pack.cooficient_matrix_,pack.right_side_vector_};
        data.push_back(start_benchmark(&gauss,"Gauss-Seidel method without dynamic relax",norm_precision,timeout));
    }

    //with dynamic relax
    for(double relax{min_relax_value}; relax < max_relax_value ; relax += 0.01)
    {
        GaussSeidelSolver gauss{pack.cooficient_matrix_,pack.right_side_vector_,false,relax};
        data.push_back(start_benchmark(&gauss,"Gauss-Seidel method with dynamic relax",norm_precision,timeout,true,relax));
    }

    return data;
}

std::string parse_test_data(TestData const& data)
{
    std::string str_data;
    std::ostringstream str{str_data};
    
    str << data.name_ << '\t';//1

    str << data.start_relax_ << '\t';//4
    
    if(data.timeout_occured)//5
        str << "Timeout occured!" << '\t';
    else str <<"Compute complete" << '\t';
    
    

    str <<  data.time_ << '\t';//6
    str <<  data.iterations_ << '\t';

    str << '\n';
    
    str_data = str.str();
    
    return str_data;
}

std::string parse_test_vector(std::vector<TestData> data)
{
    std::string str_data;
    std::ostringstream str{str_data};

    for(auto&& e : data)
        str << parse_test_data(e);

    str_data = str.str();

    return str_data;
}

void generateReport(std::string dat , std::filesystem::path dir)
{
    std::ofstream file (dir, ios_base::trunc);
    if(file.is_open())
    {
        std::cout << "Saving a report..." << std::endl;
        file << dat;
    }
    else std::cout << "Cannot open a file " << std::endl;
}

int main()
{
    MatrixStream str{ std::filesystem::current_path() / "../extern/eq1.txt"};

    EquationPackage pack;
    pack << str;

    std::cout << pack.cooficient_matrix_ << std::endl; 
    std::cout << pack.right_side_vector_ << std::endl;

    double norm_precision = 0.0000000000000001;//norma 

    std::vector<TestData>  jacobi_set {jacobi_benchmark_set(pack,norm_precision,0.1,2.0)};
    std::vector<TestData>  gauss_seidel_set {gauss_siedel_benchmark_set(pack,norm_precision,0.1,2.0)};
    std::vector<TestData>  gradient_set {gradient_benchmark_set(pack,norm_precision,0.1,2.0)};

    std::string jacobi_dat {parse_test_vector(jacobi_set)};
    std::cout << jacobi_dat ;
    std::cout << std::endl;

    std::string gauss_dat {parse_test_vector(gauss_seidel_set)};
    std::cout << gauss_dat ;
    std::cout << std::endl;

    std::string gradient_dat {parse_test_vector(gradient_set)};
    std::cout << gradient_dat ;
    std::cout << std::endl << std::endl;

    std::filesystem::path save_dir {std::filesystem::current_path() / "../extern"};

    generateReport(jacobi_dat,save_dir / "jacobi_report.txt");
    generateReport(gauss_dat,save_dir / "gauss_report.txt");
    generateReport(gradient_dat,save_dir / "gradient_report.txt");




    MatrixStream str2{ std::filesystem::current_path() / "../extern/eq2.txt"};

    EquationPackage pack2;
    pack2 << str2;

    std::vector<TestData>  jacobi_set2 {jacobi_benchmark_set(pack2,norm_precision,0.1,2.0,1)};
    std::vector<TestData>  gauss_seidel_set2 {gauss_siedel_benchmark_set(pack2,norm_precision,0.1,2.0,1)};
    std::vector<TestData>  gradient_set2 {gradient_benchmark_set(pack2,norm_precision,0.1,2.0,1)};

    
    std::string jacobi_dat2 {parse_test_vector(jacobi_set2)};

    generateReport(jacobi_dat2,save_dir / "jacobi_report2.txt");

    std::string gauss_dat2 {parse_test_vector(gauss_seidel_set2)};

    generateReport(gauss_dat2,save_dir / "gauss_report2.txt");

    std::string gradient_dat2 {parse_test_vector(gradient_set2)};
    
    generateReport(gradient_dat2,save_dir / "gradient_report2.txt");

    return 0;
}
