#include <El.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <chrono>

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;

int main (int argc, char *argv[]){

	El::Environment env (argc, argv);
    const El::Int size = El::Input("--size", "Size of the matrices to test", 100);
    El::ProcessInput();

    Eigen::initParallel();
	Mat A = Mat::Random(size,size);
    Mat B = Mat::Random(size,size);
    Mat C = Mat::Zero(size,size);

    std::cout << "Matrix size : " << size << "x" << size << std::endl;
    std::cout << "Num Threads : " <<  Eigen::nbThreads() << std::endl;

    El::Timer timer;
    timer.Start();
    C = A*B;
    timer.Stop();
    std::cout << "Run time    : " << timer.Total() << " secs" <<  std::endl;

	return(0);

}

