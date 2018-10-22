#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <omp.h>
#include <mkl.h>
// #include <mkl_pblas.h>
// #include <mkl_scalapack.h>
// #include <mkl_blacs.h>

int omp_thread_cout(){
    int n=0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;   
}

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;

int main (int argc, char *argv[]){

	//El::Environment env (argc, argv);
    //const El::Int size = El::Input("--size", "Size of the matrices to test", 100);
    //El::ProcessInput();
    int size = 4000;
    
    std::chrono::time_point<std::chrono::system_clock> start, end;

    std::cout << "Matrix size : " << size << "x" << size << std::endl;
    std::cout << "Num Threads: " <<  omp_thread_cout() << std::endl;    

    //Eigen::initParallel();
	Mat A = Mat::Random(size,size);
    Mat B = Mat::Random(size,size);
    Mat C = Mat::Zero(size,size);

    start = std::chrono::system_clock::now();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size,size,size,double(1.),&A(0),size,&B(0),size,double(0.),&C(0),size);
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_time = end-start;
    std::cout << "Run time    : " << elapsed_time.count() << " secs" <<  std::endl;

	return(0);

}

