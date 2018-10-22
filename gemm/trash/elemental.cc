/*
        Benchmark the GEMM operation using :
        - Elemental 
        - Scalapack 
        - Eigen

*/

#include <El.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <string>

using namespace std;
using std::cout;
using std::endl;


extern "C"{
        void openblas_set_num_threads(int num_threads);
}

int main (int argc, char *argv[])
{

        El::Environment env (argc, argv);
        El::mpi::Comm comm = El::mpi::COMM_WORLD;
        const El::Int commRank = El::mpi::Rank(comm);
        const El::Int commSize = El::mpi::Size(comm);

        try {

                const El::Grid grid(comm);       
                const El::Int size = El::Input("--size", "Size of the matrices to test", 100);
                const El::Int blocksize = El::Input("--blocksize","algorithmic blocksize",96);
                const bool print = El::Input("--print","print matrices?",false);
                El::ProcessInput();
                El::PrintInputReport();

                El::SetBlocksize( blocksize );


                El::Matrix<double> A, B, C;
                El::Uniform(A,size, size);
                El::Uniform(B,size, size);
                El::Zeros(C,size, size);
                El::mpi::Barrier( comm );

                El::Timer timer;
                
                if (commRank == 0)
                {

                        timer.Start();
                        El::Gemm(El::NORMAL,El::NORMAL, double(1.),A,B,double(0.),C);
                        std::cout << "Multithreaded time : " << timer.Stop() << " secs" << std::endl;

                        openblas_set_num_threads(1);
                        timer.Start();
                        El::Gemm(El::NORMAL,El::NORMAL, double(1.),A,B,double(0.),C);
                        std::cout << "Sequential time : " <<    timer.Stop() << " secs" << std::endl;

                }

        }

        catch(std::exception &e) { El::ReportException(e); }

}