#include <El.hpp>
#include <iostream>
#include <string>
#include <cblas.h>

using namespace std;
using std::cout;
using std::endl;


int omp_thread_cout(){
    int n=0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;   
}

int main (int argc, char *argv[])
{

        El::Environment env (argc, argv);
        El::mpi::Comm comm = El::mpi::COMM_WORLD;
        const El::Int commRank = El::mpi::Rank(comm);
        const El::Int commSize = El::mpi::Size(comm);

        if (commSize > 1){
                El::Output("Error : Use a single process for this test !");
                return(0);
        }
        

        try {

                const El::Grid grid(comm);       
                const El::Int size = El::Input("--size", "Size of the matrices to test", 100);
                const El::Int blocksize = El::Input("--blocksize","algorithmic blocksize",96);

                El::ProcessInput();
                El::SetBlocksize( blocksize );

                if (commRank == 0)
                {
                        El::Output(" Matrix size : ", size, "x", size);
                        El::Output(" Num Procs   : ", commSize);
                
                        int nthreads = openblas_get_num_threads();
                        El::Output(" Num Threads : ", nthreads);

                        El::Matrix<double> A, B, C;
                        El::Uniform(A,size, size);
                        El::Uniform(B,size, size);
                        El::Zeros(C,size, size);
                        El::mpi::Barrier( comm );

                        El::Timer timer;
                
                        timer.Start();
                        El::Gemm(El::NORMAL,El::NORMAL, double(1.),A,B,double(0.),C);
                        El::Output(" Time : ", timer.Total(), " secs");
                }

        }

        catch(std::exception &e) { El::ReportException(e); }

}