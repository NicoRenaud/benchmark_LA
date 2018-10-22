// taken from https://scicomp.stackexchange.com/questions/7766/performance-optimization-or-tuning-possible-for-scalapack-gemm

#include <cstdio>
#include <iostream>
#include <cmath>
using namespace std;

#include "mycblas.h"
#include "utils/NanoTimer.h"
#include "utils/stringhelper.h"
#include "args.h"

extern "C" {
    void openblas_set_num_threads(int num_threads);
}

int main( int argc, char *argv[] ) {
    int N, its, threads;
    Args( argc, argv ).arg("N", &N).arg("its", &its).arg("threads",&threads).go();

    openblas_set_num_threads( threads );    

    NanoTimer timer;
    double *A = (double*)malloc(sizeof(double)*N*N);
    double *B = (double*)malloc(sizeof(double)*N*N);
    int linsize = N * N;
    for( int i = 0; i < linsize; i++ ) {
        A[i] = i + 3;
        B[i] = i * 2;
    }
    int m = N;
    int n = N;
    int k = N;
    double alpha = 1;
    double beta = 0;
    double *C = (double*)malloc(sizeof(double)*N*N);
    timer.toc("setup input matrices");
    for( int it = 0; it < its; it++ ) {
        dgemm(false,false,N,N,N, 1, A, N, B, N, 0, C, N );
        timer.toc("it " + toString(it) );
    }
    int sum = 0;
    for( int mult = 0; mult < log(N)/log(10); mult++ ) {
        int offset = pow(10,mult);
        sum += C[offset];
    }
    cout << "sum, to prevent short-cut optimization " << sum << endl;
    return 0;
}