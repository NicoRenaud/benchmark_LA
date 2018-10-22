#include <iostream>
#include <cmath>
#include <mpi.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "nanotimer.h"
//#include "args.h"
#include "stringhelper.h"

/*
main example
 https://scicomp.stackexchange.com/questions/7766/performance-optimization-or-tuning-possible-for-scalapack-gemm
Also 
https://github.com/cjf00000/tests/blob/master/mkl/pblas3_d_example.c
Alseo
https://software.intel.com/en-us/mkl-developer-reference-c
*/

int getRootFactor( int n ) {
    for( int t = sqrt(n); t > 0; t-- ) {
        if( n % t == 0 ) {
            return t;
        }
    }
    return 1;
}

extern "C" {
    struct DESC{
        int DTYPE_;
        int CTXT_;
        int M_;
        int N_;
        int MB_;
        int NB_;
        int RSRC_;
        int CSRC_;
        int LLD_;
    } ;
}    

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;

// conventions:
// M_ by N_ matrix block-partitioned into MB_ by NB_ blocks, then
// distributed according to 2d block-cyclic scheme

// based on http://acts.nersc.gov/scalapack/hands-on/exercise3/pspblasdriver.f.html

int main( int argc, char *argv[] ) {

    MKL_INT myrank, commSize;
    blacs_pinfo_( &myrank, &commSize );

    MKL_INT size = 100;
    int numthreads = 1;

    MKL_INT nprows = getRootFactor( commSize );
    MKL_INT npcols = commSize / nprows;
    if( myrank == 0 ) std::cout << "grid: " << nprows << " x " << npcols << std::endl;

    // get the blacs context
    MKL_INT i_negone = -1, i_zero = 0, i_one = 1;
    const double zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0E+0;
    const char trans = 'N';
    MKL_INT ictxt, info;
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "C", &commSize, &i_one );

    if( myrank == 0 ) std::cout << "system context " << ictxt << " grid context: " << ictxt << std::endl;

    MKL_INT myrow, mycol;
    blacs_gridinfo_( &ictxt, &nprows, &npcols, &myrow, &mycol );


    if( myrow >= nprows || mycol >= npcols ) {
        //mpi_print("not needed, exiting");
        blacs_gridexit_( &ictxt );
        blacs_exit(0);
        exit(0);
    }

    // A     B       C
    // m x k k x n = m x n
    // nb: blocksize

    // nprows: process grid, number rows
    // npcols: process grid, number cols
    // myrow: process grid, our row
    // mycol: process grid, our col

    MKL_INT  m = size;
    MKL_INT k = size;

    double *ptr_Alocal, *ptr_Blocal;

    if( (myrow == 0) && (mycol == 0) ) 
    {
        //Eigen::initParallel();
        Mat Alocal = Mat::Random(size,size);
        Mat Blocal = Mat::Random(size,size);

        double *ptr_Alocal = &Alocal(0);
        double *ptr_Blocal = &Blocal(0);
    } else
    {
        double *ptr_Alocal = NULL;
        double *ptr_Blocal = NULL;
    }

    // descritor for local matrices
    MKL_INT desca_local[9], descb_local[9], descc_local[9];
    MKL_INT lld_local = std::max((int) numroc_(&size,&size,&myrow,&i_zero,&nprows),1);
    descinit( desca_local, &size, &size, &size, &size, &i_zero, &i_zero, &ictxt, &lld_local, &info );
    descinit( descb_local, &size, &size, &size, &size, &i_zero, &i_zero, &ictxt, &lld_local, &info );
    descinit( descc_local, &size, &size, &size, &size, &i_zero, &i_zero, &ictxt, &lld_local, &info );
    

    // nb is column block size for A, and row blocks size for B
    MKL_INT NB_MAX = 128;
    MKL_INT nb = std::min(size/commSize,NB_MAX);

    int mp = numroc_( &size, &nb, &myrow, &i_zero, &nprows ); // mp number rows owned by this process
    int nq = numroc_( &size, &nb, &mycol, &i_zero, &npcols ); // nq number cols owned by this process
    double *Adist = static_cast<double*>(std::malloc(mp*nq*sizeof(double)));
    double *Bdist = static_cast<double*>(std::malloc(mp*nq*sizeof(double)));
    double *Cdist = static_cast<double*>(std::malloc(mp*nq*sizeof(double)));
    

    // descriptor distributed matrices
    MKL_INT desca_dist[9], descb_dist[9], descc_dist[9];
    MKL_INT lld = std::max(mp,1);    
    descinit_(desca_dist, &size, &size, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info);
    descinit_(descb_dist, &size, &size, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info);
    descinit_(descc_dist, &size, &size, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info);

    // Distribute the matrices from process 0 over the grid
    pdgeadd_(&trans, &size, &size, &one, ptr_Alocal, &i_one, &i_one, desca_local, &zero, Adist, &i_one, &i_one, desca_dist);
    //pdgeadd_(&trans, &size, &size, &one, ptr_Blocal, &i_one, &i_one, descb_local, &zero, Bdist, &i_one, &i_one, descb_dist);

    if (myrank == 0)
        std::cout << "Arrays are distributed with pdgeadd" << std::endl;

/*
    // double *ipa = new double[desca.LLD_ * kq];
    // double *ipb = new double[descb.LLD_ * nq];
    // double *ipc = new double[descc.LLD_ * nq];
*/
    //blacs_gridexit( &ictxt );
    //blacs_exit(0);
    return(0);
}

//     for( int i = 0; i < desca.LLD_ * kq; i++ ) {
//         ipa[i] = p;
//     }
//     for( int i = 0; i < descb.LLD_ * nq; i++ ) {
//         ipb[i] = p;
//     }

//     if( p == 0 ) std::cout << "created matrices" << std::endl;
//     double *work = new double[nb];
//     if( n <=5 ) {
//         pdlaprnt( n, n, ipa, 1, 1, &desca, 0, 0, "A", 6, work );
//         pdlaprnt( n, n, ipb, 1, 1, &descb, 0, 0, "B", 6, work );
//     }

//     NanoTimer timer;
//     pdgemm( false, false, m, n, k, 1,
//                   ipa, 1, 1, &desca, ipb, 1, 1, &descb,
//                   1, ipc, 1, 1, &descc );
//     MPI_Barrier( MPI_COMM_WORLD );
//     if( p == 0 ) timer.toc("pdgemm");

//     blacs_gridexit( grid );
//     blacs_exit(0);

//     return 0;
// }



























//     void blacs_pinfo_( int *iam, int *nprocs );
//     void blacs_get_( int *icontxt, int *what, int *val );
//     void blacs_gridinit_( int *icontxt, char *order, int *nprow, int *npcol );
//     void blacs_gridinfo_( int *context, int *nprow, int *npcol, int *myrow, int *mycol );
//     void blacs_gridexit_( int *context );
//     void blacs_exit_( int *code );

//     int numroc_( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs );
//     void descinit_( struct DESC *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info );
//     void pdlaprnt_( int *m, int *n, double *a, int *ia, int *ja, struct DESC *desca, int *irprnt,
//         int *icprnt, const char *cmatnm, int *nout, double *work, int cmtnmlen );
//     void pdgemm_( char *transa, char *transb, int *m, int *n, int *k, double *alpha,
//          double *a, int *ia, int *ja, struct DESC *desca, double *b, int *ib, int *jb,
//         struct DESC *descb, double *beta, double *c, int *ic, int *jc, struct DESC *descc );
// }

// void blacs_pinfo( int *p, int *P ) {
//     blacs_pinfo_( p, P );
// }

// int blacs_get( int icontxt, int what ) {
//     int val;
//     blacs_get_( &icontxt, &what, &val );
//     return val;
// }

// int blacs_gridinit( int icontxt, bool isColumnMajor, int nprow, int npcol ) {
//     int newcontext = icontxt;
//     char order = isColumnMajor ? 'C' : 'R';
//     blacs_gridinit_( &newcontext, &order, &nprow, &npcol );
//     return newcontext;
// }

// void blacs_gridinfo( int context, int nprow, int npcol, int *myrow, int *mycol ) {
//     blacs_gridinfo_( &context, &nprow, &npcol, myrow, mycol );
// }

// void blacs_gridexit( int context ) {
//     blacs_gridexit_( &context );
// }

// void blacs_exit( int code ) {
//     blacs_exit_( &code );
// }

// int numroc( int n, int nb, int iproc, int isrcproc, int nprocs ) {
//     return numroc_( &n, &nb, &iproc, &isrcproc, &nprocs );
// }

// void descinit( struct DESC *desc, int m, int n, int mb, int nb, int irsrc, int icsrc, int ictxt, int lld ) {
//     int info;
//     descinit_( desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info );
//     if( info != 0 ) {
//         throw runtime_error( "non zero info: " + toString( info ) );
//     }
// //    return info;
// }

// void pdlaprnt( int m, int n, double *A, int ia, int ja, struct DESC *desc, int irprnt,
//     int icprnt, const char *cmatnm, int nout, double *work ) {
//     int cmatnmlen = strlen(cmatnm);
//     pdlaprnt_( &m, &n, A, &ia, &ja, desc, &irprnt, &icprnt, cmatnm, &nout, work, cmatnmlen );
// }

// void pdgemm( bool isTransA, bool isTransB, int m, int n, int k, double alpha,
//      double *a, int ia, int ja, struct DESC *desca, double *b, int ib, int jb,
//     struct DESC *descb, double beta, double *c, int ic, int jc, struct DESC *descc ) {
//     char transa = isTransA ? 'T' : 'N';
//     char transb = isTransB ? 'T' : 'N';
//     pdgemm_( &transa, &transb, &m, &n, &k, &alpha, a, &ia, &ja, desca, b, &ib, &jb,
//         descb, &beta, c, &ic, &jc, descc );
// }