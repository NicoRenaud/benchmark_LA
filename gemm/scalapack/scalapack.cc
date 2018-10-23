#include <iostream>
#include <cmath>
#include <mpi.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <cxxopts.hpp>


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

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> Mat;

// conventions:
// M_ by N_ matrix block-partitioned into MB_ by NB_ blocks, then
// distributed according to 2d block-cyclic scheme

// based on http://acts.nersc.gov/scalapack/hands-on/exercise3/pspblasdriver.f.html

int main( int argc, char *argv[] ) {


    // Define constants
    MKL_INT i_negone = -1, i_zero = 0, i_one = 1;
    const double zero = 0.0e+0, one = 1.0e+0, two = 2.0e+0, negone = -1.0E+0;
    bool print = false;
    const char trans = 'N';
    MKL_INT ictxt, info;

    // Get blacs info
    MKL_INT myrank, commSize;
    blacs_pinfo_( &myrank, &commSize );

    // MPI size
    MKL_INT nprows = getRootFactor( commSize );
    MKL_INT npcols = commSize / nprows;

    // Parse input
    MKL_INT size;
    try 
    {
        cxxopts::Options options(argv[0], " - PDGEMM example with Scalapack");
        options
         .positional_help("[optional args]")
         .show_positional_help();

        options.add_options()
            ("size", "dimension of the matrix", cxxopts::value<int>(), "M");

        auto result = options.parse(argc,argv);
        size = static_cast<MKL_INT>(result["size"].as<int>());

    } catch (const cxxopts::OptionException& e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    // print
    if( myrank == 0 ) 
    {
        std::cout << "Matrix size : " << size << "x" << size << std::endl;
        std::cout << "Num Procs   : " << commSize << std::endl;
        std::cout << "Grid        : " << nprows << " x " << npcols << std::endl;
    }

    // init BLACS
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "C", &commSize, &i_one );

    // create the grid
    MKL_INT myrow, mycol;
    blacs_gridinfo_( &ictxt, &nprows, &npcols, &myrow, &mycol );

    // nb is column block size for A
    MKL_INT NB_MAX = 128;
    MKL_INT nb = std::min(size/commSize,NB_MAX);

    // local matrix only held by root
    double *Alocal = NULL, *Blocal = NULL, *Clocal = NULL;
    Mat dataA, dataB;
    if( (myrow == 0) && (mycol == 0) ) {

        dataA = Mat::Random(size,size);
        Alocal = dataA.data();
        

        dataB = Mat::Random(size,size);
        Blocal = dataB.data();

        if ( print )
        {
            std::cout << "Eigen A : \n" << dataA << std::endl << std::endl;
            std::cout << "Eigen B : \n" << dataB << std::endl << std::endl;
            std::cout << "A*B \n" << dataA*dataB << std::endl << std::endl;
        }

        Clocal = static_cast<double*>(std::calloc(size*size,sizeof(double)));
    }

    // distributed matrix
    double *Adist, *Bdist, *Cdist;
    int mp = numroc_( &size, &nb, &myrow, &i_zero, &nprows ); // mp number rows owned by this process
    int nq = numroc_( &size, &nb, &mycol, &i_zero, &npcols ); // nq number cols owned by this process
    Adist = static_cast<double*>(std::calloc(mp*nq,sizeof(double)));
    Bdist = static_cast<double*>(std::calloc(mp*nq,sizeof(double)));
    Cdist = static_cast<double*>(std::calloc(mp*nq,sizeof(double)));

    // leading dimension
    MKL_INT lld_local = std::max((int) numroc_(&size,&size,&myrow,&i_zero,&nprows),1);
    MKL_INT lld = std::max(mp,1);  

    // descritor for local matrices
    MKL_INT desca_local[9], descb_local[9], descc_local[9];
    descinit( desca_local, &size, &size, &size, &size, &i_zero, &i_zero, &ictxt, &lld_local, &info );
    descinit( descb_local, &size, &size, &size, &size, &i_zero, &i_zero, &ictxt, &lld_local, &info );
    descinit( descc_local, &size, &size, &size, &size, &i_zero, &i_zero, &ictxt, &lld_local, &info );

    // descriptor of the distributed matrix
    MKL_INT desca_dist[9], descb_dist[9], descc_dist[9];
    descinit_(desca_dist, &size, &size, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info);
    descinit_(descb_dist, &size, &size, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info);
    descinit_(descc_dist, &size, &size, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info);


    // Distribute the matrices from process 0 over the grid
    pdgeadd_(&trans, &size, &size, &one, Alocal, &i_one, &i_one, desca_local, &zero, Adist, &i_one, &i_one, desca_dist);
    pdgeadd_(&trans, &size, &size, &one, Blocal, &i_one, &i_one, descb_local, &zero, Bdist, &i_one, &i_one, descb_dist);


    // start chrono
    std::chrono::time_point<std::chrono::system_clock> start, end;
    if (myrank == 0)
        start = std::chrono::system_clock::now();

    // compute C = A * B
    pdgemm_( &trans, &trans, &size, &size, &size, &one, Adist, &i_one, &i_one, desca_dist, Bdist, &i_one, &i_one, descb_dist,
             &zero, Cdist, &i_one, &i_one, descc_dist );

    // end chrono
    if (myrank == 0)
    {   
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_time = end-start;
        std::cout << "Run time    : " << elapsed_time.count() << " secs" <<  std::endl;
    }
    
    // copy result in local matrix
    pdgeadd_("N",&size,&size,&one,Cdist,&i_one,&i_one,descc_dist,&zero,Clocal,&i_one,&i_one,descc_local);

    if ((myrow == 0) && (mycol == 0))
    {
        if( print )
        {
            // Print Alocal
            std::cout << "Alocal" << std::endl;
            for (int i=0; i<size; i++){
                for(int j=0; j<size; j++){
                    std::cout << Alocal[i*size+j] << " ";
                }
                std::cout << std::endl;
            }

            // Print Blocal
            std::cout << "Blocal" << std::endl;
            for (int i=0; i<size; i++){
                for(int j=0; j<size; j++){
                    std::cout << Blocal[i*size+j] << " ";
                }
                std::cout << std::endl;
            }  

            // copy the Clocal in an Eigen matrix
            double *Ceigen;
            new (Ceigen) Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> (Clocal,size,size);
            std::cout << "Clocal \n" << Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> (Clocal,size,size) << std::endl;

        }
    }

    // termiante BLACS
    blacs_gridexit_( &ictxt );
    blacs_exit_(&i_zero);
    return(0);

}

