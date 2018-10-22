extern "C" {
#define ADD_
    // blas:
    #include <mkl_cblas.h>

    // lapack:
    void dpotrf_( char *uplo, int *n, double *A, int *lda, int *info );
    void dtrtrs_( char *uplo, char *trans, char *diag, int *n, int *nrhs, double *A, int *lda,
         double *B, int *ldb, int *info );
    void dtrsm_( char *side, char *uplo, char *transA, char *diag, const int *m, const int *n, 
                 const double *alpha, const double *A, const int *lda, double *B, const int *ldb );
}

char boolToChar( bool value ) {
    return value ? 't' : 'n';
}

// double, general matrix multiply
void dgemm( bool transa, bool transb, int m, int n, int k, double alpha, double *A,
    int lda, double *B, int ldb, double beta, double *C, int ldc ) {
    char transachar = boolToChar( transa );
    char transbchar = boolToChar( transb );
    dgemm_(&transachar,&transbchar,&m,&n,&k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );    
}
