#pragma once

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

    void blacs_pinfo_( int *iam, int *nprocs );
    void blacs_get_( int *icontxt, int *what, int *val );
    void blacs_gridinit_( int *icontxt, char *order, int *nprow, int *npcol );
    void blacs_gridinfo_( int *context, int *nprow, int *npcol, int *myrow, int *mycol );
    void blacs_gridexit_( int *context );
    void blacs_exit_( int *code );

    int numroc_( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs );
    void descinit_( struct DESC *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info );
    void pdlaprnt_( int *m, int *n, double *a, int *ia, int *ja, struct DESC *desca, int *irprnt,
        int *icprnt, const char *cmatnm, int *nout, double *work, int cmtnmlen );
    void pdgemm_( char *transa, char *transb, int *m, int *n, int *k, double *alpha,
         double *a, int *ia, int *ja, struct DESC *desca, double *b, int *ib, int *jb,
        struct DESC *descb, double *beta, double *c, int *ic, int *jc, struct DESC *descc );
}

void blacs_pinfo( int *p, int *P ) {
    blacs_pinfo_( p, P );
}

int blacs_get( int icontxt, int what ) {
    int val;
    blacs_get_( &icontxt, &what, &val );
    return val;
}

int blacs_gridinit( int icontxt, bool isColumnMajor, int nprow, int npcol ) {
    int newcontext = icontxt;
    char order = isColumnMajor ? 'C' : 'R';
    blacs_gridinit_( &newcontext, &order, &nprow, &npcol );
    return newcontext;
}

void blacs_gridinfo( int context, int nprow, int npcol, int *myrow, int *mycol ) {
    blacs_gridinfo_( &context, &nprow, &npcol, myrow, mycol );
}

void blacs_gridexit( int context ) {
    blacs_gridexit_( &context );
}

void blacs_exit( int code ) {
    blacs_exit_( &code );
}

int numroc( int n, int nb, int iproc, int isrcproc, int nprocs ) {
    return numroc_( &n, &nb, &iproc, &isrcproc, &nprocs );
}

void descinit( struct DESC *desc, int m, int n, int mb, int nb, int irsrc, int icsrc, int ictxt, int lld ) {
    int info;
    descinit_( desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info );
    if( info != 0 ) {
        throw runtime_error( "non zero info: " + toString( info ) );
    }
//    return info;
}

void pdlaprnt( int m, int n, double *A, int ia, int ja, struct DESC *desc, int irprnt,
    int icprnt, const char *cmatnm, int nout, double *work ) {
    int cmatnmlen = strlen(cmatnm);
    pdlaprnt_( &m, &n, A, &ia, &ja, desc, &irprnt, &icprnt, cmatnm, &nout, work, cmatnmlen );
}

void pdgemm( bool isTransA, bool isTransB, int m, int n, int k, double alpha,
     double *a, int ia, int ja, struct DESC *desca, double *b, int ib, int jb,
    struct DESC *descb, double beta, double *c, int ic, int jc, struct DESC *descc ) {
    char transa = isTransA ? 'T' : 'N';
    char transb = isTransB ? 'T' : 'N';
    pdgemm_( &transa, &transb, &m, &n, &k, &alpha, a, &ia, &ja, desca, b, &ib, &jb,
        descb, &beta, c, &ic, &jc, descc );
}