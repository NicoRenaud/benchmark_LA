
#source /opt/intel/mkl/mklars.sh

MKLROOT = /opt/intel/mkl
INCS = -DMKL_ILP64 -m64 -I${MKLROOT}/include -I/usr/include/eigen3 -fopenmp
LIBS =  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl 

eigen_blas: eigen_blas.cc
	${CXX} ${INCS} $< -o $@  ${LIBS}