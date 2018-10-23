

#source /opt/intel/mkl/mklars.sh

MKLROOT = /opt/intel/mkl
INCS = -DMKL_ILP64 -m64 -I${MKLROOT}/include -I/usr/lib/x86_64-linux-gnu/openmpi/include 
INCS_EIGEN = -I/usr/include/eigen3 
INCS_CXXOPTS = -I//home/nicolas/Documents/projects/cxxopts-2.1.1/include
LIBSMPI = /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so /usr/lib/libmpi.so
LIBS =  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_ilp64 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_openmpi_ilp64 -lgomp -lpthread -lm -ldl
CXX=mpic++

scalapack: scalapack.cc
	${CXX} ${INCS} ${INCS_EIGEN} ${INCS_CXXOPTS} $< -o $@  ${LIBS} ${LIBSMPI}
