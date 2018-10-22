include /usr/local/conf/ElVars

#elemental: elemental.cc
#	${CXX} ${EL_COMPILE_FLAGS} -I/usr/include/eigen3 -I/opt/OpenBLAS/include/ $< -o $@ ${EL_LINK_FLAGS} ${EL_LIBS} -L/opt/OpenBLAS/lib/ -lopenblas

eigen: eigen.cc
	g++ -O0 -fopenmp -I/usr/include/eigen3 $< -o $@