include /usr/local/conf/ElVars

elemental: elemental.cc
	${CXX} ${EL_COMPILE_FLAGS} -fopenmp -I/opt/OpenBLAS/include/ $< -o $@ ${EL_LINK_FLAGS} ${EL_LIBS} -L/opt/OpenBLAS/lib/ -lopenblas

