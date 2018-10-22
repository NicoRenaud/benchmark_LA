include /usr/local/conf/ElVars

eigen: eigen.cc
	${CXX} ${EL_COMPILE_FLAGS} -I/opt/OpenBLAS/include/ -I/usr/include/eigen3 $< -o $@ ${EL_LINK_FLAGS} ${EL_LIBS} -L/opt/OpenBLAS/lib/ -lopenblas