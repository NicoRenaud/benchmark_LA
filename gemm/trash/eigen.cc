
#define EIGEN_DONT_PARRALELIZE

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;
using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char *argv[]){

	//El::Environment env( argc, argv );
	//const int size = El::Input("--size","size of matrices", 100);
	//El::ProcessInput();

	const int size = 1000;
	Eigen::initParallel();

	Eigen::MatrixXd A = Eigen::MatrixXd::Random(size,size);
	Eigen::MatrixXd B = Eigen::MatrixXd::Random(size,size);
	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(size,size);

	
	std::cout << "Matrix size : " << size << "x" << size << std::endl;
	std::cout << "Num Threads : " <<  Eigen::nbThreads() << std::endl;

	auto start = Clock::now();
	C = A*B;
	auto end = Clock::now();
	std::cout << "Run time    : " << std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count() << " secs" <<  std::endl;
	
}