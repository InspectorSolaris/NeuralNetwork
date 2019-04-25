#ifndef _NEURAL_NET_H_
#define _NEURAL_NET_H_

#include <iostream>
#include <fstream>

#include "math.h"

std::vector<long double> activate(
	std::vector<long double> a
);

std::vector<long double> derivative(
	std::vector<long double> a
);

long double activationFunction(
	const long double & x
);

long double activationFunctionDerivative(
	const long double & x
);

void copyVector(
	const std::vector<std::vector<long double>> & a,
	std::vector<std::vector<long double>> & b
);

void copyVector(
	const std::vector<std::vector<std::vector<long double>>> & a,
	std::vector<std::vector<std::vector<long double>>> & b
);

void initializeWithRandNumbers(
	std::vector<unsigned long long> & rt,
	const unsigned long long & l,
	const unsigned long long & r
);

void initializeNewNeuralNet(
	const char * ifile,
	unsigned long long & n,
	std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & b,
	std::vector<std::vector<std::vector<long double>>> & w
);

void initializeNeuralNet(
	const char * ifile,
	unsigned long long & n,
	std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & b,
	std::vector<std::vector<std::vector<long double>>> & w
);

void initializeDataSet(
	const char * tfile,
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & input,
	std::vector<std::vector<long double>> & output
);

void teachNeuralNet(
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & b,
	std::vector<std::vector<std::vector<long double>>> & w,
	const std::vector<std::vector<long double>> & input,
	const std::vector<std::vector<long double>> & output,
	const unsigned long long & iters,
	const unsigned long long & rt_size,
	const long double & q
);

void learnSingleTestNeuralNet(
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	const std::vector<std::vector<long double>> & b,
	const std::vector<std::vector<std::vector<long double>>> & w,
	const std::vector<long double> & input,
	const std::vector<long double> & output,
	std::vector<std::vector<long double>> & db,
	std::vector<std::vector<std::vector<long double>>> & dw
);

void updateNeuralNet(
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & b,
	std::vector<std::vector<long double>> & db,
	std::vector<std::vector<std::vector<long double>>> & w,
	std::vector<std::vector<std::vector<long double>>> & dw,
	const long double & q,
	const unsigned long long & t
);

std::vector<long double> runNeuralNet(
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	const std::vector<std::vector<long double>> & b,
	const std::vector<std::vector<std::vector<long double>>> & w,
	std::vector<long double> a
);

void outputNeuralNet(
	const char * ofile,
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	const std::vector<std::vector<long double>> & b,
	const std::vector<std::vector<std::vector<long double>>> & w
);

#endif // _NEURAL_NET_H_