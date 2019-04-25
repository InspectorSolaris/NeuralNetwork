#include "main.h"

int main()
{
	srand(time(0));
	
	const char * ifile = "../io/i.io";
	const char * ofile = "../io/o.io";
	const char * tfile = "../io/t.io";
	
	const long double iter_amount   = 1e+2;
	const long double rt_size       = 4e+4;
	const long double learning_rate = 1e-0;
	
	unsigned long long n = 0;
	std::vector<unsigned long long> l;
	std::vector<std::vector<long double>> b;
	std::vector<std::vector<std::vector<long double>>> w;
	std::vector<std::vector<long double>> input;
	std::vector<std::vector<long double>> output;
	
	initializeNewNeuralNet(ifile, n, l, b, w);
	
	initializeNeuralNet(ifile, n, l, b, w);
	
	initializeDataSet(tfile, n, l, input, output);
	
	teachNeuralNet(n, l, b, w, input, output, iter_amount, rt_size, learning_rate);
	
	outputNeuralNet(ifile, n, l, b, w);
	
	return 0;
}