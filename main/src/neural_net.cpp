#include "neural_net.h"

std::vector<long double> activate(
	std::vector<long double> a
)
{
	for(unsigned long long i = 0; i < a.size(); ++i)
	{
		a[i] = activationFunction(a[i]);
	}
	
	return a;
}

std::vector<long double> derivative(
	std::vector<long double> a
)
{
	for(unsigned long long i = 0; i < a.size(); ++i)
	{
		a[i] = activationFunctionDerivative(a[i]);
	}
	
	return a;
}

long double activationFunction(
	const long double & x
)
{
	return (x >= 0 ? x : 0);
}

long double activationFunctionDerivative(
	const long double & x
)
{
	return (x >= 0 ? 1 : 0);
}

void copyVector(
	const std::vector<std::vector<long double>> & a,
	std::vector<std::vector<long double>> & b
)
{
	b = std::vector<std::vector<long double>>(a.size());
	
	for(unsigned long long i = 0; i < a.size(); ++i)
	{
		b[i] = std::vector<long double>(a[i].size());
	}
	
	return;
}

void copyVector(
	const std::vector<std::vector<std::vector<long double>>> & a,
	std::vector<std::vector<std::vector<long double>>> & b
)
{
	b = std::vector<std::vector<std::vector<long double>>>(a.size());
	
	for(unsigned long long i = 0; i < a.size(); ++i)
	{
		b[i] = std::vector<std::vector<long double>>(a[i].size());
	
		for(unsigned long long j = 0; j < a[i].size(); ++j)
		{
			b[i][j] = std::vector<long double>(a[i][j].size());
		}
	}
		
	return;
}

void initializeWithRandNumbers(
	std::vector<unsigned long long> & rt,
	const unsigned long long & l,
	const unsigned long long & r
)
{
	for(unsigned long long i = 0; i < rt.size(); ++i)
	{
		rt[i] = (rand() % r) + l;
	}
	
	return;
}

void initializeNewNeuralNet(
	const char * ifile,
	unsigned long long & n,
	std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & b,
	std::vector<std::vector<std::vector<long double>>> & w
)
{
	std::ofstream out(ifile);
	
	n = 3;
	l = {48, 96, 16};
	
	out << n << '\n';
	
	for(unsigned long long i = 0; i < n; ++i)
	{
		out << l[i] << '\n';
		
		if(i)
		{
			for(unsigned long long j = 0; j < l[i]; ++j)
			{
				out << (rand() % 2 ? +1 : -1) * (rand() % 10000 + 1) / 1000000.0L << ' ';
				
				for(unsigned long long k = 0; k < l[i - 1]; ++k)
				{
					out << (rand() % 2 ? +1 : -1) * (rand() % 10000 + 1) / 1000000.0L << ' ';
				}
				
				out << '\n';
			}
		}
		
		out << '\n';
	}
}

void initializeNeuralNet(
	const char * ifile,
	unsigned long long & n,
	std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & b,
	std::vector<std::vector<std::vector<long double>>> & w
)
{
	std::ifstream in(ifile);
	
	in >> n;
	
	l = std::vector<unsigned long long>(n);
	b = std::vector<std::vector<long double>>(n);
	w = std::vector<std::vector<std::vector<long double>>>(n);
	
	for(unsigned long long i = 0; i < n; ++i)
	{
		in >> l[i];
		
		if(i)
		{
			b[i] = std::vector<long double>(l[i]);
			w[i] = std::vector<std::vector<long double>>(l[i]);
			
			for(unsigned long long j = 0; j < l[i]; ++j)
			{
				in >> b[i][j];
				
				w[i][j] = std::vector<long double>(l[i - 1]);
				
				for(unsigned long long k = 0; k < l[i - 1]; ++k)
				{
					in >> w[i][j][k];
				}
			}
		}
	}
	
	return;
}

void initializeDataSet(
	const char * tfile,
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & input,
	std::vector<std::vector<long double>> & output
)
{
	std::ifstream in(tfile);
	
	unsigned long long tests = 0;
	
	in >> tests;
	
	input = std::vector<std::vector<long double>>(tests, std::vector<long double>(l[0 - 0]));
	output = std::vector<std::vector<long double>>(tests, std::vector<long double>(l[n - 1]));
	
	for(unsigned long long t = 0; t < tests; ++t)
	{
		for(unsigned long long i = 0; 3 * i < input[t].size(); ++i)
		{
			char ch = 0;
			
			in >> ch;
			
			input[t][3 * i + 0] = (ch == '.' ? 1 : 0);
			input[t][3 * i + 1] = (ch == 'X' ? 1 : 0);
			input[t][3 * i + 2] = (ch == 'O' ? 1 : 0);
		}
		
		unsigned long long x = 0;
		unsigned long long y = 0;
		
		in >> x >> y;
		
		output[t][4 * y + x] = 1;
	}
	
	return;
}

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
)
{
	unsigned long long tests = input.size();
	
	std::vector<std::vector<long double>> db; // delta b
	std::vector<std::vector<std::vector<long double>>> dw; // delta w
	
	copyVector(b, db);
	copyVector(w, dw);
	
	for(unsigned long long i = 0; i < iters; ++i)
	{
		std::vector<unsigned long long> rt(rt_size);
		
		initializeWithRandNumbers(rt, 0, input.size());
		
		for(unsigned long long t = 0; t < rt_size; ++t)
		{
			learnSingleTestNeuralNet(n, l, b, w, input[rt[t]], output[rt[t]], db, dw);
		}
		
		updateNeuralNet(n, l, b, db, w, dw, q, rt_size);
		
		std::cout << i + 1 << " | " << iters << '\n';
	}
	
	return;
}

void learnSingleTestNeuralNet(
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	const std::vector<std::vector<long double>> & b,
	const std::vector<std::vector<std::vector<long double>>> & w,
	const std::vector<long double> & input,
	const std::vector<long double> & output,
	std::vector<std::vector<long double>> & db,
	std::vector<std::vector<std::vector<long double>>> & dw
)
{
	std::vector<std::vector<long double>> d(n); // delta
	std::vector<std::vector<long double>> a(n); // layer output
	std::vector<std::vector<long double>> z(n); // layer input
	
	a[0] = input;
	
	for(unsigned long long i = 1; i < n; ++i)
	{
		z[i] = sum(mul(a[i - 1], w[i], false), b[i]);
		a[i] = activate(z[i]);
	}
	
	d[n - 1] = mul(dif(a[n - 1], output), derivative(z[n - 1]));
	
	for(unsigned long long i = n - 2; i >= 1; --i)
	{
		d[i] = mul(mul(d[i + 1], w[i + 1], true), derivative(z[i]));
	}
	
	for(unsigned long long i = 1; i < n; ++i)
	{
		for(unsigned long long j = 0; j < l[i]; ++j)
		{
			db[i][j] += d[i][j];
			
			for(unsigned long long k = 0; k < l[i - 1]; ++k)
			{
				dw[i][j][k] += a[i - 1][k] * d[i][j];
			}
		}
	}
	
	return;
}

void updateNeuralNet(
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	std::vector<std::vector<long double>> & b,
	std::vector<std::vector<long double>> & db,
	std::vector<std::vector<std::vector<long double>>> & w,
	std::vector<std::vector<std::vector<long double>>> & dw,
	const long double & q,
	const unsigned long long & t
)
{
	for(unsigned long long i = 1; i < n; ++i)
	{
		for(unsigned long long j = 0; j < l[i]; ++j)
		{
			b[i][j] -= q / t * db[i][j];
			db[i][j] = 0;
			
			for(unsigned long long k = 0; k < l[i - 1]; ++k)
			{
				w[i][j][k] -= q / t * dw[i][j][k];
				dw[i][j][k] = 0;
			}
		}
	}
	
	return;
}

std::vector<long double> runNeuralNet(
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	const std::vector<std::vector<long double>> & b,
	const std::vector<std::vector<std::vector<long double>>> & w,
	std::vector<long double> a
)
{
	for(unsigned long long i = 1; i < n; ++i)
	{
		a = activate(sum(mul(a, w[i], false), b[i]));
	}
	
	return a;
}

void outputNeuralNet(
	const char * ofile,
	const unsigned long long & n,
	const std::vector<unsigned long long> & l,
	const std::vector<std::vector<long double>> & b,
	const std::vector<std::vector<std::vector<long double>>> & w
)
{
	std::ofstream out(ofile);
	
	out << n << '\n';
	
	for(unsigned long long i = 0; i < n; ++i)
	{
		out << l[i] << '\n';
		
		if(i)
		{
			for(unsigned long long j = 0; j < l[i]; ++j)
			{
				out << b[i][j] << ' ';
				
				for(unsigned long long k = 0; k < l[i - 1]; ++k)
				{
					out << w[i][j][k] << ' ';
				}
				
				out << '\n';
			}
		}
		
		out << '\n';
	}
	
	return;
}