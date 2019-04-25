#include "math.h"

std::vector<long double> sum(
	std::vector<long double> a,
	const std::vector<long double> & b
)
{
	if(a.size() != b.size())
	{
		return std::vector<long double>(0);
	}
	
	for(unsigned long long i = 0; i < a.size(); ++i)
	{
		a[i] += b[i];
	}
	
	return a;
}

std::vector<long double> dif(
	std::vector<long double> a,
	const std::vector<long double> & b
)
{
	if(a.size() != b.size())
	{
		return std::vector<long double>(0);
	}
	
	for(unsigned long long i = 0; i < a.size(); ++i)
	{
		a[i] -= b[i];
	}
	
	return a;
}

std::vector<long double> mul(
	const std::vector<long double> & a,
	const std::vector<std::vector<long double>> & b,
	const bool & b_is_transpoted
)
{
	if(!b.size() || !b[0].size())
	{
		return std::vector<long double>(0);
	}
	
	unsigned long long b_n = !b_is_transpoted * b.size() + b_is_transpoted * b[0].size();
	unsigned long long b_m = b_is_transpoted * b.size() + !b_is_transpoted * b[0].size();
	
	if(a.size() != b_m)
	{
		return std::vector<long double>(0);
	}
	
	std::vector<long double> res(b_n, 0);
	
	for(unsigned long long i = 0; i < b_n; ++i)
	{
		for(unsigned long long j = 0; j < b_m; ++j)
		{
			unsigned long long true_i = !b_is_transpoted * i + b_is_transpoted * j;
			unsigned long long true_j = b_is_transpoted * i + !b_is_transpoted * j;
			
			res[i] += a[j] * b[true_i][true_j];
		}
	}
	
	return res;
}

std::vector<long double> mul(
	std::vector<long double> a,
	const std::vector<long double> & b
)
{
	if(a.size() != b.size())
	{
		return std::vector<long double>(0);
	}
	
	for(unsigned long long i = 0; i < a.size(); ++i)
	{
		a[i] *= b[i];
	}
	
	return a;
}