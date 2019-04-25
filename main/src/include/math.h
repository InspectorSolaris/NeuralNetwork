#ifndef _MATH_H_
#define _MATH_H_

#include <vector>

std::vector<long double> sum(
	std::vector<long double> a,
	const std::vector<long double> & b
);

std::vector<long double> dif(
	std::vector<long double> a,
	const std::vector<long double> & b
);

std::vector<long double> mul(
	const std::vector<long double> & a,
	const std::vector<std::vector<long double>> & b,
	const bool & b_is_transpoted
);

std::vector<long double> mul(
	std::vector<long double> a,
	const std::vector<long double> & b
);

#endif // _MATH_H_