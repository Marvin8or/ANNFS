#include "LossFunctions.h"

#ifndef MATRIX_H
#include "../Math/LinearAlgebra/Matrix.h"
#endif

Matrix<double> LossFunctions::squared_error(const Matrix<double>& y, const Matrix<double>& y_hat)
{

	Matrix<double> result = y - y_hat;
	result *= result;

	return result;
}

Matrix<double> LossFunctions::squared_error_derived(const Matrix<double>& y, const Matrix<double>& y_hat)
{
	Matrix<double> result = 2.0 * (y - y_hat);
	return result;
}

long double LossFunctions::mean_squared_error(const Matrix<double>& errors)
{
	long double result = 0;
	for (uint i = 0; i < errors.getCols(); i++)
		result += errors.get(0, i);

	result /= errors.getCols();

	return result;
}