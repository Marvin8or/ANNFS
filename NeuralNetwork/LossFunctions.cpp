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

double LossFunctions::mean_squared_error(const Matrix<double>& errors)
{
	double result = 0;
	for (uint i = 0; i < errors.getCols(); i++)
		result += errors.get(0, i);

	result /= errors.getCols();

	return result;
}