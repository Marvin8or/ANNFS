#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#ifndef MATRIX_H
#include "../Math/LinearAlgebra/Matrix.h"
#endif

enum ELossFunction
{
	MSE
};

struct LossFunctions
{
	static Matrix<double> squared_error(const Matrix<double>& y, const Matrix<double>& y_hat);
	static Matrix<double> squared_error_derived(const Matrix<double>& y, const Matrix<double>& y_hat);
	static double mean_squared_error(const Matrix<double>& errors);
};
#endif