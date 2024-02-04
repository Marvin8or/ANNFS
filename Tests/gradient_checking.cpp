#include "gradient_checking.hpp"
#include "../NeuralNetwork/LossFunctions.h"
#include <tuple>

Matrix<double> gc_rectifiedLinearUnit(const Matrix<double>& z)
{
	Matrix<double> result = Matrix<double>(1, z.getCols());
	for (uint i = 0; i < z.getCols(); i++)
	{
		if (z.get(0, i) < 0)
		{
			result.put(0, i, 0);
		}
		else
		{
			result.put(0, i, z.get(0, i));
		}
	}
	return result;
}

double gc_mean_squared_error(const Matrix<double>& y_hat, const Matrix<double>& y)
{
	Matrix<double> diffs = y_hat - y;
	double result = 0;
	for (uint i = 0; i < diffs.getCols(); i++)
	{
		result += std::pow(diffs.get(0, i), 2);
	}
	result /= diffs.getCols();
	return result;
}

Matrix<double> gc_feedforward(const std::vector<Matrix<double>>& W, const Matrix<double>& X, const std::vector<Matrix<double>>& B)
{
	std::vector<Matrix<double>> neurons;
	neurons.push_back(X);
	for (uint i = 0; i < W.size(); i++)
	{
		Matrix<double> z(1, W[i].getCols());
		if (i == 0)
		{
			z = X.dot(W[i]) + B[i];
		}
		else
		{
			z = neurons[i].dot(W[i]) + B[i];
		}
		Matrix<double> result = gc_rectifiedLinearUnit(z);
		neurons.push_back(result);
	}

	return neurons.back();
}

void check_gradients(json json_configuration)
{
	//auto X = Matrix<double>{ {1.2, 0.8, 0.5, 1.0} };

	//auto y = Matrix<double>{ {0, 1} };	// Class 1

	const std::string trainImagePath = json_configuration["data"]["train_data_path"]["images"];
	const std::string trainLabelPath = json_configuration["data"]["train_data_path"]["labels"];

	// Read MNIST training images
	std::vector<MNISTImage> trainingImages = readMNISTImages(trainImagePath, trainLabelPath);

	vector2D trainingPixels;
	vector2D trainingLabels;
	// trainingImages[0].pixels vector<double>
	auto X = Matrix<double>(1, 784);
	for (int pi = 0; pi < trainingImages[0].pixels.size(); pi++)
		X.put(0, pi, trainingImages[0].pixels[pi]);

	auto y = Matrix<double>(1, 10, 0);
	for (int i = 0; i < trainingImages[0].label.size(); i++)
	{
		if (trainingImages[0].label[i] == 1)
		{
			y.put(0, i, 1);
		}
	}
	
	std::vector<uint> topology1 = { 784, 300, 100, 10 };
	//std::vector<uint> topology = { 4, 3, 2 };
	std::vector<Matrix<double>> W; //([5x4], [4x3], [3x2])
	std::vector<Matrix<double>> gradaprox;
	std::vector<Matrix<double>> B; //([1x4], [1x3], [1x2])
	for (uint i=0; i<topology1.size() - 1; i++)
	{
		Matrix<double>w(topology1[i], topology1[i + 1]);
		Matrix<double>ga(topology1[i], topology1[i + 1]);
		Matrix<double>b(1, topology1[i + 1]);
		W.push_back(w);
		gradaprox.push_back(ga);
		B.push_back(b);
	}

	double epsilon = 1e-6;
	for (uint wi = 0; wi < W.size(); wi++)
	{
		for (uint ri = 0; ri < W[wi].getRows(); ri++)
		{
			for (uint ci = 0; ci < W[wi].getCols(); ci++)
			{
				double orig_value = W[wi].get(ri, ci);
				double new_value_plus = orig_value + epsilon;
				double new_value_minus = orig_value - epsilon;

				W[wi].put(ri, ci, new_value_plus);
				Matrix<double> ff_result_plus = gc_feedforward(W, X, B);
				double mse_plus = gc_mean_squared_error(ff_result_plus, y);

				W[wi].put(ri, ci, new_value_minus);
				Matrix<double> ff_result_minus = gc_feedforward(W, X, B);
				double mse_minus = gc_mean_squared_error(ff_result_minus, y);

				W[wi].put(ri, ci, orig_value);

				double gradapprox_wi = (mse_plus - mse_minus) / (2 * epsilon);
				//std::cout << "gradapprox_wi: " << gradapprox_wi << std::endl;
				gradaprox[wi].put(ri, ci, gradapprox_wi);

			}
		}
	}

	NeuralNetwork nn = NeuralNetwork(json_configuration);

	nn.setInputValues(X);
	nn.setTargetValues(y);
	nn.feedForward();
	nn.setErrors();
	nn.backpropagation();
	std::vector<Matrix<double>> backpropgrad = nn.getDeltaWeights();
	//for (uint dwi = 0; dwi < backpropgrad.size(); dwi++)
	//{
	//	std::cout << "BackPropGrad: " << std::endl;
	//	std::cout << backpropgrad[dwi] << std::endl;

	//	std::cout << "GradApprox: " << std::endl;
	//	std::cout << gradaprox[dwi] << std::endl;
	//}

	double numerator = 0;
	double denominator = 0;
	for (int i = 0; i < backpropgrad.size(); i++)
	{
		double tmp = 0;
		double grad = 0;
		double approx = 0;
		for (int r_idx = 0; r_idx < backpropgrad[i].getRows(); r_idx++)
		{
			for (int c_idx = 0; c_idx < backpropgrad[i].getCols(); c_idx++)
			{
				tmp = backpropgrad[i].get(r_idx, c_idx) - gradaprox[i].get(r_idx, c_idx);
				numerator += std::pow(std::pow(tmp, 2), 0.5);
				grad += std::pow(std::pow(backpropgrad[i].get(r_idx, c_idx), 2), 0.5);
				approx += std::pow(std::pow(gradaprox[i].get(r_idx, c_idx), 2), 0.5);
				denominator += grad + approx;
			}
		}
	}

	//std::cout << "denominator: " << denominator << std::endl;

	double difference = numerator / denominator;
	//std::cout << "difference: " << difference << std::endl;

	if (difference < epsilon)
	{
		std::cout << "Backpropagation algorithm is correct!" << std::endl;
	}
	else
	{
		std::cout << "Backpropagation algotithm is incorrect!" << std::endl;
	}
}
//void check_gradients(json json_configuration)
//{
//	const std::string trainImagePath = json_configuration["data"]["train_data_path"]["images"];
//	const std::string trainLabelPath = json_configuration["data"]["train_data_path"]["labels"];
//
//	// Read MNIST training images
//	std::vector<MNISTImage> trainingImages = readMNISTImages(trainImagePath, trainLabelPath);
//
//	vector2D trainingPixels;
//	vector2D trainingLabels;
//
//	trainingPixels.push_back(trainingImages[0].pixels);
//	trainingLabels.push_back(trainingImages[0].label);
//
//	auto nn_backprop = NeuralNetwork(json_configuration);
//	nn_backprop.setInputValues(trainingPixels[0]);
//	nn_backprop.setTargetValues(trainingLabels[0]);
//	nn_backprop.feedForward();
//	nn_backprop.setErrors();
//
//	std::vector<Matrix<double>> weights_backprop = nn_backprop.getWeights();
//	std::vector<Matrix<double>> biases_backprop = nn_backprop.getBiases();
//
//	std::vector<Matrix<double>> grad_weights_backprop = nn_backprop.getDeltaWeights();
//	std::vector<Matrix<double>> grad_weights_approx(grad_weights_backprop);
//
//	std::vector<Matrix<double>> grad_biases_backprop = nn_backprop.getDeltaBiases();
//	std::vector<Matrix<double>> grad_biases_approx(grad_biases_backprop);
//
//	nn_backprop.backpropagation();
//
//	auto nn_grad = NeuralNetwork(json_configuration);
//	nn_grad.setInputValues(trainingPixels[0]);
//	nn_grad.setTargetValues(trainingLabels[0]);
//
//	nn_grad.setWeights(weights_backprop);
//	nn_grad.setBiases(biases_backprop);
//
//	//for (int i = 0; i < weights_backprop.size(); i++)
//	//{
//	//	if (!(nn_grad.getWeights().at(i) == nn_backprop.getWeights().at(i)))
//	//		std::cerr << "Matrices not the same" << std::endl;
//	//}
//
//	//for (int i = 0; i < biases_backprop.size(); i++)
//	//{
//	//	if (!(nn_grad.getBiases().at(i) == nn_backprop.getBiases().at(i)))
//	//		std::cerr << "Matrices not the same" << std::endl;
//	//}
//
//	double epsilon = 10000;
//	for (int i=0; i < weights_backprop.size(); i++)
//	{
//		std::vector<Matrix<double>> weights_copy(weights_backprop);
//		for (int w_row_indx = 0; w_row_indx < weights_backprop[i].getRows(); w_row_indx++)
//		{
//			for (int w_col_indx = 0; w_col_indx < weights_backprop[i].getCols(); w_col_indx++)
//			{
//				std::vector<Matrix<double>> tmp(weights_copy);
//				double w_value_plus = tmp[i].get(w_row_indx, w_col_indx) + epsilon;
//				tmp[i].put(w_row_indx, w_col_indx, w_value_plus);
//
//				nn_grad.setWeights(tmp);
//				nn_grad.feedForward();
//				nn_grad.setErrors();
//
//				long double loss_plus = nn_grad.getCompoundError();
//
//				//Resetting weights
//				tmp = weights_copy;
//				nn_grad.setWeights(weights_copy);
//				double w_value_minus = tmp[i].get(w_row_indx, w_col_indx) - epsilon;
//				tmp[i].put(w_row_indx, w_col_indx, w_value_minus);
//				
//				nn_grad.setWeights(tmp);
//				nn_grad.feedForward();
//				nn_grad.setErrors();
//
//				long double loss_minus = nn_grad.getCompoundError();
//
//				//Resetting weights
//				tmp = weights_copy;
//
//				nn_grad.setWeights(weights_copy);
//
//				double gradapprox_wi = (loss_plus - loss_minus) / (2 * epsilon);
//				grad_weights_approx[i].put(w_row_indx, w_col_indx, gradapprox_wi);
//
//			}
//		}
//	}
//
//	double numerator = 0;
//	double denominator = 0;
//	for (int i = 0; i < grad_weights_backprop.size(); i++)
//	{
//		double tmp = 0;
//		double grad = 0;
//		double gradapprox = 0;
//		for (int r_idx = 0; r_idx < grad_weights_approx[i].getRows(); r_idx++)
//		{
//			for (int c_idx = 0; c_idx < grad_weights_approx[i].getCols(); c_idx++)
//			{
//				tmp = grad_weights_backprop[i].get(r_idx, c_idx) - grad_weights_approx[i].get(r_idx, c_idx);
//				numerator += std::pow(std::pow(tmp, 2), 0.5);
//				grad += std::pow(std::pow(grad_weights_backprop[i].get(r_idx, c_idx), 2), 0.5);
//				gradapprox += std::pow(std::pow(grad_weights_approx[i].get(r_idx, c_idx), 2), 0.5);
//				denominator += grad + gradapprox;
//			}
//		}
//	}
//
//	double difference = numerator / denominator;
//
//	if (difference < epsilon)
//	{
//		std::cout << "Backpropagation algorithm is correct!" << std::endl;
//	}
//	else
//	{
//		std::cout << "Backpropagation algotithm is incorrect!" << std::endl;
//	}
//
//}
