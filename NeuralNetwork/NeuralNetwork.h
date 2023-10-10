#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <vector>
#include "../Math/LinearAlgebra/Matrix.h"


double fastSigmoidFunction(double x); //Static
double d_fastSigmoidFunction(double x);

double rectifiedLinearUnit(double x);
double d_rectifiedLinearUnit(double x);


class NeuralNetwork
{
private:
	std::vector<uint>		    topology_;
	uint layerNum;
	std::vector<Matrix<double>> weights;
	uint weightsNum;
	std::vector<Matrix<double>> biases;
	uint biasesNum;
	std::vector<Matrix<double>> neuronValues;

	std::vector<Matrix<double>> errors;
	std::vector<double> lossFunctionErrors;

	void initialize_matrices();
	

public:
	NeuralNetwork(const std::vector<uint>& topology);
	void feedForward();
	void summary();
	void setInputValues(std::initializer_list<double> inputs, std::initializer_list<double> targets);
};

#endif
