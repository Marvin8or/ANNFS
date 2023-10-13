#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <vector>
#include "../Math/LinearAlgebra/Matrix.h"
#include "LossFunctions.h"


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

	std::vector<Matrix<double>> outputNeuronErrors; // vector that contains [1 x <num-ouput-neurons>] matrices that represent the errors of each output neuron after feedforward
	Matrix<double>(*outputNeuronErrorsFunc)(const Matrix<double>&, const Matrix<double>&);

	std::vector<double>			compoundErrors;		// vector of values that represent the value of error after loss function
	double (*compoundErrorsFunc) (const Matrix<double>&);

	void initialize_matrices();
	

public:
	NeuralNetwork(const std::vector<uint>& topology, const ELossFunction& loss);
	void setInputValues(std::initializer_list<double> inputs, std::initializer_list<double> targets);
	void feedForward();
	void setErrors();
	void backpropagation();
	void summary() const;
	
};

#endif
