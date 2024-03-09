#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#define ON 1
#define OFF 0
#define DEBUG OFF
#include <vector>
#include <utility>


#include "../Math/LinearAlgebra/Matrix.h"
#include "../ThirdParty/json.hpp"
#include "../Utils/utils.h"
#include "LossFunctions.h"



using json = nlohmann::json;

double sigmoidFunction(double x);
double d_sigmoidFunction(double x);

Matrix<double> softmaxFunction(const Matrix<double>& input);
Matrix<double> d_softmaxFunction(const Matrix<double>& input);

double fastSigmoidFunction(double x); //Static
double d_fastSigmoidFunction(double x);

double rectifiedLinearUnit(double x);
double d_rectifiedLinearUnit(double x);

enum EActivationFunctions
{
	ReLu,
	Sigmoid,
	FSigmoid,
	Softmax
};

enum EDerivedActivationFunctions
{
	d_ReLu,
	d_Sigmoid,
	d_FSigmoid,
	d_Softmax
};

class NeuralNetwork
{
private:
	std::vector<EActivationFunctions>    activation_functions;
	std::vector<uint>		    topology_;
	uint layerNum;

public:
	std::vector<Matrix<double>> weights;
	uint weightsNum;
	std::vector<Matrix<double>> biases;
	uint biasesNum;

private:
	std::vector<Matrix<double>> neuronValues;
	std::vector<Matrix<double>> neuronValuesDerived;

	Matrix<double>(*outputNeuronErrorsFunc)(const Matrix<double>&, const Matrix<double>&);
	Matrix<double>(*outputNeuronErrorsFuncDerived)(const Matrix<double>&, const Matrix<double>&);

	std::unique_ptr<Matrix<double>>	outputNeuronErrorsPtr;	// Matrix 1x<num-ouput-neurons> that contains the errors of each output neuron, updated after each feedforward
	double							compoundError;			// compound error, updated after each feedforward
	std::vector<double>				compoundErrors;			// Historical compound errors
	std::vector<double>				epochErrors;			// Error after each epoch, mean of all compund errors

	long double (*compoundErrorFunc) (const Matrix<double>&);

	std::vector<Matrix<double>> deltasBackprop;
	std::vector<Matrix<double>> deltaWeights;
	std::vector<Matrix<double>> deltaBiases;

	double learningRate;
	void initialize_matrices();

public:
	NeuralNetwork(const std::vector<uint>& topology, const double& learningRate, const ELossFunction& loss);
	NeuralNetwork(const json& json_configuration);

	//Setters
	void setInputValues(std::initializer_list<double> inputs, std::initializer_list<double> targets);
	void setInputValues(std::vector<double> inputs);
	void setInputValues(const Matrix<double>& inputs);
	void setTargetValues(std::vector<double> targets);
	void setTargetValues(const Matrix<double>& targets);
	void setWeights(uint indx, const Matrix<double>& weight);
	void setWeights(const std::vector<Matrix<double>>& new_weights);
	void setBiases(const std::vector<Matrix<double>>& new_biases);

	//Getters
	std::vector<Matrix<double>> getDeltaWeights();
	std::vector<Matrix<double>> getWeights();
	std::vector<Matrix<double>> getDeltaBiases();
	std::vector<Matrix<double>> getBiases();
	long double getCompoundError() { return compoundError; }

	
	void feedForward();
	void setErrors();
	void backpropagation();
	void gradientDescent();

	void train(const std::vector<std::vector<double>>& inputs, 
			   const std::vector<std::vector<double>>& targets,
			   const uint& nepochs);

	vector2D predict(const vector2D& inputs);

	void summary() const;
	void print_predictions() const;
	
};

#endif
