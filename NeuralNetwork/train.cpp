#include "NeuralNetwork.h"
void NeuralNetwork::train(
	vector<double> input,
	vector<double> target,
	double bias,
	double learningRate,
	double momentum)
{
	learningRate	= learningRate;
	momentum		= momentum;
	bias			= bias;

	setCurrentInput(input);
	setCurrentTarget(target);

	feedForward();
	setErrors(); // Calculate output error
	backPropagation();
	gradientDescent();
	
}
