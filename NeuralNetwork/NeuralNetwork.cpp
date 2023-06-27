#include "NeuralNetwork.h"

void NeuralNetwork::setCurrentInput(vector<double> input)
{
	this->currentInput = input;
	for (int i = 0; i < this->layers.at(0)->getSize(); i++)
	{
		this->layers.at(0)->setVal(i, this->currentInput.at(i));
	}
}

//NeuralNetwork::NeuralNetwork(
//	vector<int> topology,
//	double bias,
//	double learningRate,
//	double momentum
//)
//{
//	this.
//}