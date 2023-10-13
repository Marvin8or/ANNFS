#include "NeuralNetwork.h"
#include <cmath>

double fastSigmoidFunction(double x)
{
	return x / (1 + abs(x));
}

double d_fastSigmoidFunction(double x)
{
	return fastSigmoidFunction(x) * (1 - fastSigmoidFunction(x));
}

double rectifiedLinearUnit(double x)
{
	if (x < 0) return 0;
	return x;
}

double d_rectifiedLinearUnit(double x)
{
	if (x < 0) return 0;
	return 1;
}

void NeuralNetwork::initialize_matrices()
{
	for(uint i=0; i<weightsNum; i++)
	{
		weights.push_back(Matrix<double>(topology_.at(i), topology_.at(i + 1)));
	}

	for(uint i=0; i<biasesNum; i++)
	{
		biases.push_back(Matrix<double>(1, topology_.at(i + 1)));
	}

	for(uint i=0; i<layerNum; i++)
	{
		neuronValues.push_back(Matrix<double>(1, topology_.at(i)));
	}
	// Add target layer
	neuronValues.push_back(Matrix<double>(1, topology_.back()));
}

NeuralNetwork::NeuralNetwork(const std::vector<uint>& topology, const ELossFunction& loss)
{
	if (!(topology.size() > 1))
		throw std::invalid_argument("Number of layers in topology is not sufficient!");

	topology_.push_back(topology.at(0));

	for (auto layer : topology)
		topology_.push_back(layer);

	layerNum = topology_.size();
	weightsNum = topology_.size() - 1;
	biasesNum = topology_.size() - 1;

	initialize_matrices();

	switch (loss)
	{
	case MSE:
		outputNeuronErrorsFunc = LossFunctions::squared_error;
		compoundErrorsFunc = LossFunctions::mean_squared_error;
		break;
	}

}

void NeuralNetwork::feedForward()
{

	for(uint i=0; i<weightsNum; i++)
	{
		Matrix<double> tmp = neuronValues[i].dot(weights[i]);
		tmp.applyFunction(fastSigmoidFunction);
		tmp += biases.at(i);
		neuronValues.at(i + 1) = tmp;
	}
}

void NeuralNetwork::setErrors()
{
	neuronValues[topology_.back()] = outputNeuronErrorsFunc(neuronValues[neuronValues.size() - 1], neuronValues[neuronValues.size() - 2]);
	compoundErrors.push_back(compoundErrorsFunc(neuronValues[topology_.back()]));
}

void NeuralNetwork::backpropagation()
{
	uint lastLayerIndx = layerNum - 1;

}

void NeuralNetwork::summary() const
{
	std::cout << "Topology: ";
	for (uint i = 0; i < layerNum; i++)
	{
		std::cout << topology_.at(i) << " ";
	}

	for(uint i=0; i<layerNum; i++)
	{
		if(i == 0)
		{
			std::cout << "\nInput layer:" << std::endl;
			std::cout << "-------------------------------------" << std::endl;
			std::cout << neuronValues.at(i) << std::endl;
		}
		else
		{
			std::cout << "Hidden layer " << i << ":" << std::endl;
			std::cout << "-------------------------------------" << std::endl;
			std::cout << "X" << std::endl;
			std::cout << neuronValues.at(i) << std::endl;
			std::cout << "W" << std::endl;
			std::cout << weights.at(i - 1) << std::endl;
			std::cout << "B" << std::endl;
			std::cout << biases.at(i - 1) << std::endl;
		}
	}
}

void NeuralNetwork::setInputValues(std::initializer_list<double> inputs, std::initializer_list<double> targets)
{
	if (!(inputs.size() == neuronValues.front().getCols() && targets.size() == neuronValues.back().getCols()))
		throw std::invalid_argument("Invalid dimension for inputs or targets!");

	Matrix<double> tmpInput(1, inputs.size()), tmpTarget(1, targets.size());;
	for (uint i = 0; i < inputs.size(); i++)
		tmpInput.put(0, i, *(inputs.begin() + i));
	neuronValues[0] = tmpInput;

	for (uint i = 0; i < targets.size(); i++)
		tmpTarget.put(0, i, *(targets.begin() + i));
	neuronValues[neuronValues.size() - 1] = tmpTarget;
}
