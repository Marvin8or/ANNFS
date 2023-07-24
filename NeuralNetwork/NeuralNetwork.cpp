#include "NeuralNetwork.h"

void NeuralNetwork::setCurrentInput(vector<double> input)
{
	_input = input;
	for (int i = 0; i < layers.at(0)->getSize(); i++)
	{
		layers.at(0)->setValue(i, input.at(i));
	}
}

NeuralNetwork::NeuralNetwork(
	/*topology*/vector<int> t,
	/*learning rate*/double lr,
	/*momentum*/double m
)
{
	topology		= t;
	topologySize	= t.size();
	learningRate	= lr;
	momentum		= m;

	// Fill vector of Layer objects
	for(int i = 0; i < topologySize; i++)
	{
		layers.push_back(new Layer(topology.at(i)));
	}

	// Fill vector of Weight Matrices
	for(int i = 0; i < topologySize - 1; i++)
	{
		W.push_back(new Matrix(topology.at(i + 1), topology.at(i), true));
	}

	outputErrors = new Matrix(topology.back(), 1, false);
	derivedOutputErrors = new Matrix(topology.back(), 1, false);
	errorOverAllOutputNeurons = 0.00;
}

//NeuralNetwork::NeuralNetwork(
//	vector<int> topology,
//	ActivationFunc hiddenActivationType,
//	ActivationFunc outputActivationType,
//	CostFunctionType costFunction,
//	double bias,
//	double learningRate,
//	double momentum
//)
//{
//	topology		 = topology;
//	topologySize	 = topology.size();
//	bias			 = bias;
//	learningRate	 = learningRate;
//	momentum		 = momentum;
//
//	hiddenActivationType	 = hiddenActivationType;
//	outputActivationType	 = outputActivationType;
//	costFunction			 = costFunction;
//
//	for (int i = 0; i < topologySize; i++)
//	{
//		//Todo implement dict topology {<layer number neurons>: activation function for layer}
//		if(i > 0 && i < (topologySize - 1))
//		{
//			layers.push_back(new Layer(topology.at(i), hiddenActivationType));
//		}
//		else if(i == (topologySize - 1))
//		{
//			layers.push_back(new Layer(topology.at(i), outputActivationType));
//
//		}
//		else
//		{
//			// Input layer has always default act. function e.g. FSF
//			layers.push_back(new Layer(topology.at(i)));
//		}
//		
//	}
//
//	for (int i = 0; i < topologySize - 1; i++)
//	{
//		auto weightMatrix = new Matrix(topology.at(i), topology.at(i + 1), true);
//		weightMatrices.push_back(weightMatrix);
//	}
//
//	for (int i = 0; i < topology.at(topologySize - 1); i++)
//	{
//		errors.push_back(0.00);
//		derivedErrors.push_back(0.00);
//	}
//
//	error = 0.00;
//
//}
