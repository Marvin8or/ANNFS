#include "NeuralNetwork.h"

void NeuralNetwork::setCurrentInput(vector<double> input)
{
	this->input = input;
	for (int i = 0; i < this->layers.at(0)->getSize(); i++)
	{
		this->layers.at(0)->setValue(i, this->input.at(i));
	}
}

NeuralNetwork::NeuralNetwork(
	vector<int> topology,
	double bias,
	double learningRate,
	double momentum
)
{
	this->topology		= topology;
	this->topologySize	= this->topology.size();
	this->bias			= bias;
	this->learningRate	= learningRate;
	this->momentum		= momentum;

	for(int i = 0; i < this->topologySize; i++)
	{
		this->layers.push_back(new Layer(this->topology.at(i)));
	}

	for(int i = 0; i < this->topologySize - 1; i++)
	{
		auto weightMatrix = new Matrix(this->topology.at(i), this->topology.at(i+1), true);
		this->weightMatrices.push_back(weightMatrix);
	}

	for(int i = 0; i < this->topology.at(this->topologySize - 1); i++)
	{
		this->errors.push_back(0.00);
	}

	this->error = 0.00;
}

NeuralNetwork::NeuralNetwork(
	vector<int> topology,
	ActivationFunc hiddenActivationType,
	ActivationFunc outputActivationType,
	CostFunctionType costFunction,
	double bias,
	double learningRate,
	double momentum
)
{
	this->topology		 = topology;
	this->topologySize	 = this->topology.size();
	this->bias			 = bias;
	this->learningRate	 = learningRate;
	this->momentum		 = momentum;

	this->hiddenActivationType	 = hiddenActivationType;
	this->outputActivationType	 = outputActivationType;
	this->costFunction			 = costFunction;

	for (int i = 0; i < this->topologySize; i++)
	{
		//Todo implement dict topology {<layer number neurons>: activation function for layer}
		if(i > 0 && i < (this->topologySize - 1))
		{
			this->layers.push_back(new Layer(this->topology.at(i), this->hiddenActivationType));
		}
		else if(i == (this->topologySize - 1))
		{
			this->layers.push_back(new Layer(this->topology.at(i), this->outputActivationType));

		}
		else
		{
			// Input layer has always default act. function e.g. FSF
			this->layers.push_back(new Layer(this->topology.at(i)));
		}
		
	}

	for (int i = 0; i < this->topologySize - 1; i++)
	{
		auto weightMatrix = new Matrix(this->topology.at(i), this->topology.at(i + 1), true);
		this->weightMatrices.push_back(weightMatrix);
	}

	for (int i = 0; i < this->topology.at(this->topologySize - 1); i++)
	{
		this->errors.push_back(0.00);
	}

	this->error = 0.00;

}
