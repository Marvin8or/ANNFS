#include  "NeuralNetwork.h"
#include "../Math/LinearAlgebra/Operations.h"

void NeuralNetwork::setErrors()
{
	if(getCurrentTarget().size() == 0)
	{
		cerr << "No target for this neural network" << endl;
		assert(false);
	}

	int outputLayerSize = topology.back();

	if(getCurrentTarget().size() != outputLayerSize)
	{
		cerr << "Target size (" << getCurrentTarget().size() << ") is not the same as output layer size: " << outputLayerSize << endl;
		for (int i = 0; i < getCurrentTarget().size(); i++)
		{
			cout << getCurrentTarget().at(i) << endl;
		}
	}

	switch (costFunction)
	{
		case(COST_MSE):
			setErrorMSE();
			break;
		default:
			setErrorMSE();
			break;	
	}
}


void NeuralNetwork::setErrorMSE()
{
	int outputLayerIndex = topologySize - 1;
	vector<Neuron*> outputNeurons = layers.at(outputLayerIndex)->getNeurons();

	errorOverAllOutputNeurons = 0.00;
	for(int i = 0; i < outputNeurons.size();i++)
	{
		double y = getCurrentTarget().at(i);
		double a = outputNeurons.at(i)->getActivatedValue();

		// TODO Cost function in separate function

		double tmp = a - y;
		double C = 0.5 * pow(tmp, 2);
		double dC = tmp;

		outputErrors->setValue(i, 0, C);
		derivedOutputErrors->setValue(i, 0, dC);
		errorOverAllOutputNeurons += outputErrors->getValue(i, 0);

	}
	Matrix* delta_L = new Matrix(outputNeurons.size(), 1, false);
	LinearAlgebra::Operations::HadamardProduct(derivedOutputErrors, layers.at(outputLayerIndex)->matrixifyDerivedValues(), delta_L);
	deltas.push_back(delta_L);
}
