#include  "NeuralNetwork.h"

void NeuralNetwork::setErrors()
{
	if(target.size() == 0)
	{
		cerr << "No target for this neural network" << endl;
		assert(false);
	}

	int outputLayerSize = layers.at(layers.size() - 1)->getNeurons().size();
	if(target.size() != outputLayerSize)
	{
		cerr << "Target size (" << target.size() << ") is not the same as output layer size: " << outputLayerSize << endl;
		for (int i = 0; i < target.size(); i++)
		{
			cout << target.at(i) << endl;
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
	int outputLayerIndex = layers.size() - 1;
	vector<Neuron*> outputNeurons = layers.at(outputLayerIndex)->getNeurons();

	error = 0.00;
	for(int i = 0; i < target.size();i++)
	{
		double t = target.at(i);
		double y = outputNeurons.at(i)->getActivatedValue();

		errors.at(i) = 0.5 * pow(abs(t - y), 2);
		derivedErrors.at(i) = (y - t);

		error += errors.at(i);
	}
}
