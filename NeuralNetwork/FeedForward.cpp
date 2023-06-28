#include "NeuralNetwork.h"
#include "../Math/LinearAlgebra/Operations.h"

void NeuralNetwork::feedForward()
{
	Matrix* a; // Matrix of neurons to the left
	Matrix* b; // Matrix of weights between a and
	Matrix* c; // Matrix of neurons to the right

	for(int i = 0; i < (this->topologySize - 1); i++)
	{
		a = this->getNeuronMatrix(i);
		b = this->getWeightMatrix(i);
		c = new Matrix(
			a->getNumRows(),
			b->getNumCols(),
			false
		);

		if(i != 0)
		{
			a = this->getActivatedNeuronMatrix(i);
		}

		Operations::MultiplyMatrices(a, b, c);

		for(int c_index = 0; c_index < c->getNumCols(); c_index++)
		{
			this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index) + this->bias);
		}

		delete a;
		delete b;
		delete c;
	}
}
