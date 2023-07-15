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
			b->getNumCols(),
			1,
			false
		);

		if(i != 0)
		{
			a = this->getActivatedNeuronMatrix(i);
		}

		Operations::MultiplyMatrices(b->transpose(), a, c);

		for(int r_index = 0; r_index < c->getNumRows(); r_index++)
		{
			this->setNeuronValue(i + 1, r_index, c->getValue(r_index, 0) + this->bias);
		}

		delete a;
		delete b;
		delete c;
	}
}
