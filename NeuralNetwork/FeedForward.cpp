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
	}
}
