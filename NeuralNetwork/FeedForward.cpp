#include "NeuralNetwork.h"
#include "../Math/LinearAlgebra/Operations.h"

void NeuralNetwork::feedForward()
{
	for(int i = 1; i < topologySize; i++)
	{
		// get w_L
		Matrix* w_L = getWeightMatrix(i - 1);

		// get a_L-1
		Matrix* a_L_1;
		if (i - 1 != 0)
		{
			a_L_1 = getActivatedNeuronMatrix(i - 1);
		}
		else
		{
			a_L_1 = getNeuronMatrix(i - 1);
		}

		// get b_L For now always 1
		Matrix* b_L = getBiasMatrix(i);

		// compute z_L
		Matrix* z_L = new Matrix(topology.at(i), 1, false);

		LinearAlgebra::Operations::CalculateInputToNextLayer(w_L, a_L_1, b_L, z_L);

		// compute a_L
		for(int r_index = 0; r_index < z_L->getNumRows(); r_index++)
		{
			setNeuronValue(i, r_index, z_L->getValue(r_index, 0));
		}
	}
}
