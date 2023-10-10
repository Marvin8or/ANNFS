#include "NeuralNetwork.h"
#include "../Math/LinearAlgebra/Matrix.h"
#include "../Math/LinearAlgebra/Operations.h"

void NeuralNetwork::backPropagation()
{

	int lastLayerIndex = topologySize - 1;

	// Push back dC_dw for last delta
	Matrix* dC_dw_L = new Matrix(W.at(lastLayerIndex - 1)->getNumRows(), W.at(lastLayerIndex - 1)->getNumCols(), false);
	Matrix* a_L_1 = layers.at(lastLayerIndex - 1)->matrixifyActivatedValues();
	Matrix* lastDelta = deltas.back();
	LinearAlgebra::Operations::MultiplyMatrices(lastDelta, a_L_1->transpose(), dC_dw_L);
	dC_dw.push_back(dC_dw_L);
	dC_db.push_back(lastDelta);

	for(int layer_index = lastLayerIndex - 1; layer_index >= 1; layer_index--)
	{
		int weight_index = layer_index - 1;
		int layerSize = layers.at(layer_index)->getSize();

		Matrix* delta_L = new Matrix(layerSize, 1, false);
		Matrix* nabla_a_C = new Matrix(layerSize, 1, false);

		Matrix* w_ = W.at(layer_index)->transpose();
		Matrix* d_ = deltas.back();
		LinearAlgebra::Operations::MultiplyMatrices(w_, d_, nabla_a_C);

		Matrix* dev = layers.at(layer_index)->matrixifyDerivedValues();
		LinearAlgebra::Operations::HadamardProduct(nabla_a_C, dev, delta_L);
		deltas.push_back(delta_L);
		dC_db.push_back(delta_L);

		dC_dw_L = new Matrix(W.at(weight_index)->getNumRows(), W.at(weight_index)->getNumCols(), false);
		a_L_1 = layers.at(weight_index)->matrixifyActivatedValues();
		LinearAlgebra::Operations::MultiplyMatrices(delta_L, a_L_1->transpose(), dC_dw_L);
		dC_dw.push_back(dC_dw_L);
	}
}

void NeuralNetwork::gradientDescent()
{
	std::cout << "Learning rate: " << learningRate << std::endl;
	learningRate /= momentum;
	for(int weight_index = 0; weight_index < W.size(); weight_index++)
	{
		int rows = W.at(weight_index)->getNumRows();
		int cols = W.at(weight_index)->getNumCols();

		int dC_dw_index = dC_dw.size() - weight_index - 1;

		Matrix* newWeightMatrix = new Matrix(rows, cols, false);
		Matrix* lr_m = new Matrix(rows, cols, false);
		lr_m->populate(learningRate);
		

		Matrix* lr_m_dC_dw = new Matrix(rows, cols, false);

		LinearAlgebra::Operations::HadamardProduct(lr_m, dC_dw.at(dC_dw_index), lr_m_dC_dw);
		LinearAlgebra::Operations::SubtractMatrices(W.at(weight_index), lr_m_dC_dw, newWeightMatrix);
		W.at(weight_index) = newWeightMatrix;
	}
}
