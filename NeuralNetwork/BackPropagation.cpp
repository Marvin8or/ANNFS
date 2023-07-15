#include "NeuralNetwork.h"
#include "../Math/LinearAlgebra/Matrix.h"
#include "../Math/LinearAlgebra/Operations.h"

void NeuralNetwork::backPropagation()
{
	// Output layer
	int outputLayerIndex = topology.size() - 1;
	int lastWeightMatrixIndex = weightMatrices.size() - 1;

	vector<Matrix*> vector_dC0_cw;
	vector<Matrix*> newWeights;
	vector<Matrix*> vector_dC0_ca;

	// Calculate the vector of derived cost functions
	Matrix* derivedCostFunction = new Matrix(this->layers.at(outputLayerIndex)->getSize(), 1, false);
	for(int r = 0; r < layers.at(outputLayerIndex)->getSize(); r++)
	{
		derivedCostFunction->setValue(r, 0, 2 * getErrors().at(r));
	}
	vector_dC0_ca.push_back(derivedCostFunction);

	Matrix* derivedLayer = layers.at(outputLayerIndex)->matrixifyDerivedValues();

	Matrix* hadamardProduct = new Matrix(derivedCostFunction->getNumRows(), derivedCostFunction->getNumCols(), false);
	Operations::HadamardProduct(derivedLayer, derivedCostFunction, hadamardProduct);

	Matrix* activatedPreviousLayer = layers.at(outputLayerIndex - 1)->matrixifyActivatedValues()->transpose();

	Matrix* dC0_cw = new Matrix(hadamardProduct->getNumRows(), activatedPreviousLayer->getNumCols(), false);
	Operations::MultiplyMatrices(hadamardProduct, activatedPreviousLayer, dC0_cw);
	vector_dC0_cw.push_back(dC0_cw->transpose());

	// Calculate the new weight matrix between output layer and last hidden layer
	Matrix* newWeight = new Matrix(dC0_cw->getNumCols(), dC0_cw->getNumRows(), false);
	Operations::SubtractMatrices(weightMatrices.at(lastWeightMatrixIndex), dC0_cw->transpose(), newWeight); //TODO Multiply with learning rate
	newWeights.push_back(newWeight);
	

	// Move from last hidden to input layer
	for(int layer_index = outputLayerIndex - 1; layer_index > 0; layer_index--)
	{
		// Calculate dC0_da_L-1
		Matrix* weightRightOfCurrentLayer = weightMatrices.at(layer_index);
		Matrix* previousLayerError = vector_dC0_ca.back();
		derivedLayer = layers.at(layer_index + 1)->matrixifyDerivedValues();

		Operations::HadamardProduct(derivedLayer, previousLayerError, hadamardProduct);
		Matrix* thisLayerError = new Matrix(weightRightOfCurrentLayer->getNumRows(), hadamardProduct->getNumCols(), false);
		Operations::MultiplyMatrices(weightRightOfCurrentLayer, hadamardProduct, thisLayerError);
		vector_dC0_ca.push_back(thisLayerError);

		// Calculate dC0_cw_L-1

		// Check if next layer is input layer
		if(layer_index - 1 == 0)
		{
			activatedPreviousLayer = layers.at(layer_index - 1)->matrixifyValues()->transpose();
			derivedLayer = new Matrix(thisLayerError->getNumRows(), thisLayerError->getNumCols(), false);
		}
		else
		{
			activatedPreviousLayer = this->layers.at(layer_index)->matrixifyActivatedValues()->transpose();
			derivedLayer = layers.at(layer_index - 1)->matrixifyDerivedValues();
		}
		hadamardProduct = new Matrix(thisLayerError->getNumRows(), thisLayerError->getNumCols(), false);
		Operations::HadamardProduct(derivedLayer, thisLayerError, hadamardProduct);

		dC0_cw = new Matrix(hadamardProduct->getNumRows(), activatedPreviousLayer->getNumCols(), false);
		Operations::MultiplyMatrices(hadamardProduct, activatedPreviousLayer, dC0_cw);
		vector_dC0_cw.push_back(dC0_cw->transpose());

		newWeight = new Matrix(dC0_cw->getNumCols(), dC0_cw->getNumRows(), false);
		Operations::SubtractMatrices(weightMatrices.at(layer_index - 1), vector_dC0_cw.back(), newWeight);
		newWeights.push_back(newWeight);

		// Free up pointers
		delete weightRightOfCurrentLayer;
		delete previousLayerError;
		delete thisLayerError;
	}

	
	delete derivedLayer;
	delete hadamardProduct;
	delete activatedPreviousLayer;
	delete dC0_cw;


	reverse(newWeights.begin(), newWeights.end());
	weightMatrices = newWeights;
}
