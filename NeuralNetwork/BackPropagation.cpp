#include "NeuralNetwork.h"
#include "../Math/LinearAlgebra/Matrix.h"
#include "../Math/LinearAlgebra/Operations.h"

void NeuralNetwork::backPropagation()
{
	// Output layer
	int outputLayerIndex = this->topology.size() - 1;
	int lastWeightMatrixIndex = outputLayerIndex - 1;

	vector<Matrix*> vector_dC0_cw;
	vector<Matrix*> newWeights;
	vector<Matrix*> vector_dC0_ca;

	// Calculate the vector of derived cost functions
	Matrix* derivedCostFunction = new Matrix(this->layers.at(outputLayerIndex)->getSize(), 1, false);
	for(int r = 0; r < this->layers.at(outputLayerIndex)->getSize(); r++)
	{
		derivedCostFunction->setValue(r, 0, 2 * this->getErrors().at(r));
	}
	vector_dC0_ca.push_back(derivedCostFunction);
	Matrix* derivedLayer = this->layers.at(outputLayerIndex)->matrixifyDerivedValues()->transpose();

	Matrix* hadamardProduct;
	Operations::HadamardProduct(derivedLayer, derivedCostFunction, hadamardProduct);
	hadamardProduct->transpose();

	Matrix* activatedPreviousLayer = this->layers.at(outputLayerIndex - 1)->matrixifyActivatedValues()->transpose();

	Matrix* dC0_cw;
	Operations::MultiplyMatrices(activatedPreviousLayer, hadamardProduct, dC0_cw);
	vector_dC0_cw.push_back(dC0_cw);

	// Calculate the new weight matrix between output layer and last hidden layer
	Matrix* newWeight;
	Operations::SubtractMatrices(this->weightMatrices.at(lastWeightMatrixIndex), dC0_cw, newWeight);
	newWeights.push_back(newWeight);

	// Move from last hidden to input layer
	for(int layer_index = outputLayerIndex - 1; layer_index > 0; layer_index--)
	{
		// Calculate dC0_da_L-1
		Matrix* weightRightOfCurrentLayer = this->weightMatrices.at(layer_index);
		Matrix* previousLayerError = vector_dC0_ca.back();
		derivedLayer = this->layers.at(layer_index + 1)->matrixifyDerivedValues()->transpose();

		Operations::HadamardProduct(derivedLayer, previousLayerError, hadamardProduct);
		Matrix* thisLayerError;
		Operations::MultiplyMatrices(weightRightOfCurrentLayer, hadamardProduct, thisLayerError);
		vector_dC0_ca.push_back(thisLayerError);

		// Calculate dC0_cw_L-1

		// Check if next layer is input layer
		if(layer_index - 1 == 0)
		{
			activatedPreviousLayer = this->layers.at(layer_index - 1)->matrixifyValues()->transpose();
			derivedLayer = new Matrix(thisLayerError->getNumRows(), thisLayerError->getNumCols(), false, 1.0);
		}
		else
		{
			activatedPreviousLayer = this->layers.at(layer_index)->matrixifyActivatedValues()->transpose();
			derivedLayer = this->layers.at(layer_index - 1)->matrixifyDerivedValues()->transpose();
		}
		Operations::HadamardProduct(derivedLayer, thisLayerError, hadamardProduct);
		hadamardProduct->transpose();

		Operations::MultiplyMatrices(activatedPreviousLayer, hadamardProduct, dC0_cw);
		vector_dC0_cw.push_back(dC0_cw);

		Operations::SubtractMatrices(this->weightMatrices.at(layer_index - 1), dC0_cw, newWeight);
		newWeights.push_back(newWeight);

		// Free up pointers
		delete weightRightOfCurrentLayer;
		delete previousLayerError;
		delete thisLayerError;
	}

	delete derivedCostFunction;
	delete derivedLayer;
	delete hadamardProduct;
	delete activatedPreviousLayer;
	delete dC0_cw;
	delete newWeight;


	reverse(newWeights.begin(), newWeights.end());
	this->weightMatrices = newWeights;
}
