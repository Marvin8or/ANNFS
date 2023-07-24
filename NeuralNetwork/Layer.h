#pragma once

#include "Neuron.h"
#include "../Math/LinearAlgebra/Matrix.h"
#include "assert.h"

using namespace std;
using namespace LinearAlgebra;
class Layer
{
public:
	Layer(int size);
	Layer(int size, ActivationFunc activation);
	Layer(int size, vector<ActivationFunc> activations);
	void setValue(int indexOfNeuron, double value);

	Matrix* matrixifyValues(); // Return nx1 matrix
	Matrix* matrixifyBiasValues(); 
	Matrix* matrixifyActivatedValues();
	Matrix* matrixifyDerivedValues();

	vector<Neuron*> getNeurons() { return this->neurons; }
	vector<double> getActivatedValues();
	void setNeurons(vector<Neuron*> neurons) { this->neurons = neurons; }
	int getSize() { return this->size; }
	void setSize(int size);
private:
	int size;
	vector<Neuron*> neurons;
	Matrix* biases;
};