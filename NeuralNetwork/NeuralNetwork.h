#pragma once
#include "Neuron.h"
#include "Layer.h"
#include "../Math/LinearAlgebra/Matrix.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <time.h>

using namespace std;

enum CostFunctionType
{
	COST_MSE = 1
};

class NeuralNetwork
{
public:
	int topologySize;
	ActivationFunc hiddenActivationType = FSF;
	ActivationFunc outputActivationType = ReLU;
	CostFunctionType costFunction = COST_MSE;

	double error		= 0;
	double bias			= 1;
	double momentum;
	double learningRate;

	vector<int> topology;
	vector<Layer*> layers;
	vector<Matrix*> weightMatrices;
	vector<Matrix*> gradientMatrices;

	vector<double> input;
	vector<double> target;
	vector<double> errors;
	vector<double> derivedErrors;

	NeuralNetwork(
		vector<int> topology,
		double bias = 1,
		double learningRate = 0.05,
		double momentum = 1
	);

	NeuralNetwork(
		vector<int> topology,
		ActivationFunc hiddenActivationType,
		ActivationFunc outputActivationType,
		CostFunctionType costFunction,
		double bias = 1,
		double learningRate = 0.05,
		double momentum = 1
	);

	void setCurrentInput(vector<double> input);
	void setCurrentTarget(vector<double> target) { this->target = target; }

	void feedForward();
	void backPropagation();
	void setErrors();

	vector<double> getActivatedVals(int index) { return this->layers.at(index)->getActivatedValues(); }

	Matrix* getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyValues(); }
	Matrix* getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedValues(); }
	Matrix* getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedValues(); }
	Matrix* getWeightMatrix(int index) { return new Matrix(*this->weightMatrices.at(index)); }

	void setNeuronValue(int layerIndex, int neuronIndex, double value) { this->layers.at(layerIndex)->setValue(neuronIndex, value); }

private:
	void setErrorMSE();

};