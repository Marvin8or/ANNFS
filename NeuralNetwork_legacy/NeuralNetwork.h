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
	CostFunctionType costFunction	    = COST_MSE;

	double errorOverAllOutputNeurons		= 0.00;
	double bias								= 1;
	double momentum;
	double learningRate;

	vector<int> topology;
	vector<Layer*> layers;
	vector<Matrix*> W;
	vector<Matrix*> deltas;
	vector<Matrix*> dC_dw;
	vector<Matrix*> dC_db;

	Matrix* outputErrors;
	Matrix* derivedOutputErrors;

	NeuralNetwork(
		vector<int> t,
		double lr	= 0.05,
		double m	= 1
	);

	//NeuralNetwork(
	//	vector<int> topology,
	//	ActivationFunc hiddenActivationType,
	//	ActivationFunc outputActivationType,
	//	CostFunctionType costFunction,
	//	double bias = 1,
	//	double learningRate = 0.05,
	//	double momentum = 1
	//);

	void train(
		vector<double> input,
		vector<double> target,
		double bias,
		double learningRate,
		double momentum);
	void setCurrentInput(vector<double> input);
	vector<double> getVurrentInput() { return _input; }

	void setCurrentTarget(vector<double> target) { _target = target; }
	vector<double> getCurrentTarget() { return _target; };

	void feedForward();
	void setErrors();
	void backPropagation();
	void gradientDescent();

	//vector<double> getErrors() { return errors; }
	//vector<double> getDerivedErrors() {	return derivedErrors;	}

	vector<double> getActivatedVals(int index) { return this->layers.at(index)->getActivatedValues(); }

	Matrix* getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyValues(); }
	Matrix* getBiasMatrix(int index) { return layers.at(index)->matrixifyBiasValues(); }
	Matrix* getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedValues(); }
	Matrix* getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedValues(); }
	Matrix* getWeightMatrix(int index) { return W.at(index); }

	void setNeuronValue(int layerIndex, int neuronIndex, double value) { this->layers.at(layerIndex)->setValue(neuronIndex, value); }

private:
	void setErrorMSE();
	vector<double> _input;
	vector<double> _target;

};