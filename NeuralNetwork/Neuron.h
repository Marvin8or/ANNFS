#pragma once
#include <iostream>
#include <math.h>
using namespace std;

enum ActivationFunc
{
	FSF = 1, // Fast Sigmoid Function
	ReLU = 2
};

enum DerivedActivationFuncs
{
	d_FSF = 1,
	d_ReLU = 2
};

double fastSigmoidFunction(double x); //Static
double d_fastSigmoidFunction(double x);

double rectifiedLinearUnit(double x);
double d_rectifiedLinearUnit(double x);

class Neuron
{
public:
	Neuron(double value);
	Neuron(double value, ActivationFunc activation);

	void activate();
	void derive();

	//getters
	double getValue() { return this->val; }
	double getActivatedValue() { return this->activatedValue; }
	double getDerivedValue() { return this->derivedValue; }

	// Setter
	void setValue(double val) { this->val = val; };

private:
	// The non activated neuron value
	double val;

	double bias;

	double activatedValue;

	double derivedValue;

	double (*activationFunction)(double x);
	double (*d_activationFunction)(double x);

	
};