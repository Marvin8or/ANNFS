#include "Neuron.h"

double fastSigmoidFunction(double x)
{
	return x / (1 + abs(x));
}

double d_fastSigmoidFunction(double x)
{
	return fastSigmoidFunction(x) * (1 - fastSigmoidFunction(x));
}

double rectifiedLinearUnit(double x)
{
	if (x < 0) return 0;
	return x;
}

double d_rectifiedLinearUnit(double x)
{
	if (x < 0) return 0;
	return 1;
}
Neuron::Neuron(double value)
{
	this->val = value;
	// Set default activation function to FSF
	this->activationFunction = &fastSigmoidFunction;
	this->d_activationFunction = &d_fastSigmoidFunction;
	this->activate();
	this->derive();
}

Neuron::Neuron(double value, ActivationFunc activation)
{
	this->val = value;
	switch (activation)
	{
	case 1:
		this->activationFunction = &fastSigmoidFunction;
		this->d_activationFunction = &d_fastSigmoidFunction;
		break;
	case 2:
		this->activationFunction = &rectifiedLinearUnit;
		this->d_activationFunction = &d_rectifiedLinearUnit;
		break;
	}
	this->activate();
	this->derive();
}

void Neuron::activate()
{
	this->activatedValue = this->activationFunction(this->getValue());
}

void Neuron::derive()
{
	this->derivedValue = this->d_activationFunction(this->getValue());
}
