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
Neuron::Neuron(double newValue)
{
	value = newValue;
	// Set default activation function to FSF
	activationFunction = &fastSigmoidFunction;
	d_activationFunction = &d_fastSigmoidFunction;
	activate();
	derive();
}

Neuron::Neuron(double newValue, ActivationFunc activation)
{
	value = newValue;
	switch (activation)
	{
	case 1:
		activationFunction = &fastSigmoidFunction;
		d_activationFunction = &d_fastSigmoidFunction;
		break;
	case 2:
		activationFunction = &rectifiedLinearUnit;
		d_activationFunction = &d_rectifiedLinearUnit;
		break;
	}
	activate();
	derive();
}

void Neuron::activate()
{
	activatedValue = activationFunction(getValue());
}

void Neuron::derive()
{
	derivedValue = d_activationFunction(getValue());
}

void Neuron::setValue(double newValue)
{
	value = newValue;
	activatedValue = activationFunction(newValue);
	derivedValue = d_activationFunction(newValue);
}


