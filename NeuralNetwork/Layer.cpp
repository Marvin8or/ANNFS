#include "Layer.h"
using namespace std;

Layer::Layer(int size)
{
	this->setSize(size);
	for(int i = 0; i < this->getSize(); i++)
	{
		this->neurons.push_back(new Neuron(1.00));
	}
	biases = new Matrix(getSize(), 1, true);
}

Layer::Layer(int size, ActivationFunc activation)
{
	this->setSize(size);
	for (int i = 0; i < this->getSize(); i++)
	{
		this->neurons.push_back(new Neuron(1.00, activation));
	}
	biases = new Matrix(getSize(), 1, true);
}

Layer::Layer(int size, vector<ActivationFunc> activations)
{
	this->setSize(size);
	for (int i = 0; i < this->getSize(); i++)
	{
		this->neurons.push_back(new Neuron(1.00, activations.at(i)));
	}
	biases = new Matrix(getSize(), 1, true);

}

void Layer::setValue(int indexOfNeuron, double value)
{
	if(indexOfNeuron < 0)
	{
		cerr << "Index is invalid: " << size << " < 0" << endl;
		assert(false);
	}
	neurons.at(indexOfNeuron)->setValue(value);
	
}

void Layer::setSize(int size)
{
	if (size <= 0)
	{
		cerr << "Number of neurons invalid: " << size << " <= 0" << endl;
		assert(false);
	}

	this->size = size;
}

Matrix* Layer::matrixifyValues()
{
	Matrix* m = new Matrix(this->getSize(), 1, false);
	for (int i = 0; i < this->getSize(); i++)
	{
		m->setValue(i, 0, this->getNeurons().at(i)->getValue());
	}
	return m;
}

Matrix* Layer::matrixifyBiasValues()
{
	Matrix* m = new Matrix(getSize(), 1, false);
	m->populate(0);
	return m;
}

Matrix* Layer::matrixifyActivatedValues()
{
	Matrix* m = new Matrix(this->getSize(), 1, false);
	for (int i = 0; i < this->getSize(); i++)
	{
		m->setValue(i, 0, this->getNeurons().at(i)->getActivatedValue());
	}
	return m;
}

Matrix* Layer::matrixifyDerivedValues()
{
	Matrix* m = new Matrix(this->getSize(), 1, false);
	for (int i = 0; i < this->getSize(); i++)
	{
		m->setValue(i, 0, this->getNeurons().at(i)->getDerivedValue());
	}
	return m;
}

vector<double> Layer::getActivatedValues()
{
	vector<double> activatedValues;
	for(int i = 0; i < this->getSize(); i++)
	{
		activatedValues.push_back(this->getNeurons().at(i)->getActivatedValue());
	}
	return activatedValues;
}



