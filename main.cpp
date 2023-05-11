#include "Neuron.h"
#include "Matrix.h"
#include <iostream>

using namespace std;
int main()
{
	Neuron* neuron1 = new Neuron(0.5);
	cout <<"Neuron value: " << neuron1->getVal() << endl;
	cout <<"Neuron act. value: " << neuron1->getActivatedVal() << endl;
	cout <<"Neuron der. value: " << neuron1->getDerivedVal() << endl;

	Neuron* neuron2 = new Neuron(0.5, ReLU);
	cout << "Neuron value: " << neuron2->getVal() << endl;
	cout << "Neuron act. value: " << neuron2->getActivatedVal() << endl;
	cout << "Neuron der. value: " << neuron2->getDerivedVal() << endl;

	Matrix* matrix = new Matrix(3, 3, false, 10);
	matrix->printToConsole();

}