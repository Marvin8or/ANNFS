#include "Neuron.h"
#include "Math/LinearAlgebra/Matrix.h"
#include "Math/LinearAlgebra/Operations.h"
#include "NeuralNetwork.h"
#include <iostream>

using namespace std;
using namespace LinearAlgebra;
int main()
{

	double learningRate = 0.75;
	double momentum = 1;
	double bias = 0;
	vector<double> input{ 1, 0, 0};
	vector<double> target{ 1, 0, 0};

	vector<int> topology;

	topology.push_back(3); // Input Layer - No activation function
	topology.push_back(20);
	topology.push_back(15);
	topology.push_back(10);
	topology.push_back(5);
	topology.push_back(3);

	std::unique_ptr<NeuralNetwork> n = std::make_unique<NeuralNetwork>(
		topology,
		learningRate,
		momentum);

	for (int i = 0; i < n->layers.at(topology.size() - 1)->getSize(); i++) {
		std::cout << "Neuron " << i << ": " << n->layers.at(topology.size() - 1)->getActivatedValues().at(i) << std::endl;
	}

	for(int step_i = 0; step_i < 1000; step_i++)
	{
		//n->setCurrentInput(input);
		//n->setCurrentTarget(target);
		std::cout << "Training step " << step_i << std::endl;
		n->train(
			input,
			target,
			bias,
			learningRate,
			momentum);

		std::cout << "Error: " << n->errorOverAllOutputNeurons << std::endl;

	}
	for (int i = 0; i < n->layers.at(topology.size() -1)->getSize(); i++) {
		std::cout <<"Neuron " << i << ": " << n->layers.at(topology.size() - 1)->getActivatedValues().at(i) << std::endl;
	}
}