#include "NeuralNetwork/NeuralNetwork.h"
#include "Utils/utils.h"
#include "Tests/gradient_checking.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using json = nlohmann::json;
// TODO vector2D Matrix
// TODO Framework to load the data
// TODO implement gradient checking for backpropagation
// TODO implement proper way to initialize weight matrices


void initial_nn_implementation_example()
{
	std::vector<uint> topology{ 4, 3, 2 };
	NeuralNetwork nn = NeuralNetwork(
		topology,
		0.01,
		MSE
	);
	nn.summary();
	int epochs = 700;
	auto training_inputs = vector2D{ {1.2, 0.8, 0.5, 1.0},
									 { 0.4, 0.3, 0.9, 0.2},
									 { 0.9, 0.6, 0.7, 0.5},
									 { 0.2, 0.5, 0.3, 0.8},
									 { 0.7, 1.0, 0.4, 0.6} };

	auto training_targets = vector2D{ {0, 1},		// Class 1
									  {1, 0},		// Class 2
									  {0, 1},		// Class 1
									  {1, 0},		// Class 2
									  {0, 1} };		// Class 1

	nn.train(training_inputs, training_targets, 3);

	auto testing_inputs = training_inputs;
	auto testing_targets = training_targets;
	auto predictions = nn.predict(testing_inputs);

	for (auto pred : predictions)
		std::cout << pred;

}

void json_file_nn_implementation_example()
{

	std::string path_to_json = "C:/Users/Gabriel/Documents/Projects/ANNFS/JsonFiles/example_conf_02.json";


    const std::string trainImagePath = "C:/Users/Gabriel/Documents/Projects/ANNFS/data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte";
    const std::string trainLabelPath = "C:/Users/Gabriel/Documents/Projects/ANNFS/data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    // Read MNIST training images
    std::vector<MNISTImage> trainingImages = readMNISTImages(trainImagePath, trainLabelPath);

	/*
		Json file has all neccesary hyperparameters 
	*/

	vector2D trainingPixels;
	vector2D trainingLabels;
	vector2D testPixels;
	vector2D testLabels;

	for (int i = 0; i < 10; i++)
	{
		trainingPixels.push_back(trainingImages[i].pixels);
		trainingLabels.push_back(trainingImages[i].label);
	}
	for (int i = 5; i < 8; i++)
	{
		testPixels.push_back(trainingImages[i].pixels);
		testLabels.push_back(trainingImages[i].label);
	}

	auto json_file = openConfigurationFile(path_to_json);

	NeuralNetwork nn = NeuralNetwork(json_file);

	//TODO decide on how to implement train method when json file is configured
	nn.train(trainingPixels, trainingLabels, 2);

	std::vector<Matrix<double>> predictions = nn.predict(testPixels);
	for (Matrix<double> prediction : predictions)
	{
		std::cout << prediction;
	}
}

void gradient_checking_implementation()
{
	std::string path_to_json = "C:/Users/Gabriel/Documents/Projects/ANNFS/JsonFiles/example_conf_02.json";
	

	json configuration = openConfigurationFile(path_to_json);
	//const std::string trainImagePath = "C:/Users/Gabriel/Documents/Projects/ANNFS/data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte";
	//const std::string trainLabelPath = "C:/Users/Gabriel/Documents/Projects/ANNFS/data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
	for (int i = 0; i < 5; i++)
	{

		check_gradients(configuration);
	}
}
int main()
{
	//initial_nn_implementation_example();
	//json_file_nn_implementation_example();
	gradient_checking_implementation();
}