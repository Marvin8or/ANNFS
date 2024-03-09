#include "NeuralNetwork/NeuralNetwork.h"
#include "Utils/utils.h"
#include "Tests/gradient_checking.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using json = nlohmann::json;
// TODO implement softmax and sigmoid activations and its derivatives
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

	nn.train(training_inputs, training_targets, epochs);

	auto testing_inputs = training_inputs;
	auto testing_targets = training_targets;
	auto predictions = nn.predict(testing_inputs);

	/*for (auto pred : predictions)
		std::cout << pred;*/

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

	for (int i = 0; i < 20; i++)
	{
		std::vector<double> scaled_pixels = minMaxScaler(trainingImages[i].pixels);

		trainingPixels.push_back(scaled_pixels);
		trainingLabels.push_back(trainingImages[i].label);
	}
	for (int i = 20; i < 25; i++)
	{
		std::vector<double> scaled_pixels = minMaxScaler(trainingImages[i].pixels);

		testPixels.push_back(scaled_pixels);
		testLabels.push_back(trainingImages[i].label);
	}

	auto json_file = openConfigurationFile(path_to_json);

	NeuralNetwork nn = NeuralNetwork(json_file);

	//TODO decide on how to implement train method when json file is configured
	nn.train(trainingPixels, trainingLabels, 2);

	vector2D predictions = nn.predict(testPixels);
	for (uint pi = 0; pi < predictions.size(); pi++)
	{
		std::cout << "Actual values" << std::endl;
		std::cout << "=============" << std::endl;
		std::cout << "Example " << pi + 1 << std::endl;
		for (uint i = 0; i < testLabels[pi].size(); i++)
			std::cout << testLabels[pi][i] << " ";
		std::cout << "\n" << std::endl;

		std::cout << "Predictions";
		std::cout << "=============" << std::endl;
		std::cout << "Example " << pi + 1 << std::endl;
		for (uint i = 0; i < predictions[pi].size(); i++)
			std::cout << predictions[pi][i] << " ";
		std::cout << "\n" << std::endl;
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

void simple_dataset_example()
{
	// Generate synthetic dataset
	int num_samples_train = 80;
	int num_samples_test = 20;
	int num_features = 3; // Number of features per sample
	double noise = 0.5;
	auto training_dataset = generate_synthetic_dataset(num_samples_train, num_features, 1, noise);
	auto testing_dataset = generate_synthetic_dataset(num_samples_test, num_features, 1, noise);
	std::vector<std::vector<double>> X_train = training_dataset.first;
	std::vector<std::vector<double>> y_train = training_dataset.second;

	std::vector<std::vector<double>> X_test = testing_dataset.first;
	std::vector<std::vector<double>> y_test = testing_dataset.second;

	// Output the generated dataset
	//std::cout << "Generated Dataset:" << std::endl;
	//for (int i = 0; i < num_samples; ++i) {
	//	std::cout << "Sample " << i << ": [ ";
	//	for (int j = 0; j < num_features; ++j) {
	//		std::cout << X[i][j];
	//		if (j != num_features - 1)
	//			std::cout << ", ";
	//	}
	//	std::cout << " ]" << " y = " << y[i] << std::endl;
	//}

	std::vector<uint> topology{ 3, 3, 2, 1 };
	NeuralNetwork nn = NeuralNetwork(
		topology,
		0.01,
		MSE
	);

	int epochs = 50;
	nn.train(X_train, y_train, epochs);

	auto predictions = nn.predict(X_test);

	//for (int i = 0; i < y_test.size(); i++)
	//	std::cout << predictions[i];

}
int main()
{
	//initial_nn_implementation_example();
	json_file_nn_implementation_example();
	//gradient_checking_implementation();
	//simple_dataset_example();
}