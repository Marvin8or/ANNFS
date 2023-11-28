#include "NeuralNetwork/NeuralNetwork.h"
#include "../ThirdParty/json.hpp"
#include <iostream>
#include <fstream>

using json = nlohmann::json;
// TODO vector2D Matrix
// TODO Framework to load the data

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

	for (auto pred : predictions)
		std::cout << pred;

}

json openConfigurationFile(const std::string& path)
{
	
	std::ifstream jsonFile(path);

	if (!jsonFile.is_open())
	{
		std::cerr << "Error opening json file!!" << std::endl;
	}

	// Read the contents of the file into a string
	std::string jsonData((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());

	try {
		// Parse the JSON string
		json jsonObj = json::parse(jsonData);
		// Access the data in jsonObj as needed
		std::cout << "Parsing successful!\n";
		return jsonObj;

	}
	catch (const nlohmann::json::parse_error& e) {
		std::cerr << "JSON parsing error: " << e.what() << std::endl;
		std::cerr << "At offset: " << e.byte << std::endl;
	}


}

void json_file_nn_implementation_example()
{
	std::string path_to_json = "C:/Users/Gabriel/Documents/Projects/ANNFS/JsonFiles/example_conf_02.json";

	/*
		Json file has all neccesary hyperparameters 
	*/
	auto json_file = openConfigurationFile(path_to_json);

	NeuralNetwork nn = NeuralNetwork(json_file);

	//TODO decide on how to implement train method when json file is configured
	nn.train();
}

int main()
{
	json_file_nn_implementation_example();
}