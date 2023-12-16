#include "NeuralNetwork/NeuralNetwork.h"
#include "../ThirdParty/json.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using json = nlohmann::json;
// TODO vector2D Matrix
// TODO Framework to load the data
// TODO implement gradient checking for backpropagation
// TODO implement proper way to initialize weight matrices
struct MNISTImage {
    std::vector<double> label;
    std::vector<double> pixels;
};

std::vector<MNISTImage> readMNISTImages(const std::string& imagePath, const std::string& labelPath) {
    std::vector<MNISTImage> images;

    // Read labels
    std::ifstream labelFile(labelPath, std::ios::binary);
    if (!labelFile) {
        std::cerr << "Error opening label file: " << labelPath << std::endl;
        return images;
    }

    int magicNumber;
    int numLabels;
    labelFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    labelFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    numLabels = _byteswap_ulong(numLabels);

    // Read images
    std::ifstream imageFile(imagePath, std::ios::binary);
    if (!imageFile) {
        std::cerr << "Error opening image file: " << imagePath << std::endl;
        return images;
    }

    int magicNumberImages;
    int numImages;
    int numRows;
    int numCols;
    imageFile.read(reinterpret_cast<char*>(&magicNumberImages), sizeof(magicNumberImages));
    imageFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    imageFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    imageFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    numImages = _byteswap_ulong(numImages);
    numRows = _byteswap_ulong(numRows);
    numCols = _byteswap_ulong(numCols);

    for (int i = 0; i < numImages; ++i) {
        MNISTImage mnistImage;
		unsigned char uclabel;
        labelFile.read(reinterpret_cast<char*>(&uclabel), 1);
		mnistImage.label.resize(10, 0.0);
		mnistImage.label[static_cast<size_t>(uclabel)] = 1.0;

		mnistImage.pixels.resize(numRows * numCols);
		std::vector<unsigned char> tmp;
		tmp.resize(numRows * numCols);
		imageFile.read(reinterpret_cast<char*>(tmp.data()), numRows * numCols);

		std::transform(tmp.begin(), tmp.end(), mnistImage.pixels.begin(),
			[](unsigned char c) { return static_cast<double>(c); });

        images.push_back(mnistImage);
    }

    return images;
}

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

int main()
{
	//initial_nn_implementation_example();
	json_file_nn_implementation_example();
}