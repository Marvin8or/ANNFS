#include "NeuralNetwork/NeuralNetwork.h"
#include "../ThirdParty/json.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using json = nlohmann::json;
// TODO vector2D Matrix
// TODO Framework to load the data


// Define a struct to represent an MNIST image
struct MNISTImage {
    std::vector<double> label;            // Label of the digit (0 to 9)
    std::vector<double> pixels; // Pixel values of the image
};

// Function to read MNIST images from binary files
std::vector<MNISTImage> readMNISTImages(const std::string& imagePath,
                     const std::string& labelPath) 
{
    std::vector<MNISTImage> images; // Vector to store MNIST images

    // Read labels
    std::ifstream labelFile(labelPath, std::ios::binary); // Open label file in binary mode
    if (!labelFile) {
        std::cerr << "Error opening label file: " << labelPath << std::endl;
        return images; // Return empty vector if there's an error
    }

    int magicNumber;
    int numLabels;
    labelFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber)); // Read magic number
    labelFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));     // Read number of labels
    numLabels = _byteswap_ulong(numLabels); // Swap bytes (MNIST data is in big-endian format)

    // Read images
    std::ifstream imageFile(imagePath, std::ios::binary); // Open image file in binary mode
    if (!imageFile) {
        std::cerr << "Error opening image file: " << imagePath << std::endl;
        return images; // Return empty vector if there's an error
    }

    int magicNumberImages;
    int numImages;
    int numRows;
    int numCols;
    imageFile.read(reinterpret_cast<char*>(&magicNumberImages), sizeof(magicNumberImages)); // Read magic number for images
    imageFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));               // Read number of images
    imageFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));                   // Read number of rows
    imageFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));                   // Read number of columns
    numImages = _byteswap_ulong(numImages); // Swap bytes
    numRows = _byteswap_ulong(numRows);
    numCols = _byteswap_ulong(numCols);

    // Loop through each image in the dataset
    for (int i = 0; i < numImages; ++i) {
        MNISTImage mnistImage;

        mnistImage.label.resize(10, 0.0); // Initialize the label vector with 10 elements, all set to 0.0
        char label_indx;
        labelFile.read(reinterpret_cast<char*>(&label_indx), 1); // Read the label of the current image

        // Set the corresponding position to 1.0 in the one-hot encoded vector
        mnistImage.label[(int)label_indx] = 1.0;

        mnistImage.pixels.resize(numRows * numCols);                  // Resize the pixel vector

        // Read pixel values and store them in the vector
        imageFile.read(reinterpret_cast<char*>(mnistImage.pixels.data()), numRows * numCols);

        // Add the MNIST image to the vector
        images.push_back(mnistImage);
    }

    return images; // Return the vector of MNIST images
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


    const std::string trainImagePath = "C:/Users/Gabriel/Documents/Projects/ANNFS/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte";
    const std::string trainLabelPath = "C:/Users/Gabriel/Documents/Projects/ANNFS/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    // Read MNIST training images
    std::vector<MNISTImage> trainingImages = readMNISTImages(trainImagePath, trainLabelPath);

	/*
		Json file has all neccesary hyperparameters 
	*/

	vector2D trainingPixels;
	vector2D trainingLabels;

	for (int i = 0; i < 20; i++)
	{
		trainingPixels.push_back(trainingImages[i].pixels);
		trainingLabels.push_back(trainingImages[i].label);
	}

	auto json_file = openConfigurationFile(path_to_json);

	NeuralNetwork nn = NeuralNetwork(json_file);

	//TODO decide on how to implement train method when json file is configured
	nn.train(trainingPixels, trainingLabels, 1);
}

int main()
{
	json_file_nn_implementation_example();
}