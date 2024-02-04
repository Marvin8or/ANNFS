#include "utils.h"
#include <set>
#include <iostream>
#include <fstream>

double generateRandomNumber()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distribution(-1, 1);
	return distribution(gen);
}

double generateHeRandomNumber(const int& input_size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, sqrt(2.0 / input_size));
    return distribution(generator);
}

double generateGaussianRandomNumber()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.1, 0.1);
    return distribution(generator);
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