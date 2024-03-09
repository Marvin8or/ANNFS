#include "utils.h"
#include <set>
#include <iostream>
#include <fstream>

double generateRandomNumber()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distribution(0, 1);
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

// Generate a synthetic dataset based on a simple mathematical function with noise
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_synthetic_dataset(int num_samples, int num_input_features, int num_output_features, double noise) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Initialize vectors to store input features (X) and target labels (y)
    std::vector<std::vector<double>> X(num_samples, std::vector<double>(num_input_features));
    std::vector<std::vector<double>> y(num_samples, std::vector<double>(num_output_features));

    // Generate synthetic dataset
    for (int i = 0; i < num_samples; ++i) {
        // Generate random input features
        for (int j = 0; j < num_input_features; ++j) {
            X[i][j] = dist(gen);
        }

        for (int j = 0; j < num_output_features; ++j)
        {
            double y_true = 0;
            for (int j = 0; j < num_input_features; ++j) {
                y_true += 2 * X[i][j] * X[i][j] + X[i][j] + 1;
            }
            // Add Gaussian noise to the function
            std::normal_distribution<double> noise_dist(0.0, noise);
            double noise_value = noise_dist(gen);
            y[i][j] = y_true + noise_value;
        }
    }

    // Return the generated dataset
    return std::make_pair(X, y);
}

//Matrix<double> minMaxNormalization(const Matrix<double>& input)
//{
//    Matrix<double> normalizedMatrix(input.getRows(), input.getCols());
//    double max_value = 0;
//    double min_value = 0;
//
//    for (uint ri = 0; ri < input.getRows(); ri++)
//    {
//        for (uint ci = 0; ci < input.getCols(); ci++)
//        {
//            double current_value = input.get(ri, ci);
//            if (current_value > max_value)
//                max_value = current_value;
//            if (current_value < min_value)
//                min_value = current_value;
//        }
//    }
//
//    for (uint ri = 0; ri < input.getRows(); ri++)
//    {
//        for (uint ci = 0; ci < input.getCols(); ci++)
//        {
//            double current_value = input.get(ri, ci);
//            double new_value = (current_value - min_value) / (max_value - min_value);
//            normalizedMatrix.put(ri, ci, new_value);
//        }
//    }
//
//    return normalizedMatrix;
//}

std::vector<double> minMaxScaler(const std::vector<double>& input)
{
    uint input_size = input.size();
    vector2D normalizedInput;
    double max_value = 0;
    double min_value = 0;

    for (uint i = 0; i < input_size; i++)
    {
        double current_value = input[i];
        if (current_value > max_value)
            max_value = current_value;
        if (current_value < min_value)
            min_value = current_value;

    }

    std::vector<double> normalized_input;
    for (uint i = 0; i < input_size; i++)
    {
        double current_value = input[i];
        double new_value = (current_value - min_value) / (max_value - min_value);
        normalized_input.push_back(new_value);
    }

    return normalized_input;
}

//Matrix<double> l1Normalization(const Matrix<double>& input)
//{
//    uint rows = input.getRows();
//    uint cols = input.getCols();
//    Matrix<double> normalizedInput(rows, cols);
//    double abs_sum_denom = 0;
//
//
//    for (uint ri = 0; ri < rows; ri++)
//    {
//        for (uint ci = 0; ci < cols; ci++)
//        {
//            double current_value = input.get(ri, ci);
//            abs_sum_denom += std::abs(current_value);
//        }
//    }
//
//    for (uint ri = 0; ri < rows; ri++)
//    {
//
//        for (uint ci = 0; ci < cols; ci++)
//        {
//            double current_value = input.get(ri, ci);
//            double new_value = current_value / abs_sum_denom;
//            normalizedInput.put(ri, ci, new_value);
//        }
//    }
//
//    return normalizedInput;
//}
//
//vector2D l1Normalization(const vector2D& input)
//{
//    uint rows = input.size();
//    uint cols = input[0].size();
//    vector2D normalizedInput;
//    double abs_sum_denom = 0;
//
//
//    for (uint ri = 0; ri < rows; ri++)
//    {
//        for (uint ci = 0; ci < cols; ci++)
//        {
//            double current_value = input[ri][ci];
//            abs_sum_denom += std::abs(current_value);
//        }
//    }
//
//    for (uint ri = 0; ri < rows; ri++)
//    {
//        std::vector<double> new_row;
//        for (uint ci = 0; ci < cols; ci++)
//        {
//            double current_value = input[ri][ci];
//            double new_value = current_value / abs_sum_denom;
//            new_row.push_back(new_value);
//        }
//        normalizedInput.push_back(new_row);
//    }
//
//    return normalizedInput;
//}