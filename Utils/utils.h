#ifndef UTILS_H
#define UTILS_H
#include <random>
#include <vector>
#include "../ThirdParty/json.hpp"
using json = nlohmann::json;

struct MNISTImage {
    std::vector<double> label;
    std::vector<double> pixels;
};

double generateRandomNumber();
double generateHeRandomNumber(const int& input_size);
double generateGaussianRandomNumber();

std::vector<MNISTImage> readMNISTImages(const std::string& imagePath, const std::string& labelPath);
json openConfigurationFile(const std::string& path);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_synthetic_dataset(int num_samples, int num_input_features, int num_output_features, double noise);
#endif /*UTILS_H*/