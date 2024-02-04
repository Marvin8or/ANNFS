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

#endif /*UTILS_H*/