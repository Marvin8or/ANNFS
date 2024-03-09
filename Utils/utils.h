#ifndef UTILS_H
#define UTILS_H
#include <random>
#include <vector>

#ifndef INCLUDE_NLOHMANN_JSON_HPP_
#include "../ThirdParty/json.hpp"
#endif

#ifndef MATRIX_H
#include "../Math/LinearAlgebra/Matrix.h"
#endif

using vector2D = std::vector<std::vector<double>>;
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
//Matrix<double> minMaxNormalization(const Matrix<double>& input);
std::vector<double> minMaxScaler(const std::vector<double>& input);
//Matrix<double> l1Normalization(const Matrix<double>& input);
//vector2D l1Normalization(const vector2D& input);
#endif /*UTILS_H*/