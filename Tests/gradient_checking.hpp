#ifndef GRADIENT_CHECKING_HPP
#define GRADIENT_CHECKING_HPP

#include <iostream>
#include <vector>
#include "../NeuralNetwork/NeuralNetwork.h"
#ifndef MATRIX_H
#include "../Math/LinearAlgebra/Matrix.h"
#endif // !MATRIX_H

#include "../ThirdParty/json.hpp"

using json = nlohmann::json;
void check_gradients(json json_configuration);
#endif // GRADIENT_CHECKING_HPP
