#include "NeuralNetwork.h"
#include <cmath>
#include <numeric>



double sigmoidFunction(double x)
{
	return x / (1 + exp(-x));
}

double d_sigmoidFunction(double x)
{
	return sigmoidFunction(x) / (1 - sigmoidFunction(x));
}

Matrix<double> softmaxFunction(const Matrix<double>& input)
{
	if (!(input.getRows() == 1))
		throw std::invalid_argument("Input must have one row!");

	Matrix<double> softmax_values(1, input.getCols());

	double input_max_value = 0;
	for (int ci = 0; ci < input.getCols(); ci++)
	{
		if (input.get(0, ci) > input_max_value)
			input_max_value = input.get(0, ci);
	}

	double exp_sum = 0.0;
	for (int ci = 0; ci < input.getCols(); ci++)
	{
		exp_sum += std::exp(input.get(0, ci) - input_max_value);
	}

	for (int ci = 0; ci < input.getCols(); ci++)
	{
		double new_value = std::exp(input.get(0, ci) - input_max_value);
		softmax_values.put(0, ci, new_value / exp_sum);
	}

	return softmax_values;
}

Matrix<double> d_softmaxFunction(const Matrix<double>& input)
{
	Matrix<double> softmax_values = softmaxFunction(input);
	Matrix<double> d_softmax_values(1, input.getCols());
	for (int ci = 0; ci < input.getCols(); ci++)
	{
		d_softmax_values.put(0, ci, softmax_values.get(0, ci) / (1 - softmax_values.get(0, ci)));
	}
	return d_softmax_values;
}

double fastSigmoidFunction(double x)
{
	return x / (1 + abs(x));
}

double d_fastSigmoidFunction(double x)
{
	return fastSigmoidFunction(x) * (1 - fastSigmoidFunction(x));
}

double rectifiedLinearUnit(double x)
{
	if (x < 0) return 0;
	return x;
}

double d_rectifiedLinearUnit(double x)
{
	if (x < 0) return 0;
	return 1;
}

void NeuralNetwork::initialize_matrices()
{
#if DEBUG==ON
	for (uint i = 0; i < weightsNum; i++)
	{
		weights.push_back(Matrix<double>(topology_.at(i), topology_.at(i + 1), 1.0));
		deltaWeights.push_back(Matrix<double>(topology_.at(i), topology_.at(i + 1), 1.0));
	}

	for (uint i = 0; i < biasesNum; i++)
	{
		biases.push_back(Matrix<double>(1, topology_.at(i + 1), 1.0));
		deltaBiases.push_back(Matrix<double>(1, topology_.at(i + 1), 1.0));
		deltasBackprop.push_back(Matrix<double>(1, topology_.at(i + 1), 1.0));
	}

	for (uint i = 0; i < layerNum; i++)
	{
		neuronValues.push_back(Matrix<double>(1, topology_.at(i), 1.0));
		neuronValuesDerived.push_back(Matrix<double>(1, topology_.at(i), 1.0));
	}
	// Add target layer
	neuronValues.push_back(Matrix<double>(1, topology_.back(), 1.0));

#else
	for (uint i = 0; i < weightsNum; i++)
	{
		weights.push_back(Matrix<double>(topology_.at(i), topology_.at(i + 1)));
		deltaWeights.push_back(Matrix<double>(topology_.at(i), topology_.at(i + 1)));
	}

	for (uint i = 0; i < biasesNum; i++)
	{
		biases.push_back(Matrix<double>(1, topology_.at(i + 1)));
		deltaBiases.push_back(Matrix<double>(1, topology_.at(i + 1)));
		deltasBackprop.push_back(Matrix<double>(1, topology_.at(i + 1)));
	}

	for (uint i = 0; i < layerNum; i++)
	{
		neuronValues.push_back(Matrix<double>(1, topology_.at(i)));
		neuronValuesDerived.push_back(Matrix<double>(1, topology_.at(i)));
	}
	// Add target layer
	neuronValues.push_back(Matrix<double>(1, topology_.back()));

	// Setup output neurons pointer
	outputNeuronErrorsPtr = std::make_unique<Matrix<double>>(1, topology_.back());
#endif
}

void NeuralNetwork::print_predictions() const
{
	std::cout << "Predictions: ";
	std::cout << neuronValues[layerNum - 1];
}


NeuralNetwork::NeuralNetwork(const std::vector<uint>& topology, const double& learningRate, const ELossFunction& loss)
{
	if (!(topology.size() > 1))
		throw std::invalid_argument("Number of layers in topology is not sufficient!");

	if (!(learningRate > 0.0))
		throw std::invalid_argument("Learning rate must be greater than zero!");

	this->learningRate = learningRate;

	topology_.push_back(topology.at(0));

	for (auto layer : topology)
		topology_.push_back(layer);

	layerNum = topology_.size();
	weightsNum = topology_.size() - 1;
	biasesNum = topology_.size() - 1;

	initialize_matrices();

	switch (loss)
	{
	case MSE:
		outputNeuronErrorsFunc = LossFunctions::squared_error;
		outputNeuronErrorsFuncDerived = LossFunctions::squared_error_derived;
		compoundErrorFunc = LossFunctions::mean_squared_error;
		break;
	}

}

//TODO redefine uint typdef as uint32, or use already existing
NeuralNetwork::NeuralNetwork(const json& json_configuration)
{	
	json network_json_configuration = json_configuration["network"];

	uint input_size		= network_json_configuration["architecture"]["input_size"];
	auto hidden_layers  = network_json_configuration["architecture"]["hidden_layers"].get<std::vector<uint>>();
	uint output_size	= network_json_configuration["architecture"]["output_size"];

	double learningRate	= network_json_configuration["training"]["learning_rate"];

	auto activaton_function_names = network_json_configuration["architecture"]["activation_functions"].get<std::vector<std::string>>();
	
	for (std::string name : activaton_function_names)
	{
		if (name == "sigmoid")
		{
			activation_functions.push_back(EActivationFunctions::Sigmoid);
		}
		else if (name == "softmax")
		{
			activation_functions.push_back(EActivationFunctions::Softmax);
		}
		else if (name == "relu")
		{
			activation_functions.push_back(EActivationFunctions::ReLu);
		}
		else if(name == "fsigmoid")
		{
			activation_functions.push_back(EActivationFunctions::FSigmoid);
		}
	}

	auto loss	= network_json_configuration["training"]["loss_function"].get<std::string>();

	//default loss if not defined is MSE
	if (loss.empty())
		loss = "MSE";

	if (!(input_size > 0))
		throw std::invalid_argument("Number of neurons in input layer must be grater than zero!");

	topology_.push_back(input_size);

	for (auto layer : hidden_layers)
	{
		if (!(layer >= 0))
			throw std::invalid_argument("Number of neurons in hidden layer must be grater or equal to zero!");

		topology_.push_back(layer);
	}

	if (!(output_size > 0))
		throw std::invalid_argument("Number of neurons in ouput layer must be grater than zero!");

	topology_.push_back(output_size);

	if (!(activaton_function_names.size() == topology_.size() - 1))
		throw std::invalid_argument("Size of list of activation functions must match number of layers!");

	if (!(learningRate > 0.0))
		throw std::invalid_argument("Learning rate must be greater than zero!");

	this->learningRate = learningRate;
	layerNum = topology_.size();
	weightsNum = topology_.size() - 1;
	biasesNum = topology_.size() - 1;

	initialize_matrices();

	if (loss == "MSE")
	{
		outputNeuronErrorsFunc = LossFunctions::squared_error;
		outputNeuronErrorsFuncDerived = LossFunctions::squared_error_derived;
		compoundErrorFunc = LossFunctions::mean_squared_error;
	}
}

void NeuralNetwork::setInputValues(std::vector<double> inputs)
{
	if (!(inputs.size() == neuronValues.front().getCols()))
		throw std::invalid_argument("Invalid dimension for inputs!");

	Matrix<double> tmpInput(1, inputs.size());
	for (uint i = 0; i < inputs.size(); i++)
		tmpInput.put(0, i, *(inputs.begin() + i));
	neuronValues[0] = tmpInput;
}

void NeuralNetwork::setInputValues(const Matrix<double>& inputs)
{
	if (!(inputs.getCols() == neuronValues.front().getCols()))
		throw std::invalid_argument("Invalid dimension for inputs!");

	Matrix<double> tmpInput(1, inputs.getCols());
	for (uint i = 0; i < inputs.getCols(); i++)
		tmpInput.put(0, i, inputs.get(0, i));
	neuronValues[0] = tmpInput;
}

void NeuralNetwork::setTargetValues(std::vector<double> targets)
{
	if (!(targets.size() == neuronValues.back().getCols()))
		throw std::invalid_argument("Invalid dimension for targets!");

	Matrix<double> tmpTarget(1, targets.size());;
	for (uint i = 0; i < targets.size(); i++)
		tmpTarget.put(0, i, *(targets.begin() + i));
	neuronValues[neuronValues.size() - 1] = tmpTarget;
}

void NeuralNetwork::setTargetValues(const Matrix<double>& targets)
{
	if (!(targets.getCols() == neuronValues.back().getCols()))
		throw std::invalid_argument("Invalid dimension for targets!");

	Matrix<double> tmpTarget(1, targets.getCols());;
	for (uint i = 0; i < targets.getCols(); i++)
		tmpTarget.put(0, i, targets.get(0, i));
	neuronValues[neuronValues.size() - 1] = tmpTarget;
}

void NeuralNetwork::setWeights(uint indx, const Matrix<double>& weight)
{
	if (!(weight.getRows() == weights[indx].getRows()) || !(weight.getCols() == weights[indx].getCols()))
		throw std::invalid_argument("Weight Matrix you are trying to set has invalid dimensions!");
	Matrix<double> tmp(weight.getRows(), weight.getCols());
	for (int r = 0; r < tmp.getRows(); r++)
	{
		for (int c = 0; c < tmp.getCols(); c++)
		{
			tmp.put(r, c, weight.get(r, c));
		}
	}
	weights.at(indx) = tmp;
}

void NeuralNetwork::setWeights(const std::vector<Matrix<double>>& new_weights)
{
	std::vector<Matrix<double>> tmp_v;
	for (int i = 0; i < new_weights.size(); i++)
	{
		if (!(weights[i].getRows() == new_weights[i].getRows()) || !(weights[i].getCols() == new_weights[i].getCols()))
			throw std::invalid_argument("Weight Matrix you are trying to set has invalid dimensions!");

		Matrix<double> tmp_m(new_weights[i].getRows(), new_weights[i].getCols());
		for (int r = 0; r < tmp_m.getRows(); r++)
		{
			for (int c = 0; c < tmp_m.getCols(); c++)
			{
				tmp_m.put(r, c, new_weights[i].get(r, c));
			}
		}
		tmp_v.push_back(tmp_m);
	}
	weights = tmp_v;
}

void NeuralNetwork::setBiases(const std::vector<Matrix<double>>& new_biases)
{
	std::vector<Matrix<double>> tmp_v;
	for (int i = 0; i < new_biases.size(); i++)
	{
		if (!(biases[i].getRows() == new_biases[i].getRows()) || !(biases[i].getCols() == new_biases[i].getCols()))
			throw std::invalid_argument("Bias Matrix you are trying to set has invalid dimensions!");

		Matrix<double> tmp_m(new_biases[i].getRows(), new_biases[i].getCols());
		for (int r = 0; r < tmp_m.getRows(); r++)
		{
			for (int c = 0; c < tmp_m.getCols(); c++)
			{
				tmp_m.put(r, c, new_biases[i].get(r, c));
			}
		}
		tmp_v.push_back(tmp_m);
	}
	biases = tmp_v;
}

void NeuralNetwork::setInputValues(
	std::initializer_list<double> inputs,
	std::initializer_list<double> targets
)
{
	if (!(inputs.size() == neuronValues.front().getCols() && targets.size() == neuronValues.back().getCols()))
		throw std::invalid_argument("Invalid dimension for inputs or targets!");

	Matrix<double> tmpInput(1, inputs.size()), tmpTarget(1, targets.size());;
	for (uint i = 0; i < inputs.size(); i++)
		tmpInput.put(0, i, *(inputs.begin() + i));
	neuronValues[0] = tmpInput;

	for (uint i = 0; i < targets.size(); i++)
		tmpTarget.put(0, i, *(targets.begin() + i));
	neuronValues[neuronValues.size() - 1] = tmpTarget;
}

std::vector<Matrix<double>> NeuralNetwork::getDeltaWeights()
{
	std::vector<Matrix<double>> deltaWeights_copy(deltaWeights);
	return deltaWeights_copy;
}

std::vector<Matrix<double>> NeuralNetwork::getWeights()
{
	std::vector<Matrix<double>> weights_copy(weights);
	return weights_copy;
}

std::vector<Matrix<double>> NeuralNetwork::getDeltaBiases()
{
	std::vector<Matrix<double>> deltaBiases_copy(deltaBiases);
	return deltaBiases_copy;
}

std::vector<Matrix<double>> NeuralNetwork::getBiases()
{
	std::vector<Matrix<double>> biases_copy(biases);
	return biases_copy;
}

void NeuralNetwork::feedForward()
{
	for(uint i=0; i<layerNum - 1; i++)
	{
		Matrix<double> z = neuronValues[i].dot(weights[i]) + biases.at(i);
		Matrix<double> a = z;
		Matrix<double> a_d = z;

		switch (activation_functions[i])
		{
		case EActivationFunctions::ReLu:
			a = a.applyFunction(rectifiedLinearUnit);
			a_d = a_d.applyFunction(d_rectifiedLinearUnit);
			break;
		case EActivationFunctions::FSigmoid:
			a = a.applyFunction(fastSigmoidFunction);
			a_d = a_d.applyFunction(d_fastSigmoidFunction);
			break;
		case EActivationFunctions::Sigmoid:
			a = a.applyFunction(sigmoidFunction);
			a_d = a_d.applyFunction(d_sigmoidFunction);
			break;
		case EActivationFunctions::Softmax:
			a = a.applyFunction(softmaxFunction);
			a_d = a_d.applyFunction(d_softmaxFunction);
			break;
		}


#if DEBUG==ON
		std::cout << a << std::endl;
		std::cout << a_d << std::endl;
#endif
		neuronValues.at(i + 1) = a;
		neuronValuesDerived.at(i + 1) = a_d;
	}
}

void NeuralNetwork::setErrors()
{
#if DEBUG==ON
	Matrix<double> errors = outputNeuronErrorsFunc(neuronValues[neuronValues.size() - 1], neuronValues[neuronValues.size() - 2]);
	double error = compoundErrorsFunc(errors);
	std::cout << "ERRORS: " << std::endl;
	std::cout << errors << "\n" << std::endl;
	std::cout << "ERROR: " << std::endl;
	std::cout << error << "\n" << std::endl;
#else
	*outputNeuronErrorsPtr = outputNeuronErrorsFunc(neuronValues[neuronValues.size() - 1], neuronValues[neuronValues.size() - 2]);
	compoundError = compoundErrorFunc(*outputNeuronErrorsPtr);
#endif
}

void NeuralNetwork::backpropagation()
{
	uint L = layerNum - 1;
	Matrix<double> lastDelta = outputNeuronErrorsFuncDerived(neuronValues[neuronValues.size() - 2], neuronValues[neuronValues.size() - 1]) * neuronValuesDerived[L];

	Matrix<double> dC_dw = (neuronValues[L - 1].transpose());
	dC_dw = dC_dw.dot(lastDelta);

	deltasBackprop[L - 1]	 = lastDelta;
	deltaBiases[L - 1]		 = lastDelta;
	deltaWeights[L -1]		 = dC_dw;
#if DEBUG==ON
	std::cout << "dC_dw" << std::endl;
	std::cout << dC_dw << std::endl;
	std::cout << "delta" << std::endl;
	std::cout << lastDelta << std::endl;
#endif


	while (--L > 0)
	{
#if DEBUG==ON
		std::cout << weights[L] << std::endl;
		std::cout << deltasBackprop[L] << std::endl;
		std::cout << neuronValuesDerived[L] << std::endl;
#endif
		lastDelta = weights[L].dot(deltasBackprop[L].transpose()).transpose() * neuronValuesDerived[L];
		dC_dw = neuronValues[L - 1].transpose().dot(lastDelta);
		deltasBackprop[L - 1] = lastDelta;
		deltaBiases[L - 1] = lastDelta;
		deltaWeights[L - 1] = dC_dw;
#if DEBUG==ON
	std::cout << "dC_dw" << std::endl;
	std::cout << dC_dw << std::endl;
	std::cout << "delta" << std::endl;
	std::cout << lastDelta << std::endl;
#endif
	}


}

void NeuralNetwork::gradientDescent()
{
	uint L = layerNum - 1;
	while (L-- > 0)
	{
		weights[L] = weights[L] - (learningRate * deltaWeights[L]);
#if DEBUG==ON
		std::cout << weights[L] << std::endl;
#endif
		biases[L]  = biases[L] -  (learningRate * deltaBiases[L]);
	}
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
						  const std::vector<std::vector<double>>& targets,
						  const uint& nepochs)
{
	// Check if inputs and targets are not empty
	if (!(inputs.size() != 0 && targets.size() != 0))
		throw std::invalid_argument("Cannot pass empty 2d vectors!");

	// Check if inputs and targets have same number of rows
	if (!(inputs.size() == targets.size()))
		throw std::invalid_argument("inputs.rows() != targets.rows()");

	// Check if inputs have same number of columns as first layer fo nn
	if (!(inputs[0].size() == topology_[0]))
		throw std::invalid_argument("Number of columns in INPUT dataset doesnt match number of neurons in input layer");

	// Check if targets have same number of columns as last layer of nn
	if (!(targets[0].size() == topology_[layerNum - 1]))
		throw std::invalid_argument("Number of columns in INPUT dataset doesnt match number of neurons in input layer");

	auto examples = inputs.size();

	// iterate through each row
	for (uint epoch = 0; epoch < nepochs; epoch++)
	{
		std::cout << "Epoch: " << epoch + 1 << std::endl;

		for (auto example = 0; example < examples; example++)
		{
			// TODO make it verbose
			//std::cout << "example: " << example + 1 << std::endl;
			setInputValues(inputs[example]);
			setTargetValues(targets[example]);
			feedForward();
			setErrors();
			//std::cout << "error: " << compoundError << std::endl;
			compoundErrors.push_back(compoundError);
			backpropagation();
			gradientDescent();
		}
		double epochError = accumulate(compoundErrors.begin(), compoundErrors.end(), 0.0) / examples;

		std::cout << "Epoch Error: " << epochError << std::endl;
		epochErrors.push_back(epochError);
	}
}

vector2D NeuralNetwork::predict(const vector2D& inputs)
{
	// Check if inputs have same number of columns as first layer fo nn
	if (!(inputs[0].size() == topology_[0]))
		throw std::invalid_argument("Number of columns in INPUT dataset doesn't match number of neurons in input layer");

	vector2D predictions;

	for (auto example = 0; example < inputs.size(); example++)
	{
		setInputValues(inputs[example]);
		feedForward();
		std::vector<double> outputLayer = neuronValues[layerNum - 1].toVector();
		predictions.push_back(outputLayer);
	}
	return predictions;
}

void NeuralNetwork::summary() const
{
	std::cout << "Topology: ";
	for (uint i = 0; i < layerNum; i++)
	{
		std::cout << topology_.at(i) << " ";
	}

	for(uint i=0; i<layerNum; i++)
	{
		if(i == 0)
		{
			std::cout << "\nInput layer:" << std::endl;
			std::cout << "-------------------------------------" << std::endl;
			std::cout << neuronValues.at(i) << std::endl;
		}
		else
		{
			std::cout << "Hidden layer " << i << ":" << std::endl;
			std::cout << "-------------------------------------" << std::endl;
			std::cout << "X" << std::endl;
			std::cout << neuronValues.at(i) << std::endl;
			std::cout << "W" << std::endl;
			std::cout << weights.at(i - 1) << std::endl;
			std::cout << "B" << std::endl;
			std::cout << biases.at(i - 1) << std::endl;
		}
	}
}

