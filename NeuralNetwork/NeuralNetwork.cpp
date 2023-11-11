#include "NeuralNetwork.h"
#include <cmath>

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

void NeuralNetwork::setInputValues(std::vector<double> inputs)
{
	if (!(inputs.size() == neuronValues.front().getCols()))
		throw std::invalid_argument("Invalid dimension for inputs!");

	Matrix<double> tmpInput(1, inputs.size());
	for (uint i = 0; i < inputs.size(); i++)
		tmpInput.put(0, i, *(inputs.begin() + i));
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

void NeuralNetwork::feedForward()
{
	for(uint i=0; i<layerNum - 1; i++)
	{
		Matrix<double> z = neuronValues[i].dot(weights[i]) + biases.at(i);

		Matrix<double> a = z;
		Matrix<double> a_d = z;
		a = a.applyFunction(rectifiedLinearUnit);
		a_d = a_d.applyFunction(d_rectifiedLinearUnit);
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
	//std::cout << *outputNeuronErrorsPtr;
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
			setInputValues(inputs[example]);
			setTargetValues(targets[example]);
			feedForward();
			setErrors();
			backpropagation();
			gradientDescent();
		}
		std::cout << "Epoch Error: " << compoundError << std::endl;
		epochErrors.push_back(compoundError);
		//print_predictions();
	}
}

std::vector<Matrix<double>> NeuralNetwork::predict(const vector2D& inputs)
{
	// Check if inputs have same number of columns as first layer fo nn
	if (!(inputs[0].size() == topology_[0]))
		throw std::invalid_argument("Number of columns in INPUT dataset doesnt match number of neurons in input layer");

	std::vector<Matrix<double>> predictions;

	for (auto example = 0; example < inputs.size() - 1; example++)
	{
		setInputValues(inputs[example]);
		feedForward();
		Matrix<double> tmp = neuronValues[layerNum - 1];
		predictions.push_back(tmp);
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

