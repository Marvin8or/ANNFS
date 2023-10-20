#include "NeuralNetwork/NeuralNetwork.h"
#include "LossFunctions.h"

int main()
{
	std::vector<uint> topology{ 4, 3, 2 };
	NeuralNetwork nn = NeuralNetwork(
		topology,
		0.01,
		MSE
	);
	nn.setInputValues({ 1, 2, 3, 4 }, {200, 300});
	nn.feedForward();
	nn.setErrors();
	//nn.summary();
	nn.backpropagation();
	nn.gradientDescent();
	//nn.print_predictions();

	//nn.setInputValues({ 2, 3, 4, 5 }, { 200, 300});
	//nn.feedForward();
	//nn.setErrors();
	//nn.backpropagation();
	//nn.gradientDescent();
	//nn.print_predictions();
	//NeuralNetwork nn = NeuralNetwork(
	//	learningRate,
	//	momtentum
	//);

	//nn.addLayer(20);
	//nn.addLayer(10);
	//nn.addLayer(2);

	//NeuralNetwork nn = NeuralNetwork(
	//		jsonFile
	//);

	//nn.train(input_data, output_data)
}