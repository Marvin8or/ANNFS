#include "NeuralNetwork/NeuralNetwork.h"
#include "LossFunctions.h"

int main()
{
	std::vector<uint> topology{ 4, 3, 2 };
	NeuralNetwork nn = NeuralNetwork(
		topology,
		MSE
	);
	nn.setInputValues({ 1, 2, 3, 4 }, {0, 1});
	nn.feedForward();
	nn.setErrors();
	nn.backpropagation();

	nn.setInputValues({ 2, 3, 4, 5 }, { 1, 2 });
	nn.feedForward();
	nn.setErrors();
	nn.backpropagation();
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