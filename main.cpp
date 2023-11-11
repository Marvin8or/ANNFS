#include "NeuralNetwork/NeuralNetwork.h"
#include "LossFunctions.h"


// TODO vector2D Matrix
int main()
{
	std::vector<uint> topology{ 4, 3, 2 };
	NeuralNetwork nn = NeuralNetwork(
		topology,
		0.01,
		MSE
	);
	//nn.summary();
	
	int epochs = 700;

	//auto training_inputs = vector2D{ {0.264983 ,0.561863 ,0.124875 ,0.893281},
	//						{0.731487 ,0.665182 ,0.944304 ,0.780735},
	//						{0.488565 ,0.319929 ,0.057103 ,0.119785},
	//						{0.642041 ,0.936592 ,0.826147 ,0.317345},
	//						{0.184974 ,0.702384 ,0.872834 ,0.991872}};


	auto training_inputs = vector2D{ {1.2, 0.8, 0.5, 1.0},
									 { 0.4, 0.3, 0.9, 0.2},
									 { 0.9, 0.6, 0.7, 0.5},
									 { 0.2, 0.5, 0.3, 0.8},
									 { 0.7, 1.0, 0.4, 0.6} };

	//auto training_targets = vector2D{ {0.524632 ,0.189419},
	//						{0.432819 ,0.811143},
	//						{0.675018 ,0.342973},
	//						{0.159135 ,0.783314},
	//						{0.952368 ,0.493812}};

	auto training_targets = vector2D{ {0, 1},		// Class 1
									  {1, 0},		// Class 2
									  {0, 1},		// Class 1
									  {1, 0},		// Class 2
									  {0, 1} };		// Class 1

	nn.train(training_inputs, training_targets, epochs);

	auto testing_inputs = training_inputs;
	auto testing_targets = training_targets;
	auto predictions = nn.predict(testing_inputs);

	for (auto pred : predictions)
		std::cout << pred;

	//nn.setInputValues({ 1, 2, 3, 4 }, {200, 300});
	//nn.feedForward();
	//nn.setErrors();
	//nn.backpropagation();
	//nn.gradientDescent();
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