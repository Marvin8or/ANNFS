#include "utils.h"

double generateRandomNumber()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distribution(-1, 1);
	return distribution(gen);
}