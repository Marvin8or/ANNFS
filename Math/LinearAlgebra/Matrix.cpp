#include "Matrix.h"

LinearAlgebra::Matrix::Matrix(int rows, int columns, bool isRandom, double default_value)
{
	this->numRows = rows;
	this->numCols = columns;

	for (int i = 0; i < this->getNumRows(); i++)
	{
		vector<double> colValues;
		for (int j = 0; j < this->getNumCols(); j++)
		{
			double r = isRandom == true ? this->generateRandomNumber() : default_value;
			colValues.push_back(r);
		}
		this->values.push_back(colValues);
	}
}

tuple<int, int> LinearAlgebra::Matrix::getShape()
{
	tuple<int, int> t(this->getNumRows(), this->getNumCols());
	return t;
}

void LinearAlgebra::Matrix::printToConsole()
{
	for (int i = 0; i < this->getNumRows(); i++)
	{
		for (int j = 0; j < this->getNumCols(); j++)
		{
			std::cout << this->getValue(i, j) << "\t";
		}
		std::cout << std::endl;
	}
}

LinearAlgebra::Matrix* LinearAlgebra::Matrix::transpose()
{
	Matrix* m_t = new Matrix(this->getNumCols(), this->getNumRows(), false);
	for (int i = 0; i < this->getNumRows(); i++)
	{
		for (int j = 0; j < this->getNumCols(); j++)
		{
			m_t->setValue(j, i, this->getValue(i, j));
		}
	}
	return m_t;
}

LinearAlgebra::Matrix* LinearAlgebra::Matrix::copy()
{
	Matrix* m_t = new Matrix(this->getNumCols(), this->getNumRows(), false);
	for (int i = 0; i < this->getNumRows(); i++)
	{
		for (int j = 0; j < this->getNumCols(); j++)
		{
			m_t->setValue(i, j, this->getValue(i, j));
		}
	}
	return m_t;
}

void LinearAlgebra::Matrix::populate(double value)
{
	for (int i = 0; i < this->getNumRows(); i++)
	{
		for (int j = 0; j < this->getNumCols(); j++)
		{
			this->setValue(i, j, value);
		}
	}
}
double LinearAlgebra::Matrix::generateRandomNumber()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distribution(-0.0001, 0.0001);
	return distribution(gen);
}

