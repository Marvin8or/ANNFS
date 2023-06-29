#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <tuple>
#include <assert.h>
using namespace std;
namespace LinearAlgebra
{
	class Matrix
	{
	public:
		Matrix(int rows, int columns, bool isRandom, double default_value = 0.00);
		Matrix(Matrix& matrix);

		Matrix* transpose();
		Matrix* copy();

		void populate(double value); //Pass lambda
		void setValue(int row, int column, double value) { this->values.at(row).at(column) = value; }
		double getValue(int row, int column) { return this->values.at(row).at(column); }
		int getNumRows() { return this->numRows; }
		int getNumCols() { return this->numCols; }
		tuple<int, int> getShape();
		void printToConsole();

	private:
		int numRows;
		int numCols;
		double generateRandomNumber();
		vector<vector<double>> values;


	};
}
